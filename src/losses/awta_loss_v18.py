#!/usr/bin/env python3
"""Adaptive Winner-Takes-All Loss V18 with Uniform 80-Step ADE & Speed-Weighted Sample Loss.

Key Principles:
1. [Uniform 80-Step ADE]: Strictly equal weighting across all 80 timesteps (matching official challenge metric).
2. [Velocity-Proportional Sample Weighting]: w_sample = 1.0 + clamp(v_cur / 10.0, 0.0, 2.0) applied to sample loss.
3. [Type-Aware Soft Acceleration Hinge]: Soft penalty for vehicle acceleration beyond [-8.0, +4.0] m/s^2.
4. [Deep Supervision & Classification]: Full probability matching + Cross-Entropy on Stage 2 Logits (preserving 66%+ Pick Acc).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWTALossV18(nn.Module):
    def __init__(
        self,
        tau_start: float = 0.35,
        tau_mid: float = 0.20,
        tau_end: float = 0.08,
        top_m_start: int = 2,
        top_m_mid: int = 2,
        top_m_end: int = 1,
        tau_cls: float = 0.5,
        weight_stage1: float = 0.3,
        weight_stage2: float = 1.0,
        weight_reg: float = 1.0,
        weight_goal: float = 0.2,
        weight_cls: float = 0.3,
        weight_hard_cls: float = 0.2,
        weight_div: float = 0.05,
        div_margin: float = 2.5,
        weight_kin: float = 0.01,
        v0: float = 10.0,
    ):
        super().__init__()
        self.tau_start = tau_start
        self.tau_mid = tau_mid
        self.tau_end = tau_end
        self.top_m_start = top_m_start
        self.top_m_mid = top_m_mid
        self.top_m_end = top_m_end
        self.tau_cls = tau_cls
        self.weight_stage1 = weight_stage1
        self.weight_stage2 = weight_stage2
        self.weight_reg = weight_reg
        self.weight_goal = weight_goal
        self.weight_cls = weight_cls
        self.weight_hard_cls = weight_hard_cls
        self.weight_div = weight_div
        self.div_margin = div_margin
        self.weight_kin = weight_kin
        self.v0 = v0

    def get_temperature_and_top_m(self, epoch: int, total_epochs: int) -> tuple[float, int]:
        progress = min(1.0, max(0.0, epoch / max(1, total_epochs - 1)))
        if progress < 0.40:
            p_sub = progress / 0.40
            tau = self.tau_start * ((self.tau_mid / self.tau_start) ** p_sub)
            top_m = self.top_m_start
        elif progress < 0.75:
            p_sub = (progress - 0.40) / 0.35
            tau = self.tau_mid * ((0.12 / self.tau_mid) ** p_sub)
            top_m = self.top_m_mid
        else:
            p_sub = (progress - 0.75) / 0.25
            tau = 0.12 * ((self.tau_end / 0.12) ** p_sub)
            top_m = self.top_m_end

        top_m = max(self.top_m_end, min(self.top_m_start, top_m))
        return tau, top_m

    def _compute_awta_reg_loss(
        self,
        pred_traj: torch.Tensor,     # [B, K, T, 2]
        pred_goals: torch.Tensor,    # [B, K, 2]
        future_gt: torch.Tensor,     # [B, T, 2]
        gt_goal: torch.Tensor,       # [B, 2]
        mask: torch.Tensor,          # [B, 1, T]
        denom: torch.Tensor,         # [B, 1]
        sample_weights: torch.Tensor,# [B]
        tau: float,
        top_m: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, T, _ = pred_traj.shape
        device = pred_traj.device

        diff = pred_traj - future_gt.unsqueeze(1)
        disp = torch.linalg.vector_norm(diff, dim=-1)  # [B, K, T]
        ade_per_mode = (disp * mask).sum(dim=-1) / denom  # [B, K]

        goal_diff = pred_goals - gt_goal.unsqueeze(1)  # [B, K, 2]
        goal_dist_per_mode = torch.linalg.vector_norm(goal_diff, dim=-1)  # [B, K]

        mode_error = ade_per_mode + 0.3 * goal_dist_per_mode  # [B, K]

        sorted_errors, sorted_indices = torch.sort(mode_error, dim=-1)
        top_m_mask = torch.zeros_like(mode_error, dtype=torch.bool)
        top_m_indices = sorted_indices[:, :top_m]
        top_m_mask.scatter_(1, top_m_indices, True)

        masked_errors = mode_error.clone()
        masked_errors[~top_m_mask] = float("inf")
        wta_weights = F.softmax(-masked_errors / max(0.01, tau), dim=-1).detach()  # [B, K]

        gt_traj = future_gt.unsqueeze(1).expand_as(pred_traj)
        smooth_l1_traj = F.smooth_l1_loss(pred_traj, gt_traj, reduction="none").sum(dim=-1)  # [B, K, T]

        # Strictly uniform 80-step regression (no time distortion)
        step_reg_loss = (smooth_l1_traj * mask).sum(dim=-1) / denom  # [B, K]
        reg_per_sample = (wta_weights * step_reg_loss).sum(dim=-1)  # [B]
        loss_reg = (reg_per_sample * sample_weights).mean()

        gt_goals = gt_goal.unsqueeze(1).expand_as(pred_goals)
        smooth_l1_goal = F.smooth_l1_loss(pred_goals, gt_goals, reduction="none").sum(dim=-1)
        goal_per_sample = (wta_weights * smooth_l1_goal).sum(dim=-1)  # [B]
        loss_goal = (goal_per_sample * sample_weights).mean()

        return loss_reg, loss_goal, ade_per_mode, disp

    def _soft_kinematic_hinge(
        self,
        traj: torch.Tensor,      # [B, K, 80, 2]
        type_idx: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Soft penalty on vehicle acceleration beyond [-8.0, +4.0] m/s^2."""
        d_xy = traj[:, :, 1:, :] - traj[:, :, :-1, :]
        v = torch.linalg.vector_norm(d_xy, dim=-1) * 10.0  # [B, K, 79] in m/s
        a = (v[:, :, 1:] - v[:, :, :-1]) * 10.0            # [B, K, 78] in m/s^2

        is_vehicle = (type_idx == 0).unsqueeze(-1).unsqueeze(-1).float()  # [B, 1, 1]

        pos_excess = F.relu(a - 4.0)
        neg_excess = F.relu(-8.0 - a)
        acc_penalty = ((pos_excess + neg_excess) * is_vehicle).mean()
        return acc_penalty

    def forward(
        self,
        pred_traj_2: torch.Tensor,   # [B, K, T=80, 2] (Stage 2 Refined)
        pred_goals_2: torch.Tensor,  # [B, K, 2] (Stage 2 Refined)
        logits_2: torch.Tensor,      # [B, K] (Stage 2 Logits)
        pred_traj_1: torch.Tensor,   # [B, K, T=80, 2] (Stage 1 Proposals)
        pred_goals_1: torch.Tensor,  # [B, K, 2] (Stage 1 Proposals)
        future_gt: torch.Tensor,     # [B, T=80, 2]
        future_valid: torch.Tensor,  # [B, T=80]
        type_idx: torch.Tensor,      # [B]
        cur_speed: torch.Tensor,     # [B]
        epoch: int = 0,
        total_epochs: int = 15,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, K, T, _ = pred_traj_2.shape
        device = pred_traj_2.device

        mask = future_valid.unsqueeze(1).float()  # [B, 1, T]
        denom = mask.sum(dim=-1).clamp_min(1.0)    # [B, 1]

        t_ix = torch.arange(T, device=device).view(1, T).expand(B, T)
        last_valid_idx = torch.where(future_valid, t_ix, t_ix.new_zeros(())).max(dim=-1).values
        gt_goal = torch.gather(
            future_gt, 1, last_valid_idx[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)  # [B, 2]

        tau, top_m = self.get_temperature_and_top_m(epoch, total_epochs)

        # Velocity-proportional sample weights [B]
        sample_weights = 1.0 + (cur_speed / self.v0).clamp(0.0, 2.0)

        # 1. Stage 1 Proposal aWTA Loss
        loss_reg_1, loss_goal_1, ade_modes_1, _ = self._compute_awta_reg_loss(
            pred_traj_1, pred_goals_1, future_gt, gt_goal, mask, denom, sample_weights, tau, top_m
        )
        loss_stage1 = self.weight_reg * loss_reg_1 + self.weight_goal * loss_goal_1

        # 2. Stage 2 Refinement aWTA Loss
        loss_reg_2, loss_goal_2, ade_modes_2, _ = self._compute_awta_reg_loss(
            pred_traj_2, pred_goals_2, future_gt, gt_goal, mask, denom, sample_weights, tau, top_m
        )

        # 3. Stage 2 Mode Classification & Ranking Supervision (Essential for Pick Acc)
        target_probs = F.softmax(-ade_modes_2 / self.tau_cls, dim=-1).detach()
        log_probs = F.log_softmax(logits_2, dim=-1)
        loss_cls = -(target_probs * log_probs).sum(dim=-1).mean()

        winner_2 = ade_modes_2.argmin(dim=-1)
        loss_hard_cls = F.cross_entropy(logits_2, winner_2)

        # 4. Stage 2 Mode Diversity Loss
        goal_pairwise_dist = torch.cdist(pred_goals_2, pred_goals_2, p=2)
        eye_mask = torch.eye(K, device=device).bool().unsqueeze(0)
        diversity_dist = torch.clamp(goal_pairwise_dist, max=self.div_margin)
        diversity_dist = diversity_dist.masked_fill(eye_mask, 0.0)
        loss_div = -diversity_dist.sum(dim=(1, 2)).mean() / (K * (K - 1))

        # 5. Soft Kinematic Hinge Penalty
        loss_kin = self._soft_kinematic_hinge(pred_traj_2, type_idx)

        # Total Loss
        loss_stage2 = (
            self.weight_reg * loss_reg_2
            + self.weight_goal * loss_goal_2
            + self.weight_cls * loss_cls
            + self.weight_hard_cls * loss_hard_cls
            + self.weight_div * loss_div
            + self.weight_kin * loss_kin
        )

        total_loss = self.weight_stage1 * loss_stage1 + self.weight_stage2 * loss_stage2

        with torch.no_grad():
            top1_s2 = logits_2.argmax(dim=-1)
            pick_acc = (top1_s2 == winner_2).float().mean().item()
            min_ade6_s1 = ade_modes_1.min(dim=-1).values.mean().item()
            min_ade6_s2 = ade_modes_2.min(dim=-1).values.mean().item()
            ade1_s2 = ade_modes_2.gather(1, top1_s2.unsqueeze(-1)).squeeze(-1).mean().item()

        metrics = {
            "loss": total_loss.item(),
            "loss_s1": loss_stage1.item(),
            "loss_s2": loss_stage2.item(),
            "loss_cls": loss_cls.item(),
            "loss_kin": loss_kin.item(),
            "s1_ade6": min_ade6_s1,
            "s2_ade6": min_ade6_s2,
            "s2_ade1": ade1_s2,
            "pick_acc": pick_acc,
            "tau": tau,
            "top_m": top_m,
        }

        return total_loss, metrics
