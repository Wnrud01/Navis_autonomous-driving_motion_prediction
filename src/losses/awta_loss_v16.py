"""Adaptive Winner-Takes-All Loss V16 with Deep Supervision for Two-Stage Decoders.

Combines:
1. Stage 1 Proposal aWTA Loss: 3-Stage Annealing (Top-4 -> Top-2 -> Top-1, tau 1.5 -> 0.1)
   to ensure Stage 1 produces diverse, sharp 0.57m multi-modal trajectory proposals.
2. Stage 2 Refinement aWTA Loss: Supervises locally refined trajectories traj_2 and goals_2.
3. Multi-Future Aligned Classification + Winner CE on Stage 2 logits.
4. Total Loss = weight_stage1 * Loss_stage1 + Loss_stage2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZON_STEPS = (9, 19, 39, 59, 79)


class AdaptiveWTALossV16(nn.Module):
    def __init__(
        self,
        modes: int = 6,
        tau_start: float = 1.5,
        tau_mid: float = 0.5,
        tau_end: float = 0.1,
        tau_cls: float = 0.5,
        top_m_start: int = 4,
        top_m_mid: int = 2,
        top_m_end: int = 1,
        weight_stage1: float = 0.5,
        weight_reg: float = 1.0,
        weight_goal: float = 0.15,
        weight_cls: float = 0.8,
        weight_hard_cls: float = 0.5,
        weight_div: float = 0.08,
        div_margin: float = 3.0,
        horizon_weight_scale: float = 0.5,
    ):
        super().__init__()
        self.modes = modes
        self.tau_start = tau_start
        self.tau_mid = tau_mid
        self.tau_end = tau_end
        self.tau_cls = tau_cls
        self.top_m_start = top_m_start
        self.top_m_mid = top_m_mid
        self.top_m_end = top_m_end
        self.weight_stage1 = weight_stage1
        self.weight_reg = weight_reg
        self.weight_goal = weight_goal
        self.weight_cls = weight_cls
        self.weight_hard_cls = weight_hard_cls
        self.weight_div = weight_div
        self.div_margin = div_margin
        self.horizon_weight_scale = horizon_weight_scale

    def get_temperature_and_top_m(self, epoch: int, total_epochs: int) -> tuple[float, int]:
        """3-stage progressive annealing:
        - Stage 1 (0% ~ 35%): Top-4, tau: start -> mid
        - Stage 2 (35% ~ 70%): Top-2, tau: mid -> 0.25
        - Stage 3 (70% ~ 100%): Top-1, tau: 0.25 -> end
        """
        progress = min(1.0, max(0.0, epoch / max(1, total_epochs - 1)))
        if progress < 0.35:
            p_sub = progress / 0.35
            tau = self.tau_start * ((self.tau_mid / self.tau_start) ** p_sub)
            top_m = self.top_m_start
        elif progress < 0.70:
            p_sub = (progress - 0.35) / 0.35
            tau = self.tau_mid * ((0.25 / self.tau_mid) ** p_sub)
            top_m = self.top_m_mid
        else:
            p_sub = (progress - 0.70) / 0.30
            tau = 0.25 * ((self.tau_end / 0.25) ** p_sub)
            top_m = self.top_m_end

        top_m = max(self.top_m_end, min(self.top_m_start, top_m))
        return tau, top_m

    def _compute_awta_reg_loss(
        self,
        pred_traj: torch.Tensor,
        pred_goals: torch.Tensor,
        future_gt: torch.Tensor,
        gt_goal: torch.Tensor,
        mask: torch.Tensor,
        denom: torch.Tensor,
        tau: float,
        top_m: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute aWTA regression loss for a given stage's trajectory predictions."""
        B, K, T, _ = pred_traj.shape
        device = pred_traj.device

        # Step-wise distance & ADE per mode
        diff = pred_traj - future_gt.unsqueeze(1)
        disp = torch.linalg.vector_norm(diff, dim=-1)  # [B, K, T]
        ade_per_mode = (disp * mask).sum(dim=-1) / denom  # [B, K]

        # Goal distance per mode
        goal_diff = pred_goals - gt_goal.unsqueeze(1)  # [B, K, 2]
        goal_dist_per_mode = torch.linalg.vector_norm(goal_diff, dim=-1)  # [B, K]

        mode_error = ade_per_mode + 0.5 * goal_dist_per_mode  # [B, K]

        # Soft top-m WTA assignment
        sorted_errors, sorted_indices = torch.sort(mode_error, dim=-1)
        top_m_mask = torch.zeros_like(mode_error, dtype=torch.bool)
        top_m_indices = sorted_indices[:, :top_m]
        top_m_mask.scatter_(1, top_m_indices, True)

        masked_errors = mode_error.clone()
        masked_errors[~top_m_mask] = float("inf")
        weights = F.softmax(-masked_errors / max(0.01, tau), dim=-1).detach()  # [B, K]

        # Multi-Horizon Smooth L1 Trajectory Regression Loss
        gt_traj = future_gt.unsqueeze(1).expand_as(pred_traj)
        smooth_l1_traj = F.smooth_l1_loss(pred_traj, gt_traj, reduction="none").sum(dim=-1)  # [B, K, T]

        step_weights = torch.ones(T, device=device, dtype=smooth_l1_traj.dtype)
        if self.horizon_weight_scale > 0:
            for h_idx in HORIZON_STEPS:
                if h_idx < T:
                    step_weights[h_idx] += self.horizon_weight_scale

        wmask = mask * step_weights.view(1, 1, T)
        wdenom = wmask.sum(dim=-1).clamp_min(1.0)
        step_reg_loss = (smooth_l1_traj * wmask).sum(dim=-1) / wdenom  # [B, K]
        loss_reg = (weights * step_reg_loss).sum(dim=-1).mean()

        # Goal Regression Loss
        gt_goals = gt_goal.unsqueeze(1).expand_as(pred_goals)
        smooth_l1_goal = F.smooth_l1_loss(pred_goals, gt_goals, reduction="none").sum(dim=-1)  # [B, K]
        loss_goal = (weights * smooth_l1_goal).sum(dim=-1).mean()

        return loss_reg, loss_goal, ade_per_mode, disp

    def forward(
        self,
        pred_traj_2: torch.Tensor,   # [B, K, T=80, 2] (Stage 2 Refined)
        pred_goals_2: torch.Tensor,  # [B, K, 2] (Stage 2 Refined)
        logits_2: torch.Tensor,      # [B, K] (Stage 2 Logits)
        pred_traj_1: torch.Tensor,   # [B, K, T=80, 2] (Stage 1 Proposals)
        pred_goals_1: torch.Tensor,  # [B, K, 2] (Stage 1 Proposals)
        future_gt: torch.Tensor,     # [B, T=80, 2]
        future_valid: torch.Tensor,  # [B, T=80]
        epoch: int = 0,
        total_epochs: int = 30,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, K, T, _ = pred_traj_2.shape
        device = pred_traj_2.device

        mask = future_valid.unsqueeze(1).float()  # [B, 1, T]
        denom = mask.sum(dim=-1).clamp_min(1.0)    # [B, 1]

        # Last True valid frame
        t_ix = torch.arange(T, device=device).view(1, T).expand(B, T)
        last_valid_idx = torch.where(future_valid, t_ix, t_ix.new_zeros(())).max(dim=-1).values
        gt_goal = torch.gather(
            future_gt, 1, last_valid_idx[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)  # [B, 2]

        tau, top_m = self.get_temperature_and_top_m(epoch, total_epochs)

        # 1. Stage 1 Proposal aWTA Loss (guarantees diverse 0.57m base proposals)
        loss_reg_1, loss_goal_1, ade_modes_1, _ = self._compute_awta_reg_loss(
            pred_traj_1, pred_goals_1, future_gt, gt_goal, mask, denom, tau, top_m
        )
        loss_stage1 = self.weight_reg * loss_reg_1 + self.weight_goal * loss_goal_1

        # 2. Stage 2 Refined aWTA Loss (guarantees refined trajectories fit Ground Truth)
        loss_reg_2, loss_goal_2, ade_modes_2, _ = self._compute_awta_reg_loss(
            pred_traj_2, pred_goals_2, future_gt, gt_goal, mask, denom, tau, top_m
        )

        # 3. Mode Classification Loss on Stage 2 Logits
        target_probs = F.softmax(-ade_modes_2 / self.tau_cls, dim=-1).detach()
        log_probs = F.log_softmax(logits_2, dim=-1)
        loss_cls = -(target_probs * log_probs).sum(dim=-1).mean()

        winner_2 = ade_modes_2.argmin(dim=-1)
        loss_hard_cls = F.cross_entropy(logits_2, winner_2)

        # 4. Mode Diversity Regularization on Stage 2 Goals
        goal_pairwise_dist = torch.cdist(pred_goals_2, pred_goals_2, p=2)
        eye_mask = torch.eye(K, device=device).bool().unsqueeze(0)
        diversity_dist = torch.clamp(goal_pairwise_dist, max=self.div_margin)
        diversity_dist = diversity_dist.masked_fill(eye_mask, 0.0)
        loss_div = -diversity_dist.sum(dim=(1, 2)).mean() / (K * (K - 1))

        # 5. Total Deep Supervision Loss
        loss_stage2 = (
            self.weight_reg * loss_reg_2
            + self.weight_goal * loss_goal_2
            + self.weight_cls * loss_cls
            + self.weight_hard_cls * loss_hard_cls
            + self.weight_div * loss_div
        )
        total_loss = self.weight_stage1 * loss_stage1 + loss_stage2

        # Metrics computation
        b_idx = torch.arange(B, device=device)
        top1_mode = logits_2.argmax(dim=-1)
        pick_correct = (top1_mode == winner_2).float().mean()

        minade6_s1 = ade_modes_1.min(dim=-1).values.mean().detach()
        minade6_s2 = ade_modes_2.min(dim=-1).values.mean().detach()
        minade1_s2 = ade_modes_2[b_idx, top1_mode].mean().detach()

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_stage1": loss_stage1.detach(),
            "loss_stage2": loss_stage2.detach(),
            "loss_reg_2": loss_reg_2.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_hard_cls": loss_hard_cls.detach(),
            "tau": tau,
            "top_m": float(top_m),
            "pick_acc": pick_correct.detach(),
            "minade6_s1": minade6_s1,
            "minade6_batch": minade6_s2,
            "minade1_batch": minade1_s2,
        }
        return total_loss, metrics
