"""Adaptive Winner-Takes-All Loss V14 with Margin-based ADE-Proportional Ranking Loss.

Key Features:
1. Multi-Horizon Waypoint Supervision (1s, 2s, 4s, 6s, 8s anchor boosts).
2. 3-Stage aWTA Annealing (Top-4 -> Top-2 -> Top-1, tau 1.5 -> 0.1).
3. ADE-Proportional Margin Ranking Loss:
   L_rank = max(0, gamma + alpha * (ADE_k - ADE_winner) - (s_winner - s_k))
   Strongly penalizes poor modes while allowing close runner-ups without noisy penalty.
4. Soft Target CE + Mode Diversity Regularization.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZON_STEPS = (9, 19, 39, 59, 79)


class AdaptiveWTALossV14(nn.Module):
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
        weight_reg: float = 1.0,
        weight_goal: float = 0.15,
        weight_cls: float = 0.5,
        weight_rank: float = 0.3,
        weight_div: float = 0.08,
        div_margin: float = 3.0,
        horizon_weight_scale: float = 0.5,
        rank_margin_base: float = 0.2,
        rank_margin_alpha: float = 0.5,
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
        self.weight_reg = weight_reg
        self.weight_goal = weight_goal
        self.weight_cls = weight_cls
        self.weight_rank = weight_rank
        self.weight_div = weight_div
        self.div_margin = div_margin
        self.horizon_weight_scale = horizon_weight_scale
        self.rank_margin_base = rank_margin_base
        self.rank_margin_alpha = rank_margin_alpha

    def get_temperature_and_top_m(self, epoch: int, total_epochs: int) -> tuple[float, int]:
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

    def forward(
        self,
        pred_traj: torch.Tensor,    # [B, K, T=80, 2]
        pred_goals: torch.Tensor,   # [B, K, 2]
        logits: torch.Tensor,       # [B, K]
        future_gt: torch.Tensor,    # [B, T=80, 2]
        future_valid: torch.Tensor, # [B, T=80]
        epoch: int = 0,
        total_epochs: int = 30,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, K, T, _ = pred_traj.shape
        device = pred_traj.device

        mask = future_valid.unsqueeze(1).float()  # [B, 1, T]
        denom = mask.sum(dim=-1).clamp_min(1.0)    # [B, 1]

        # 1. Compute Step-wise Distance & ADE per Mode
        diff = pred_traj - future_gt.unsqueeze(1)
        disp = torch.linalg.vector_norm(diff, dim=-1)  # [B, K, T]
        ade_per_mode = (disp * mask).sum(dim=-1) / denom  # [B, K]

        # Last True valid frame
        t_ix = torch.arange(T, device=device).view(1, T).expand(B, T)
        last_valid_idx = torch.where(future_valid, t_ix, t_ix.new_zeros(())).max(dim=-1).values
        gt_goal = torch.gather(
            future_gt, 1, last_valid_idx[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)  # [B, 2]

        goal_diff = pred_goals - gt_goal.unsqueeze(1)  # [B, K, 2]
        goal_dist_per_mode = torch.linalg.vector_norm(goal_diff, dim=-1)  # [B, K]

        mode_error = ade_per_mode + 0.5 * goal_dist_per_mode  # [B, K]

        # 2. Adaptive Winner-Takes-All Soft Assignment (aWTA)
        tau, top_m = self.get_temperature_and_top_m(epoch, total_epochs)

        sorted_errors, sorted_indices = torch.sort(mode_error, dim=-1)  # [B, K]
        top_m_mask = torch.zeros_like(mode_error, dtype=torch.bool)
        top_m_indices = sorted_indices[:, :top_m]
        top_m_mask.scatter_(1, top_m_indices, True)

        masked_errors = mode_error.clone()
        masked_errors[~top_m_mask] = float("inf")
        weights = F.softmax(-masked_errors / max(0.01, tau), dim=-1)  # [B, K]
        weights = weights.detach()

        # 3. Multi-Horizon Smooth L1 Trajectory Regression Loss
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

        # 4. Goal Regression Loss
        gt_goals = gt_goal.unsqueeze(1).expand_as(pred_goals)
        smooth_l1_goal = F.smooth_l1_loss(pred_goals, gt_goals, reduction="none").sum(dim=-1)  # [B, K]
        loss_goal = (weights * smooth_l1_goal).sum(dim=-1).mean()

        # 5. Method 2: Margin-based ADE-Proportional Ranking Loss (Log-Sigmoid)
        winner = ade_per_mode.argmin(dim=-1)  # [B]
        b_idx = torch.arange(B, device=device)
        winner_ade = ade_per_mode[b_idx, winner].unsqueeze(1)  # [B, 1]
        winner_logit = logits[b_idx, winner].unsqueeze(1)      # [B, 1]

        # ADE delta for each mode vs winner: [B, K] >= 0
        ade_delta = (ade_per_mode - winner_ade).clamp_min(0.0)

        # Dynamic target margin: gamma + alpha * ΔADE, clamped to prevent extreme values
        target_margin = (self.rank_margin_base + self.rank_margin_alpha * ade_delta).clamp_max(3.0)  # [B, K]
        logit_diff = winner_logit - logits  # [B, K], should be >= 0 ideally

        # Log-Sigmoid ranking loss: bounded in [0, ~7], smooth gradients
        # We want logit_diff >= target_margin, i.e. sigmoid(logit_diff - target_margin) -> 1
        loss_per_pair = -F.logsigmoid(logit_diff - target_margin)  # [B, K]

        # Mask out winner itself
        non_winner_mask = torch.ones_like(loss_per_pair, dtype=torch.bool)
        non_winner_mask[b_idx, winner] = False
        loss_rank = (loss_per_pair * non_winner_mask.float()).sum(dim=-1).mean() / max(1, K - 1)

        # Soft Target Cross Entropy (for smooth probability calibration)
        target_probs = F.softmax(-ade_per_mode / self.tau_cls, dim=-1).detach()  # [B, K]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, K]
        loss_cls = -(target_probs * log_probs).sum(dim=-1).mean()

        # 6. Mode Diversity Regularization Loss
        goal_pairwise_dist = torch.cdist(pred_goals, pred_goals, p=2)  # [B, K, K]
        eye_mask = torch.eye(K, device=device).bool().unsqueeze(0)      # [1, K, K]
        diversity_dist = torch.clamp(goal_pairwise_dist, max=self.div_margin)
        diversity_dist = diversity_dist.masked_fill(eye_mask, 0.0)
        loss_div = -diversity_dist.sum(dim=(1, 2)).mean() / (K * (K - 1))

        # 7. Total Weighted Loss
        total_loss = (
            self.weight_reg * loss_reg
            + self.weight_goal * loss_goal
            + self.weight_cls * loss_cls
            + self.weight_rank * loss_rank
            + self.weight_div * loss_div
        )

        top1_mode = logits.argmax(dim=-1)
        pick_correct = (top1_mode == winner).float().mean()

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_reg": loss_reg.detach(),
            "loss_goal": loss_goal.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_rank": loss_rank.detach(),
            "loss_div": loss_div.detach(),
            "tau": tau,
            "top_m": float(top_m),
            "pick_acc": pick_correct.detach(),
            "minade6_batch": ade_per_mode.min(dim=-1).values.mean().detach(),
            "minade1_batch": ade_per_mode[b_idx, top1_mode].mean().detach(),
        }
        return total_loss, metrics
