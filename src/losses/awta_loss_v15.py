"""Adaptive Winner-Takes-All Loss V15: All-Mode Training + Lane-Following Loss.

Key Changes from V13:
1. aWTA keeps Top-6 throughout (NO annealing to Top-1).
   - All 6 modes receive gradient every step, ensuring all predictions stay high quality.
   - tau anneals from 1.5 → 0.3 (soft → moderately focused, but never kills any mode).
2. Lane-Following Auxiliary Loss:
   - Penalizes each mode's trajectory for deviating from the nearest polyline centerline.
   - Forces ALL 6 modes to stay on valid road geometry.
3. Reduced diversity loss weight to allow modes to converge when appropriate
   (e.g., straight road — all 6 should predict similar trajectories).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZON_STEPS = (9, 19, 39, 59, 79)


class AdaptiveWTALossV15(nn.Module):
    def __init__(
        self,
        modes: int = 6,
        tau_start: float = 1.5,
        tau_end: float = 0.3,
        tau_cls: float = 0.5,
        top_m: int = 6,           # FIXED at 6 — all modes always trained
        weight_reg: float = 1.0,
        weight_goal: float = 0.15,
        weight_cls: float = 0.8,
        weight_hard_cls: float = 0.5,
        weight_div: float = 0.02,  # Reduced: allow convergence on unambiguous scenes
        weight_lane: float = 0.3,  # NEW: lane-following auxiliary loss
        div_margin: float = 3.0,
        horizon_weight_scale: float = 0.5,
    ):
        super().__init__()
        self.modes = modes
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_cls = tau_cls
        self.top_m = top_m
        self.weight_reg = weight_reg
        self.weight_goal = weight_goal
        self.weight_cls = weight_cls
        self.weight_hard_cls = weight_hard_cls
        self.weight_div = weight_div
        self.weight_lane = weight_lane
        self.div_margin = div_margin
        self.horizon_weight_scale = horizon_weight_scale

    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """Simple cosine annealing: tau_start → tau_end over all epochs."""
        progress = min(1.0, max(0.0, epoch / max(1, total_epochs - 1)))
        # Cosine decay
        tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1 + __import__('math').cos(progress * 3.14159265))
        return max(self.tau_end, tau)

    @staticmethod
    def compute_lane_following_loss(
        pred_traj: torch.Tensor,   # [B, K, T, 2]
        map_feat: torch.Tensor,    # [B, 16, 20, 8] — polyline points (x, y, ...)
        map_valid: torch.Tensor,   # [B, 16]
        future_valid: torch.Tensor,  # [B, T]
        n_sample: int = 10,        # Sample 10 waypoints from trajectory
    ) -> torch.Tensor:
        """Lane-following loss: penalize trajectory deviation from nearest polyline.

        For each of the 6 mode trajectories, sample 10 waypoints and compute
        the minimum distance to any point on any valid polyline. Average across
        all waypoints and modes.
        """
        B, K, T, _ = pred_traj.shape
        device = pred_traj.device

        # Sample waypoint indices evenly across the trajectory
        wp_idx = torch.linspace(0, T - 1, n_sample, device=device).long()  # [10]
        traj_pts = pred_traj[:, :, wp_idx, :2]  # [B, K, 10, 2]

        # Extract polyline xy coordinates
        poly_xy = map_feat[:, :, :, :2]  # [B, 16, 20, 2]

        # Reshape for broadcasting distance computation
        # traj_pts: [B, K, 10, 1, 1, 2]
        # poly_xy:  [B, 1, 1, 16, 20, 2]
        traj_exp = traj_pts.unsqueeze(3).unsqueeze(4)   # [B, K, 10, 1, 1, 2]
        poly_exp = poly_xy.unsqueeze(1).unsqueeze(2)     # [B, 1, 1, 16, 20, 2]

        # Distance from each trajectory waypoint to each polyline point
        dist = torch.linalg.vector_norm(traj_exp - poly_exp, dim=-1)  # [B, K, 10, 16, 20]

        # Mask invalid polylines: set distance to inf
        poly_mask = map_valid.unsqueeze(1).unsqueeze(2).unsqueeze(4)  # [B, 1, 1, 16, 1]
        dist = dist.masked_fill(~poly_mask, float('inf'))

        # Also mask invalid polyline points (column 4 is valid flag in map_feat)
        if map_feat.shape[-1] >= 5:
            pt_valid = (map_feat[:, :, :, 4] > 0.5)  # [B, 16, 20]
            pt_mask = pt_valid.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 16, 20]
            dist = dist.masked_fill(~pt_mask, float('inf'))

        # Min distance across all polyline points and all polylines
        min_dist_to_lane = dist.reshape(B, K, n_sample, -1).min(dim=-1).values  # [B, K, 10]

        # Clamp to avoid inf in loss (if no valid polylines exist)
        min_dist_to_lane = min_dist_to_lane.clamp_max(10.0)

        # Smooth L1 to encourage staying close (target = 0 lateral deviation)
        lane_loss = F.smooth_l1_loss(min_dist_to_lane,
                                      torch.zeros_like(min_dist_to_lane),
                                      reduction='none')  # [B, K, 10]

        # Average across waypoints, modes, and batch
        return lane_loss.mean()

    def forward(
        self,
        pred_traj: torch.Tensor,    # [B, K, T=80, 2]
        pred_goals: torch.Tensor,   # [B, K, 2]
        logits: torch.Tensor,       # [B, K]
        future_gt: torch.Tensor,    # [B, T=80, 2]
        future_valid: torch.Tensor, # [B, T=80]
        map_feat: torch.Tensor,     # [B, 16, 20, 8] — NEW: needed for lane loss
        map_valid: torch.Tensor,    # [B, 16] — NEW: needed for lane loss
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

        # 2. All-Mode Soft Assignment (Top-6, NO annealing to Top-1)
        tau = self.get_temperature(epoch, total_epochs)
        top_m = self.top_m  # Fixed at 6

        sorted_errors, sorted_indices = torch.sort(mode_error, dim=-1)
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
        smooth_l1_goal = F.smooth_l1_loss(pred_goals, gt_goals, reduction="none").sum(dim=-1)
        loss_goal = (weights * smooth_l1_goal).sum(dim=-1).mean()

        # 5. Classification Loss
        target_probs = F.softmax(-ade_per_mode / self.tau_cls, dim=-1).detach()
        log_probs = F.log_softmax(logits, dim=-1)
        loss_cls = -(target_probs * log_probs).sum(dim=-1).mean()

        winner = ade_per_mode.argmin(dim=-1)
        loss_hard_cls = F.cross_entropy(logits, winner)

        # 6. Mode Diversity Regularization (reduced weight)
        goal_pairwise_dist = torch.cdist(pred_goals, pred_goals, p=2)
        eye_mask = torch.eye(K, device=device).bool().unsqueeze(0)
        diversity_dist = torch.clamp(goal_pairwise_dist, max=self.div_margin)
        diversity_dist = diversity_dist.masked_fill(eye_mask, 0.0)
        loss_div = -diversity_dist.sum(dim=(1, 2)).mean() / (K * (K - 1))

        # 7. NEW: Lane-Following Auxiliary Loss (ALL 6 modes)
        loss_lane = self.compute_lane_following_loss(
            pred_traj, map_feat, map_valid, future_valid,
        )

        # 8. Total Loss
        total_loss = (
            self.weight_reg * loss_reg
            + self.weight_goal * loss_goal
            + self.weight_cls * loss_cls
            + self.weight_hard_cls * loss_hard_cls
            + self.weight_div * loss_div
            + self.weight_lane * loss_lane
        )

        b_idx = torch.arange(B, device=device)
        top1_mode = logits.argmax(dim=-1)
        pick_correct = (top1_mode == winner).float().mean()

        # Per-mode ADE stats for monitoring
        ade_all_modes_mean = ade_per_mode.mean(dim=-1).mean()  # average ADE across ALL modes
        ade_worst_mode = ade_per_mode.max(dim=-1).values.mean()  # worst mode's ADE

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_reg": loss_reg.detach(),
            "loss_goal": loss_goal.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_hard_cls": loss_hard_cls.detach(),
            "loss_div": loss_div.detach(),
            "loss_lane": loss_lane.detach(),
            "tau": tau,
            "top_m": float(top_m),
            "pick_acc": pick_correct.detach(),
            "minade6_batch": ade_per_mode.min(dim=-1).values.mean().detach(),
            "minade1_batch": ade_per_mode[b_idx, top1_mode].mean().detach(),
            "ade_all_mean": ade_all_modes_mean.detach(),    # NEW: 6개 모드 전체 평균 ADE
            "ade_worst": ade_worst_mode.detach(),            # NEW: 최악 모드 ADE
        }
        return total_loss, metrics
