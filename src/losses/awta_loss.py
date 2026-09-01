"""Adaptive Winner-Takes-All (aWTA) & Multi-Future Loss for Motion Prediction.

Based on:
- "Multiple Futures Prediction with Adaptive Winner-Takes-All Loss" (Valeo AI)
- Multi-hypothesis trajectory learning with temperature-scheduled soft-assignment
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveWTALoss(nn.Module):
    def __init__(
        self,
        modes: int = 6,
        tau_start: float = 1.5,
        tau_end: float = 0.25,
        tau_cls: float = 0.5,
        top_m_start: int = 3,
        top_m_end: int = 1,
        weight_reg: float = 1.0,
        weight_goal: float = 0.5,
        weight_cls: float = 0.8,
        weight_hard_cls: float = 0.5,
        weight_div: float = 0.05,
        div_margin: float = 3.0,
        time_weight_end: float = 1.0,
        weight_fde: float = 0.0,
    ):
        super().__init__()
        self.modes = modes
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_cls = tau_cls
        self.top_m_start = top_m_start
        self.top_m_end = top_m_end
        self.weight_reg = weight_reg
        self.weight_goal = weight_goal
        self.weight_cls = weight_cls
        self.weight_hard_cls = weight_hard_cls
        self.weight_div = weight_div
        self.div_margin = div_margin
        self.time_weight_end = time_weight_end
        self.weight_fde = weight_fde

    def get_temperature_and_top_m(self, epoch: int, total_epochs: int) -> tuple[float, int]:
        progress = min(1.0, max(0.0, epoch / max(1, total_epochs - 1)))
        tau = self.tau_start * ((self.tau_end / self.tau_start) ** progress)
        top_m = int(round(self.top_m_start - progress * (self.top_m_start - self.top_m_end)))
        top_m = max(self.top_m_end, min(self.top_m_start, top_m))
        return tau, top_m

    def forward(
        self,
        pred_traj: torch.Tensor,   # [B, K, T=80, 2]
        pred_goals: torch.Tensor,  # [B, K, 2]
        logits: torch.Tensor,      # [B, K]
        future_gt: torch.Tensor,   # [B, T=80, 2]
        future_valid: torch.Tensor,# [B, T=80]
        epoch: int = 0,
        total_epochs: int = 30
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, K, T, _ = pred_traj.shape
        device = pred_traj.device

        mask = future_valid.unsqueeze(1).float()  # [B, 1, T]
        denom = mask.sum(dim=-1).clamp_min(1.0)   # [B, 1]

        # 1. Compute Step-wise Distance & ADE per Mode
        # diff: [B, K, T, 2]
        diff = pred_traj - future_gt.unsqueeze(1)
        disp = torch.linalg.vector_norm(diff, dim=-1)  # [B, K, T]
        ade_per_mode = (disp * mask).sum(dim=-1) / denom # [B, K]

        # Last True valid frame (not count-1, which breaks non-prefix masks)
        t_ix = torch.arange(T, device=device).view(1, T).expand(B, T)
        last_valid_idx = torch.where(future_valid, t_ix, t_ix.new_zeros(())).max(dim=-1).values
        gt_goal = torch.gather(
            future_gt, 1, last_valid_idx[:, None, None].expand(-1, 1, 2)
        ).squeeze(1) # [B, 2]

        # Goal error per mode
        goal_diff = pred_goals - gt_goal.unsqueeze(1) # [B, K, 2]
        goal_dist_per_mode = torch.linalg.vector_norm(goal_diff, dim=-1) # [B, K]

        # Combined mode error for WTA ranking
        mode_error = ade_per_mode + 0.5 * goal_dist_per_mode # [B, K]

        # 2. Adaptive Winner-Takes-All Soft Assignment (aWTA)
        tau, top_m = self.get_temperature_and_top_m(epoch, total_epochs)
        
        # Sort modes by error
        sorted_errors, sorted_indices = torch.sort(mode_error, dim=-1) # [B, K]
        
        # Top-M mask
        top_m_mask = torch.zeros_like(mode_error, dtype=torch.bool)
        top_m_indices = sorted_indices[:, :top_m]
        top_m_mask.scatter_(1, top_m_indices, True)

        # Softmax weights over top-M modes
        masked_errors = mode_error.clone()
        masked_errors[~top_m_mask] = float('inf')
        weights = F.softmax(-masked_errors / max(0.01, tau), dim=-1) # [B, K]
        weights = weights.detach() # Weights act as gradient routers

        # 3. Smooth L1 Trajectory Regression Loss (optional late-horizon emphasis)
        gt_traj = future_gt.unsqueeze(1).expand_as(pred_traj)
        smooth_l1_traj = F.smooth_l1_loss(pred_traj, gt_traj, reduction='none').sum(dim=-1)  # [B, K, T]
        if abs(self.time_weight_end - 1.0) > 1e-6:
            t_w = torch.linspace(1.0, self.time_weight_end, T, device=device, dtype=smooth_l1_traj.dtype)
            wmask = mask * t_w.view(1, 1, T)
            wdenom = wmask.sum(dim=-1).clamp_min(1.0)
            step_reg_loss = (smooth_l1_traj * wmask).sum(dim=-1) / wdenom
        else:
            step_reg_loss = (smooth_l1_traj * mask).sum(dim=-1) / denom  # [B, K]
        loss_reg = (weights * step_reg_loss).sum(dim=-1).mean()

        # 4. Goal Regression Loss
        gt_goals = gt_goal.unsqueeze(1).expand_as(pred_goals)
        smooth_l1_goal = F.smooth_l1_loss(pred_goals, gt_goals, reduction='none').sum(dim=-1)  # [B, K]
        loss_goal = (weights * smooth_l1_goal).sum(dim=-1).mean()

        # 5. Multi-Future Aligned Classification Loss (Soft Target Cross-Entropy)
        # Generate soft target distribution based on true trajectory accuracy
        # Use ADE for the official metric, while retaining a soft target to avoid
        # unstable winner switching early in training.
        target_probs = F.softmax(-ade_per_mode / self.tau_cls, dim=-1).detach() # [B, K]
        log_probs = F.log_softmax(logits, dim=-1) # [B, K]
        loss_cls = -(target_probs * log_probs).sum(dim=-1).mean()
        # The benchmark selects exactly argmax(logits). Add a direct hard-winner
        # term so the selected mode is explicitly trained to be the ADE winner.
        winner = ade_per_mode.argmin(dim=-1)
        loss_hard_cls = F.cross_entropy(logits, winner)

        # 6. Mode Diversity Regularization Loss (Penalize overlapping goals)
        # Pairwise distance between predicted 6 goals: [B, K, K]
        goal_pairwise_dist = torch.cdist(pred_goals, pred_goals, p=2) # [B, K, K]
        eye_mask = torch.eye(K, device=device).bool().unsqueeze(0) # [1, K, K]
        diversity_dist = torch.clamp(goal_pairwise_dist, max=self.div_margin)
        diversity_dist = diversity_dist.masked_fill(eye_mask, 0.0)
        # Maximize pairwise distance -> minimize negative sum
        loss_div = -diversity_dist.sum(dim=(1, 2)).mean() / (K * (K - 1))

        # 7. Optional last-valid-step FDE (differs from goal when horizon is truncated)
        if self.weight_fde > 0:
            gather_idx = last_valid_idx[:, None, None, None].expand(-1, K, 1, 2)
            pred_end = torch.gather(pred_traj, 2, gather_idx).squeeze(2)
            fde_l1 = F.smooth_l1_loss(
                pred_end, gt_goal.unsqueeze(1).expand_as(pred_end), reduction="none"
            ).sum(dim=-1)
            loss_fde = (weights * fde_l1).sum(dim=-1).mean()
        else:
            loss_fde = pred_traj.new_zeros(())

        # 8. Total Weighted Loss
        total_loss = (
            self.weight_reg * loss_reg
            + self.weight_goal * loss_goal
            + self.weight_cls * loss_cls
            + self.weight_hard_cls * loss_hard_cls
            + self.weight_div * loss_div
            + self.weight_fde * loss_fde
        )

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_reg": loss_reg.detach(),
            "loss_goal": loss_goal.detach(),
            "loss_cls": loss_cls.detach(),
            "loss_hard_cls": loss_hard_cls.detach(),
            "loss_div": loss_div.detach(),
            "loss_fde": loss_fde.detach(),
            "tau": tau,
            "top_m": float(top_m),
            "minade6_batch": ade_per_mode.min(dim=-1).values.mean().detach(),
            "minade1_batch": ade_per_mode[torch.arange(B, device=device), logits.argmax(dim=-1)].mean().detach(),
        }

        return total_loss, metrics
