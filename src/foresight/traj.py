"""8 one-second residual waypoints interpolated to 80 steps on frozen rollouts."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from src.foresight.scorer import FEAT_DIM

N_WP = 8


class TrajHead(nn.Module):
    def __init__(self, dim: int = FEAT_DIM, n_wp: int = N_WP):
        super().__init__()
        self.n_wp = n_wp
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, n_wp * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def residual(self, features: torch.Tensor, steps: int) -> torch.Tensor:
        b, k, _ = features.shape
        wp = self.net(features).view(b, k, self.n_wp, 2)
        seq = wp.reshape(b * k, self.n_wp, 2).permute(0, 2, 1)
        dense = F.interpolate(seq, size=steps, mode="linear", align_corners=True)
        return dense.permute(0, 2, 1).view(b, k, steps, 2)

    def warp(self, paths: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return paths + self.residual(features, paths.size(2))
