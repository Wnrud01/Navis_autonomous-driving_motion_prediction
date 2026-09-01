#!/usr/bin/env python3
"""Motion Prediction V6: temporal history GRU + trajectory refiner on V3.

Starts as V3: new projections are zero-initialized so a loaded V3 checkpoint
produces identical trajectories at step 0. The GRU and refiner then learn
late-horizon corrections that V3's flatten-MLP decoder missed.

minFDE6 on V3 is ~13.5m with minADE6 ~0.88m — the 8s tail is the bottleneck.
"""
from __future__ import annotations

import torch
from torch import nn

from src.train_motion_prediction_v3 import MotionPredictorV3


class MotionPredictorV6(MotionPredictorV3):
    def __init__(self, hidden: int = 256, modes: int = 6, nhead: int = 4):
        super().__init__(hidden=hidden, modes=modes, nhead=nhead)
        self.hist_gru = nn.GRU(5, hidden, num_layers=1, batch_first=True)
        self.hist_skip = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.hist_skip.weight)
        nn.init.zeros_(self.hist_skip.bias)
        self.refine_gru = nn.GRU(2, hidden // 4, num_layers=1, batch_first=True)
        self.refine_out = nn.Linear(hidden // 4, 2)
        nn.init.zeros_(self.refine_out.weight)
        nn.init.zeros_(self.refine_out.bias)

    def encode_target(self, target_hist: torch.Tensor) -> torch.Tensor:
        target = self.target_encoder(target_hist)
        gru_out, _ = self.hist_gru(target_hist)
        return target + self.hist_skip(gru_out[:, -1])

    def decode_heads(self, h: torch.Tensor):
        trajectories, goals, logits = super().decode_heads(h)
        b = h.shape[0]
        flat = trajectories.reshape(b * self.modes, 80, 2)
        ref, _ = self.refine_gru(flat)
        trajectories = trajectories + self.refine_out(ref).view(b, self.modes, 80, 2)
        time = torch.linspace(1 / 80, 1, 80, device=h.device, dtype=h.dtype).view(1, 1, 80, 1)
        trajectories = trajectories + time * (goals.unsqueeze(2) - trajectories[:, :, -1:, :])
        return trajectories, goals, logits
