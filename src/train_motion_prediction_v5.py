#!/usr/bin/env python3
"""Motion Prediction V5: scene-level target self-attention + all agents in a scene.

Builds on V4 polyline/lane-anchored modes. Targets in the same scene attend to
each other before the trajectory heads. Training uses every valid target
(no 24-cap) with per-scene padding.
"""
from __future__ import annotations

import torch
from torch import nn

from src.train_motion_prediction_v3 import HIST_STEPS, NEIGHBOR_HIST_DIM, SIGNAL_K, load_compatible_state
from src.train_motion_prediction_v4 import (
    MODES,
    POLY_FEAT_DIM,
    POLY_K,
    MotionPredictorV4,
    WindowSampleCollateV4,
    window_to_samples_v4,
)
from src.map_polylines import POLYLINE_POINTS

FLAT_KEYS = (
    "target_hist", "neighbors", "neighbor_valid", "map_feat", "map_valid",
    "signals", "type_idx", "future", "future_valid",
    "lane_goals", "lane_goal_valid", "mode_poly_idx",
)


def _empty_scene(neighbor_k: int, poly_k: int, signal_k: int):
    return {
        "target_hist": torch.zeros(0, HIST_STEPS, 5),
        "neighbors": torch.zeros(0, neighbor_k, HIST_STEPS, NEIGHBOR_HIST_DIM),
        "neighbor_valid": torch.zeros(0, neighbor_k, dtype=torch.bool),
        "map_feat": torch.zeros(0, poly_k, POLYLINE_POINTS, POLY_FEAT_DIM),
        "map_valid": torch.zeros(0, poly_k, dtype=torch.bool),
        "signals": torch.zeros(0, signal_k, 4),
        "type_idx": torch.zeros(0, dtype=torch.long),
        "future": torch.zeros(0, 80, 2),
        "future_valid": torch.zeros(0, 80, dtype=torch.bool),
        "lane_goals": torch.zeros(0, MODES, 2),
        "lane_goal_valid": torch.zeros(0, MODES, dtype=torch.bool),
        "mode_poly_idx": torch.zeros(0, MODES, dtype=torch.long),
    }


def _pad_to(tensor: torch.Tensor, n_max: int) -> torch.Tensor:
    n = int(tensor.shape[0])
    if n == n_max:
        return tensor
    extra = tensor.new_zeros((n_max - n,) + tuple(tensor.shape[1:]))
    return torch.cat([tensor, extra], dim=0)


class WindowSampleCollateV5:
    """Pad every scene to the batch-max target count. max_targets<=0 means all agents."""

    def __init__(self, max_targets: int = 0, neighbor_k: int = 16, signal_k: int = SIGNAL_K, poly_k: int = POLY_K, train: bool = False):
        self.max_targets = max_targets if max_targets and max_targets > 0 else 10**6
        self.neighbor_k = neighbor_k
        self.signal_k = signal_k
        self.poly_k = poly_k
        self.train = train

    def __call__(self, batch):
        scenes = []
        counts = []
        for item in batch:
            sample = window_to_samples_v4(
                [item], self.max_targets, self.neighbor_k, self.signal_k, self.poly_k, self.train,
            )
            if sample is None or sample["target_hist"].shape[0] == 0:
                sample = _empty_scene(self.neighbor_k, self.poly_k, self.signal_k)
            scenes.append(sample)
            counts.append(int(sample["target_hist"].shape[0]))
        n_max = max(counts) if counts else 0
        n_max = max(n_max, 1)
        out = {}
        for key in FLAT_KEYS:
            out[key] = torch.stack([_pad_to(scene[key], n_max) for scene in scenes], dim=0)
        valid = torch.zeros(len(scenes), n_max, dtype=torch.bool)
        for i, n in enumerate(counts):
            if n > 0:
                valid[i, :n] = True
        out["target_valid"] = valid
        return out


class MotionPredictorV5(MotionPredictorV4):
    def __init__(self, hidden: int = 256, modes: int = 6, nhead: int = 4):
        super().__init__(hidden=hidden, modes=modes, nhead=nhead)
        self.agent_attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.agent_norm = nn.LayerNorm(hidden)

    def forward(
        self, target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
        lane_goals, lane_goal_valid, mode_poly_idx, target_valid,
    ):
        b, n = target_hist.shape[:2]

        def flat(x):
            return x.reshape(b * n, *x.shape[2:])

        h, poly_tok = self.encode_context(
            flat(target_hist), flat(neighbors), flat(neighbor_valid),
            flat(map_feat), flat(map_valid), flat(signals), flat(type_idx),
        )
        h = h.view(b, n, -1)
        pad = ~target_valid
        all_pad = pad.all(dim=1)
        if all_pad.any():
            pad = pad.clone()
            pad[all_pad, 0] = False
        h2, _ = self.agent_attn(h, h, h, key_padding_mask=pad)
        h = self.agent_norm(h + h2)
        h = h * target_valid.unsqueeze(-1).to(dtype=h.dtype)
        pred, goals, logits = self.decode_heads(
            h.reshape(b * n, -1),
            poly_tok,
            flat(lane_goals),
            flat(lane_goal_valid),
            flat(mode_poly_idx),
        )
        pred = pred.view(b, n, self.modes, 80, 2)
        goals = goals.view(b, n, self.modes, 2)
        logits = logits.view(b, n, self.modes)
        logits = logits.masked_fill(~target_valid.unsqueeze(-1), -1.0e4)
        return pred, goals, logits
