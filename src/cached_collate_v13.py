"""Load precomputed V13 collate tensors (with 16x20x8 polylines). Subsample targets at train time."""
from __future__ import annotations

import os
import torch
from torch.utils.data import Dataset

from src.train_motion_prediction_v1 import list_pt_paths
from src.train_motion_prediction_v3 import (
    HIST_STEPS,
    NEIGHBOR_HIST_DIM,
    NEIGHBOR_K,
    SIGNAL_K,
)
from src.train_motion_prediction_v4 import (
    POLY_FEAT_DIM,
    POLY_K,
    POLYLINE_POINTS,
)
from src.train_motion_prediction_v10 import (
    AGENT_DIM,
    INTER_DIM,
    LANE_DIM,
    MAP_DIM,
)


def _empty_batch_v13():
    return {
        "target_hist": torch.zeros(0, HIST_STEPS, 5),
        "neighbors": torch.zeros(0, NEIGHBOR_K, HIST_STEPS, NEIGHBOR_HIST_DIM),
        "neighbor_valid": torch.zeros(0, NEIGHBOR_K, dtype=torch.bool),
        "map_feat": torch.zeros(0, POLY_K, POLYLINE_POINTS, POLY_FEAT_DIM),
        "map_valid": torch.zeros(0, POLY_K, dtype=torch.bool),
        "signals": torch.zeros(0, SIGNAL_K, 4),
        "type_idx": torch.zeros(0, dtype=torch.long),
        "future": torch.zeros(0, 80, 2),
        "future_valid": torch.zeros(0, 80, dtype=torch.bool),
        "agent_tok": torch.zeros(0, AGENT_DIM),
        "lane_tok": torch.zeros(0, LANE_DIM),
        "lane_valid": torch.zeros(0, dtype=torch.bool),
        "map_tok": torch.zeros(0, MAP_DIM),
        "inter_tok": torch.zeros(0, INTER_DIM),
    }


def subsample_packed(packed: dict, max_targets: int, train: bool) -> dict:
    t = int(packed["type_idx"].shape[0])
    if not max_targets or t <= max_targets:
        return packed
    if train:
        idx = torch.randperm(t)[:max_targets]
    else:
        idx = torch.arange(max_targets)
    out = {}
    for k, v in packed.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == t:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


class CachedSceneDatasetV13(Dataset):
    def __init__(self, root: str, split: str, max_packs: int = 0, paths: list[str] | None = None):
        self.paths = list(paths) if paths is not None else list_pt_paths(root, split, max_packs)
        if max_packs and paths is not None:
            self.paths = self.paths[:max_packs]
        if not self.paths:
            raise FileNotFoundError(f"No cached {split} packs under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return torch.load(self.paths[index], map_location="cpu", weights_only=False)


class CachedWindowCollateV13:
    def __init__(self, max_targets: int = 24, train: bool = False):
        self.max_targets = max_targets
        self.train = train

    def __call__(self, batch):
        buckets = None
        for packed in batch:
            if packed is None:
                continue
            packed = subsample_packed(packed, self.max_targets, self.train)
            if buckets is None:
                buckets = {k: [] for k in packed}
            for k, v in packed.items():
                buckets[k].append(v)
        if not buckets:
            return _empty_batch_v13()
        out = {k: torch.cat(v, 0) for k, v in buckets.items()}
        for k, v in out.items():
            if torch.is_floating_point(v):
                out[k] = torch.nan_to_num(v.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return out


def try_cached_loader_v13(cache_root: str, split: str, max_packs: int = 0):
    folder = os.path.join(cache_root, split)
    if not os.path.isdir(folder):
        return None
    try:
        paths = list_pt_paths(cache_root, split, max_packs)
    except FileNotFoundError:
        return None
    return CachedSceneDatasetV13(cache_root, split, max_packs, paths=paths)
