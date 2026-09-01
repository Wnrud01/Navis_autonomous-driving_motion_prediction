"""Build Foresight samples from 85k prediction_pt packs."""
from __future__ import annotations

import numpy as np
import torch

from src.foresight.geom import FUTURE_STEPS, ade
from src.foresight.lanes import polylines_from_pack
from src.foresight.mixture import predict_agent
from src.foresight.scorer import FEAT_DIM, heuristic_logit, mode_features, score_context

N_MODES = 6
MIN_GT = 20
KEEP_TYPES = (1, 2, 3)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def scene_samples(static: dict, window: dict, max_targets: int = 0, train: bool = False, min_gt: int = MIN_GT):
    hist = _to_np(window["inputs"]["agent_history_world"]).astype(np.float32)
    hist_valid = _to_np(window["inputs"]["agent_history_valid"]).astype(np.float32)
    sizes = _to_np(window["inputs"]["agent_size_m"]).astype(np.float32)
    types = _to_np(static["agent_types"]).astype(np.int32)
    is_sdc = _to_np(static["agent_is_sdc"]).astype(bool)
    targets = window["targets"]
    target_rows = _to_np(targets["target_rows"]).astype(np.int64)
    if "agent_future_world" in targets:
        future = _to_np(targets["agent_future_world"])[:, :, :2].astype(np.float32)
        future_valid = _to_np(targets["agent_future_valid"]).astype(np.float32)
    else:
        future = _to_np(targets["future_xy_world"]).astype(np.float32)
        future_valid = _to_np(targets["future_valid"]).astype(np.float32)
        # future already indexed by target_rows in some schemas
        indexed = True
    indexed = "future_xy_world" in targets

    n_agents = hist.shape[0]
    live_mask = (hist_valid[:, -1] > 0.5) & (~is_sdc[:n_agents])
    live_idx = np.nonzero(live_mask)[0]

    ego_xy = None
    ego_yaw = 0.0
    sdc_idx = np.nonzero(is_sdc[:n_agents] & (hist_valid[:, -1] > 0.5))[0]
    if len(sdc_idx):
        ei = int(sdc_idx[0])
        ego_xy = hist[ei, -1, :2]
        ego_yaw = float(hist[ei, -1, 2])

    lanes = polylines_from_pack(static)

    rows = []
    for r in target_rows.tolist():
        r = int(r)
        if r < 0 or r >= n_agents:
            continue
        if is_sdc[r] or hist_valid[r, -1] < 0.5:
            continue
        tid = int(types[r])
        if tid not in KEEP_TYPES:
            continue
        if indexed:
            # not used for 85k schema
            pass
        fv = future_valid[r]
        if float(fv.sum()) < min_gt:
            continue
        rows.append(r)

    if not rows:
        return None
    if train and max_targets and len(rows) > max_targets:
        pick = np.random.permutation(len(rows))[:max_targets]
        rows = [rows[i] for i in pick]
    elif max_targets and len(rows) > max_targets:
        rows = rows[:max_targets]

    live_xy = hist[live_idx, -1, :2] if len(live_idx) else np.zeros((0, 2), dtype=np.float32)
    live_speed = np.hypot(hist[live_idx, -1, 3], hist[live_idx, -1, 4]) if len(live_idx) else np.zeros((0,), dtype=np.float32)

    feats, heus, ades, paths, gts, valids, type_idx = [], [], [], [], [], [], []
    for r in rows:
        modes = predict_agent(hist[r], hist_valid[r], int(types[r]), lanes)
        others = live_idx != r
        nb_xy = live_xy[others] if len(live_idx) else np.zeros((0, 2), dtype=np.float32)
        nb_sp = live_speed[others] if len(live_idx) else np.zeros((0,), dtype=np.float32)
        ctx = score_context(hist[r], hist_valid[r], lanes, nb_xy, nb_sp, ego_xy, ego_yaw)
        feat = np.stack([mode_features(int(types[r]), float(sizes[r, 0]), float(sizes[r, 1]), True, m, ctx) for m in modes])
        heu = np.asarray([heuristic_logit(int(types[r]), m, ctx) for m in modes], dtype=np.float32)
        gt = future[r]
        gv = future_valid[r]
        path = np.stack([m.path[:FUTURE_STEPS] for m in modes], axis=0).astype(np.float32)
        mode_ade = np.asarray([ade(m.path, gt, gv) for m in modes], dtype=np.float32)
        feats.append(feat)
        heus.append(heu)
        ades.append(mode_ade)
        paths.append(path)
        gts.append(gt)
        valids.append(gv)
        type_idx.append(0 if types[r] == 1 else (1 if types[r] == 2 else 2))

    return {
        "features": torch.from_numpy(np.stack(feats)),
        "heuristics": torch.from_numpy(np.stack(heus)),
        "ades": torch.from_numpy(np.stack(ades)),
        "paths": torch.from_numpy(np.stack(paths)),
        "future": torch.from_numpy(np.stack(gts)),
        "future_valid": torch.from_numpy(np.stack(valids) > 0.5),
        "type_idx": torch.tensor(type_idx, dtype=torch.long),
    }


def _empty(max_targets: int = 0):
    n = 0
    return {
        "features": torch.zeros(n, N_MODES, FEAT_DIM),
        "heuristics": torch.zeros(n, N_MODES),
        "ades": torch.zeros(n, N_MODES),
        "paths": torch.zeros(n, N_MODES, FUTURE_STEPS, 2),
        "future": torch.zeros(n, FUTURE_STEPS, 2),
        "future_valid": torch.zeros(n, FUTURE_STEPS, dtype=torch.bool),
        "type_idx": torch.zeros(n, dtype=torch.long),
    }


class ForesightCollate:
    def __init__(self, max_targets: int = 24, train: bool = False, min_gt: int = MIN_GT):
        self.max_targets = max_targets
        self.train = train
        self.min_gt = min_gt

    def __call__(self, batch):
        buckets = {k: [] for k in _empty()}
        for static, window in batch:
            packed = scene_samples(static, window, self.max_targets, self.train, self.min_gt)
            if packed is None:
                continue
            for k, v in packed.items():
                buckets[k].append(v)
        if not buckets["future"]:
            return _empty()
        return {k: torch.cat(v, dim=0) for k, v in buckets.items()}
