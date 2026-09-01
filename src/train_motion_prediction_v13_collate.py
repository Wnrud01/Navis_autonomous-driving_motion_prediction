#!/usr/bin/env python3
"""Collate and sample extraction for V13 (Polyline map + Tokens)."""
from __future__ import annotations

import torch

from src.train_motion_prediction_v1 import (
    TYPE_TO_INDEX,
    local_vec,
    local_xy,
    select_neighbor_indices,
    wrap_angle,
)
from src.train_motion_prediction_v3 import (
    HIST_STEPS,
    NEIGHBOR_HIST_DIM,
    NEIGHBOR_K,
    SIGNAL_K,
    pack_neighbor_history,
)
from src.train_motion_prediction_v4 import (
    POLY_FEAT_DIM,
    POLY_K,
    POLYLINE_POINTS,
    encode_map_polylines,
)
from src.train_motion_prediction_v10 import (
    AGENT_DIM,
    INTER_DIM,
    LANE_DIM,
    MAP_DIM,
    agent_token,
    inter_token,
    lane_token_np,
    map_token,
    split_hist,
)


def _scene_v13(static: dict, window: dict, max_targets: int = 24, train: bool = False) -> dict[str, torch.Tensor] | None:
    raw_hist = window["inputs"]["agent_history_world"].float()
    hist_valid = window["inputs"]["agent_history_valid"].bool() if "agent_history_valid" in window["inputs"] else None
    hist, valid = split_hist(raw_hist, hist_valid)
    sizes = window["inputs"]["agent_size_m"].float()
    types = static["agent_types"].long()
    is_sdc = static["agent_is_sdc"].bool()
    targets = window["targets"]
    target_rows = targets["target_rows"].long()
    future_world = targets["agent_future_world"].float()
    future_valid = targets["agent_future_valid"].bool()

    keep = []
    for r in target_rows.tolist():
        r = int(r)
        if r < 0 or r >= hist.shape[0]:
            continue
        if bool(is_sdc[r]) or not bool(valid[r, -1]):
            continue
        if int(types[r]) not in TYPE_TO_INDEX:
            continue
        if int(future_valid[r].sum()) < 20:
            continue
        keep.append(r)
    if not keep:
        return None
    if train and max_targets and len(keep) > max_targets:
        perm = torch.randperm(len(keep))[:max_targets]
        keep = [keep[int(i)] for i in perm]
    elif max_targets:
        keep = keep[:max_targets]

    rows = torch.tensor(keep, dtype=torch.long)
    t = int(rows.numel())
    origins = hist[rows, -1, :2]
    yaws = hist[rows, -1, 2]
    th = hist[rows]
    target_hist = torch.cat([
        local_xy(th[:, :, :2], origins[:, None, :], yaws[:, None]),
        wrap_angle(th[:, :, 2] - yaws[:, None]).unsqueeze(-1),
        local_vec(th[:, :, 3:5], yaws[:, None]),
    ], dim=-1)
    target_hist = target_hist * valid[rows].unsqueeze(-1).to(target_hist.dtype)
    speed = torch.linalg.vector_norm(target_hist[:, -1, 3:5], dim=-1)

    n = hist.shape[0]
    if n <= 1:
        nfeat = target_hist.new_zeros(t, NEIGHBOR_K, HIST_STEPS, NEIGHBOR_HIST_DIM)
        nvalid = torch.zeros(t, NEIGHBOR_K, dtype=torch.bool)
    else:
        cur_xy = hist[:, -1, :2]
        cur_local = local_xy(cur_xy.unsqueeze(0), origins[:, None, :], yaws[:, None])
        candidate = valid[:, -1].unsqueeze(0).expand(t, n).clone()
        candidate[torch.arange(t), rows] = False
        nidx, is_lane, is_dir, valid_pick = select_neighbor_indices(
            cur_local, hist[:, -1, 2], yaws, candidate, neighbor_k=NEIGHBOR_K,
        )
        nfeat, nvalid = pack_neighbor_history(
            hist, valid, sizes, types, nidx, is_lane, is_dir, valid_pick, origins, yaws, NEIGHBOR_K,
        )

    # 16 Polylines x 20 Points x 8 Feats
    poly_feat, poly_valid, _, _, _ = encode_map_polylines(
        static, origins, yaws, speed, poly_k=POLY_K,
    )

    mapped = torch.zeros(t, dtype=torch.long)
    for src, dst in TYPE_TO_INDEX.items():
        mapped[types[rows] == src] = dst
    future = local_xy(future_world[rows, :, :2], origins[:, None, :], yaws[:, None])

    agent = agent_token(target_hist, sizes[rows], mapped)
    lane, lane_ok = lane_token_np(static, origins, yaws)
    tl_xy = window["inputs"].get("tl_xy")
    tl_state = window["inputs"].get("tl_state")
    tl_valid = window["inputs"].get("tl_valid")
    if tl_xy is None:
        tl_xy = origins.new_zeros(0, 11, 2)
        tl_state = torch.zeros(0, 11, dtype=torch.int16)
        tl_valid = torch.zeros(0, 11, dtype=torch.bool)
    mmap = map_token(static, origins, yaws, tl_xy.float(), tl_state.long(), tl_valid.bool())
    inter = inter_token(nfeat, nvalid, origins, yaws, hist, valid, is_sdc)
    dummy_sig = target_hist.new_zeros(t, SIGNAL_K, 4)

    return {
        "target_hist": target_hist,
        "neighbors": nfeat,
        "neighbor_valid": nvalid,
        "map_feat": poly_feat,
        "map_valid": poly_valid,
        "signals": dummy_sig,
        "type_idx": mapped,
        "future": future,
        "future_valid": future_valid[rows],
        "agent_tok": agent,
        "lane_tok": lane,
        "lane_valid": lane_ok,
        "map_tok": mmap,
        "inter_tok": inter,
    }


class WindowSampleCollateV13:
    def __init__(self, max_targets: int = 24, train: bool = False):
        self.max_targets = max_targets
        self.train = train

    def __call__(self, batch):
        buckets = None
        for pack in batch:
            if pack is None or "windows" not in pack or not pack["windows"]:
                continue
            static = pack["static"]
            for window in pack["windows"]:
                packed = _scene_v13(static, window, self.max_targets, self.train)
                if packed is None:
                    continue
                if buckets is None:
                    buckets = {k: [] for k in packed}
                for k, v in packed.items():
                    buckets[k].append(v)
        if not buckets:
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
        return {key: torch.cat(value, dim=0) for key, value in buckets.items()}
