#!/usr/bin/env python3
"""Motion Prediction V3: HD map points, neighbor history, lead/follow attention.

Keeps the V1/V2 target-centric K=6 heads and aWTA training recipe.
Adds:
  1. Nearest lane/edge map points in the target frame
  2. Dedicated lead/follow tokens (no mean-pool over neighbors)
  3. 11-step neighbor history instead of a current-frame snapshot
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.train_motion_prediction_v1 import (
    TYPE_TO_INDEX,
    local_vec,
    local_xy,
    select_neighbor_indices,
    wrap_angle,
)

NEIGHBOR_K = 16
NEIGHBOR_HIST_DIM = 14
MAP_K = 64
MAP_FEAT_DIM = 8
SIGNAL_K = 4
HIST_STEPS = 11
MAP_RANGE_M = 80.0


def _empty_map(t: int, map_k: int, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ref.new_zeros(t, map_k, MAP_FEAT_DIM), torch.zeros(t, map_k, dtype=torch.bool)


def encode_map_points(
    static: dict,
    origins: torch.Tensor,
    yaws: torch.Tensor,
    map_k: int = MAP_K,
    max_range: float = MAP_RANGE_M,
    keep_types: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest lane-center / edge / crosswalk points in each target frame."""
    t = int(origins.shape[0])
    rg_xyz = static["roadgraph_xyz_world"].float()
    rg_dir = static["roadgraph_dir_world"].float()
    rg_type = static["roadgraph_type"].long()
    rg_valid = static["roadgraph_valid"].bool()
    if rg_xyz.ndim != 2 or rg_xyz.shape[0] == 0:
        return _empty_map(t, map_k, origins)

    if keep_types is None:
        type_ok = (
            (rg_type == 1) | (rg_type == 2) | (rg_type == 3)
            | (rg_type == 15) | (rg_type == 16) | (rg_type == 19)
        )
    else:
        type_ok = torch.zeros_like(rg_type, dtype=torch.bool)
        for tid in keep_types:
            type_ok |= rg_type == int(tid)
    useful = rg_valid & type_ok
    if int(useful.sum()) == 0:
        return _empty_map(t, map_k, origins)

    xy = rg_xyz[useful, :2]
    direction = rg_dir[useful, :2]
    types = rg_type[useful].float()
    center = origins.mean(dim=0)
    near = torch.linalg.vector_norm(xy - center, dim=-1) < (max_range + 40.0)
    if bool(near.any()):
        xy, direction, types = xy[near], direction[near], types[near]
    p = int(xy.shape[0])
    if p == 0:
        return _empty_map(t, map_k, origins)

    local = local_xy(xy.unsqueeze(0).expand(t, p, 2), origins[:, None, :], yaws[:, None])
    loc_dir = local_vec(direction.unsqueeze(0).expand(t, p, 2), yaws[:, None])
    dist = torch.linalg.vector_norm(local, dim=-1)
    dist = dist.masked_fill(dist > max_range, float("inf"))
    k = min(map_k, p)
    nidx = torch.topk(dist, k=k, largest=False).indices
    gather_t = torch.arange(t).unsqueeze(1)
    sel_xy = local[gather_t, nidx]
    sel_dir = loc_dir[gather_t, nidx]
    sel_dist = dist[gather_t, nidx]
    sel_type = types[nidx]
    valid = torch.isfinite(sel_dist) & (sel_dist <= max_range)
    is_lane = ((sel_type == 1) | (sel_type == 2) | (sel_type == 3)).to(sel_xy.dtype)
    is_edge = ((sel_type == 15) | (sel_type == 16)).to(sel_xy.dtype)
    feat = torch.cat([
        sel_xy,
        sel_dir,
        (sel_type / 20.0).unsqueeze(-1),
        is_lane.unsqueeze(-1),
        is_edge.unsqueeze(-1),
        (sel_dist.clamp(max=max_range) / max_range).unsqueeze(-1),
    ], dim=-1)
    feat = torch.where(valid.unsqueeze(-1), feat, torch.zeros_like(feat))
    if k < map_k:
        feat = torch.cat([feat, feat.new_zeros(t, map_k - k, MAP_FEAT_DIM)], dim=1)
        valid = torch.cat([valid, torch.zeros(t, map_k - k, dtype=torch.bool)], dim=1)
    return feat, valid


def pack_neighbor_history(
    hist: torch.Tensor,
    hist_valid: torch.Tensor,
    sizes: torch.Tensor,
    types: torch.Tensor,
    nidx: torch.Tensor,
    is_lane: torch.Tensor,
    is_dir: torch.Tensor,
    valid_pick: torch.Tensor,
    origins: torch.Tensor,
    yaws: torch.Tensor,
    neighbor_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [T, K, 11, 14] neighbor histories in the target frame, plus a [T, K] mask."""
    t, k = nidx.shape
    n_agents, n_steps, _ = hist.shape
    xy = local_xy(hist[None, :, :, :2], origins[:, None, None, :], yaws[:, None, None])
    vel = local_vec(hist[None, :, :, 3:5], yaws[:, None, None])
    yaw_loc = wrap_angle(hist[None, :, :, 2] - yaws[:, None, None])
    idx2 = nidx[:, :, None, None].expand(t, k, n_steps, 2)
    idx1 = nidx[:, :, None].expand(t, k, n_steps)
    n_xy = xy.gather(1, idx2)
    n_vel = vel.gather(1, idx2)
    n_yaw = yaw_loc.gather(1, idx1)
    n_valid = hist_valid[nidx]
    n_size = sizes[nidx].clamp(0.0, 30.0)
    if n_size.shape[-1] >= 3:
        n_size = n_size.clone()
        n_size[..., 2] = 0.0
    n_size = n_size.unsqueeze(2).expand(t, k, n_steps, n_size.shape[-1])
    n_type = types[nidx].to(dtype=hist.dtype).unsqueeze(-1).unsqueeze(2).expand(t, k, n_steps, 1)

    cur_x = n_xy[:, :, -1, 0]
    lane = is_lane & valid_pick
    lead_score = torch.where(lane & (cur_x > 0.5), -cur_x, cur_x.new_full(cur_x.shape, -1.0e9))
    follow_score = torch.where(lane & (cur_x < -0.5), cur_x, cur_x.new_full(cur_x.shape, -1.0e9))
    lead_idx = lead_score.argmax(dim=1)
    follow_idx = follow_score.argmax(dim=1)
    has_lead = lead_score.max(dim=1).values > -1.0e8
    has_follow = follow_score.max(dim=1).values > -1.0e8
    rows = torch.arange(t)
    is_lead = torch.zeros_like(is_lane)
    is_follow = torch.zeros_like(is_lane)
    is_lead[rows, lead_idx] = has_lead
    is_follow[rows, follow_idx] = has_follow
    is_follow = is_follow & ~is_lead

    ones = hist.new_ones(t, k, n_steps, 1)
    feat = torch.cat([
        n_xy,
        n_yaw.unsqueeze(-1),
        n_vel,
        n_valid.to(dtype=hist.dtype).unsqueeze(-1),
        n_size,
        n_type,
        is_lane.to(dtype=hist.dtype).view(t, k, 1, 1) * ones,
        is_dir.to(dtype=hist.dtype).view(t, k, 1, 1) * ones,
        is_lead.to(dtype=hist.dtype).view(t, k, 1, 1) * ones,
        is_follow.to(dtype=hist.dtype).view(t, k, 1, 1) * ones,
    ], dim=-1)
    feat = feat * valid_pick.view(t, k, 1, 1).to(dtype=feat.dtype)
    if k < neighbor_k:
        feat = torch.cat([feat, feat.new_zeros(t, neighbor_k - k, n_steps, feat.shape[-1])], dim=1)
        valid_pick = torch.cat([valid_pick, torch.zeros(t, neighbor_k - k, dtype=torch.bool)], dim=1)
    return feat, valid_pick


def _scene_targets_v3(
    static: dict,
    window: dict,
    max_targets: int,
    neighbor_k: int,
    signal_k: int,
    map_k: int,
    train: bool,
):
    hist = window["inputs"]["agent_history_world"].float()
    valid = window["inputs"]["agent_history_valid"].bool()
    sizes = window["inputs"]["agent_size_m"].float()
    types = static["agent_types"].long()
    sig = window["inputs"]["signal_history_world_state"].float()
    sig_valid = window["inputs"]["signal_history_valid"].bool()
    targets = window["targets"]
    target_rows = targets["target_rows"].long()
    n_all = int(target_rows.numel())
    if n_all == 0:
        return None
    if "target_types" in targets:
        target_types = targets["target_types"].long()
    else:
        target_types = types[target_rows]
    if "future_xy_world" in targets:
        future_world = targets["future_xy_world"].float()
        future_valid = targets["future_valid"].bool()
    else:
        future_world = targets["agent_future_world"].float()[target_rows, :, :2]
        future_valid = targets["agent_future_valid"].bool()[target_rows]

    keep_mask = torch.zeros(n_all, dtype=torch.bool)
    for src in TYPE_TO_INDEX:
        keep_mask |= target_types == src
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    if keep_idx.numel() == 0:
        return None
    if train and int(keep_idx.numel()) > max_targets:
        keep_idx = keep_idx[torch.randperm(keep_idx.numel())[:max_targets]]
    else:
        keep_idx = keep_idx[:max_targets]

    rows = target_rows[keep_idx]
    t = int(rows.numel())
    n = hist.shape[0]
    origins = hist[rows, -1, :2]
    yaws = hist[rows, -1, 2]
    yaw_th = yaws[:, None]
    th = hist[rows]
    target_hist = torch.cat([
        local_xy(th[:, :, :2], origins[:, None, :], yaw_th),
        (th[:, :, 2] - yaw_th).unsqueeze(-1),
        local_vec(th[:, :, 3:5], yaw_th),
    ], dim=-1)

    if n <= 1:
        nfeat = target_hist.new_zeros(t, neighbor_k, HIST_STEPS, NEIGHBOR_HIST_DIM)
        nvalid = torch.zeros(t, neighbor_k, dtype=torch.bool)
    else:
        cur_xy = hist[:, -1, :2]
        yaw_n = yaws[:, None]
        cur_local = local_xy(cur_xy.unsqueeze(0), origins[:, None, :], yaw_n)
        candidate = valid[:, -1].unsqueeze(0).expand(t, n).clone()
        candidate[torch.arange(t), rows] = False
        nidx, is_lane, is_dir, valid_pick = select_neighbor_indices(
            cur_local, hist[:, -1, 2], yaws, candidate, neighbor_k=neighbor_k,
        )
        nfeat, nvalid = pack_neighbor_history(
            hist, valid, sizes, types, nidx, is_lane, is_dir, valid_pick, origins, yaws, neighbor_k,
        )

    n_sig = sig.shape[0]
    if n_sig == 0:
        sfeat = target_hist.new_zeros(t, signal_k, 4)
    else:
        yaw_s = yaws[:, None]
        sig_local = local_xy(sig[:, -1, :2].unsqueeze(0), origins[:, None, :], yaw_s)
        sdist = torch.linalg.vector_norm(sig_local, dim=-1)
        if sig_valid.numel():
            sdist = sdist.masked_fill(~sig_valid[:, -1].unsqueeze(0), float("inf"))
        sidx = torch.topk(sdist, k=min(signal_k, n_sig), largest=False).indices
        gather_t = torch.arange(t).unsqueeze(1)
        if sig.shape[1] > 1:
            changes = (sig[:, 1:, 2] != sig[:, :-1, 2]).float().sum(dim=1)
        else:
            changes = sig.new_zeros(n_sig)
        sfeat = torch.cat([
            sig_local[gather_t, sidx],
            sig[:, -1, 2][sidx].unsqueeze(-1),
            changes[sidx].unsqueeze(-1),
        ], dim=-1)
        pad = signal_k - sfeat.shape[1]
        if pad > 0:
            sfeat = torch.cat([sfeat, sfeat.new_zeros(t, pad, 4)], dim=1)

    map_feat, map_valid = encode_map_points(static, origins, yaws, map_k=map_k)
    mapped = torch.empty(t, dtype=torch.long)
    selected_types = target_types[keep_idx]
    for src, dst in TYPE_TO_INDEX.items():
        mapped[selected_types == src] = dst
    future = local_xy(future_world[keep_idx], origins[:, None, :], yaws[:, None])
    return (
        target_hist, nfeat, nvalid, map_feat, map_valid, sfeat, mapped, future, future_valid[keep_idx],
        origins, yaws, sizes[rows],
    )


def window_to_samples_v3(
    batch: list[tuple[dict, dict]],
    max_targets: int = 24,
    neighbor_k: int = NEIGHBOR_K,
    signal_k: int = SIGNAL_K,
    map_k: int = MAP_K,
    train: bool = False,
) -> dict[str, torch.Tensor] | None:
    keys = (
        "target_hist", "neighbors", "neighbor_valid", "map_feat", "map_valid",
        "signals", "type_idx", "future", "future_valid",
    )
    buckets = {k: [] for k in keys}
    for static, window in batch:
        packed = _scene_targets_v3(static, window, max_targets, neighbor_k, signal_k, map_k, train)
        if packed is None:
            continue
        target_hist, nfeat, nvalid, map_feat, map_valid, sfeat, mapped, future, future_valid = packed[:9]
        buckets["target_hist"].append(target_hist)
        buckets["neighbors"].append(nfeat)
        buckets["neighbor_valid"].append(nvalid)
        buckets["map_feat"].append(map_feat)
        buckets["map_valid"].append(map_valid)
        buckets["signals"].append(sfeat)
        buckets["type_idx"].append(mapped)
        buckets["future"].append(future)
        buckets["future_valid"].append(future_valid)
    if not buckets["future"]:
        return None
    return {key: torch.cat(value, dim=0) for key, value in buckets.items()}


class WindowSampleCollateV3:
    def __init__(
        self,
        max_targets: int = 24,
        neighbor_k: int = NEIGHBOR_K,
        signal_k: int = SIGNAL_K,
        map_k: int = MAP_K,
        train: bool = False,
    ):
        self.max_targets = max_targets
        self.neighbor_k = neighbor_k
        self.signal_k = signal_k
        self.map_k = map_k
        self.train = train

    def __call__(self, batch):
        samples = window_to_samples_v3(
            batch, self.max_targets, self.neighbor_k, self.signal_k, self.map_k, self.train,
        )
        if samples is not None:
            return samples
        return {
            "target_hist": torch.zeros(0, HIST_STEPS, 5),
            "neighbors": torch.zeros(0, self.neighbor_k, HIST_STEPS, NEIGHBOR_HIST_DIM),
            "neighbor_valid": torch.zeros(0, self.neighbor_k, dtype=torch.bool),
            "map_feat": torch.zeros(0, self.map_k, MAP_FEAT_DIM),
            "map_valid": torch.zeros(0, self.map_k, dtype=torch.bool),
            "signals": torch.zeros(0, self.signal_k, 4),
            "type_idx": torch.zeros(0, dtype=torch.long),
            "future": torch.zeros(0, 80, 2),
            "future_valid": torch.zeros(0, 80, dtype=torch.bool),
        }


class MotionPredictorV3(nn.Module):
    def __init__(self, hidden: int = 256, modes: int = 6, nhead: int = 4):
        super().__init__()
        self.modes = modes
        self.hidden = hidden
        self.type_emb = nn.Embedding(3, 16)
        self.role_emb = nn.Embedding(3, hidden)
        self.target_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(55, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.neigh_step = nn.Sequential(
            nn.Linear(NEIGHBOR_HIST_DIM, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden),
        )
        self.map_encoder = nn.Sequential(
            nn.Linear(MAP_FEAT_DIM, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden),
        )
        self.signal_encoder = nn.Sequential(
            nn.Linear(4, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden),
        )
        self.cross_attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3 + 16, hidden * 2), nn.GELU(), nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden), nn.GELU(),
        )
        self.goal_head = nn.Linear(hidden, modes * 2)
        self.delta_head = nn.Linear(hidden, modes * 80 * 2)
        self.mode_head = nn.Linear(hidden, modes)

    def _pool_neighbors(self, neighbors: torch.Tensor) -> torch.Tensor:
        # neighbors: [B, K, 11, 14]
        step_h = self.neigh_step(neighbors)
        step_valid = neighbors[..., 5:6]
        step_h = step_h * step_valid
        denom = step_valid.sum(dim=2).clamp_min(1.0)
        return step_h.sum(dim=2) / denom

    def encode_target(self, target_hist: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(target_hist)

    def encode_context(self, target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx):
        target = self.encode_target(target_hist)
        neigh_tok = self._pool_neighbors(neighbors)
        map_tok = self.map_encoder(map_feat)

        lead_w = neighbors[:, :, -1, 12].clamp(min=0.0)
        follow_w = neighbors[:, :, -1, 13].clamp(min=0.0)
        lead_tok = (neigh_tok * lead_w.unsqueeze(-1)).sum(dim=1, keepdim=True)
        follow_tok = (neigh_tok * follow_w.unsqueeze(-1)).sum(dim=1, keepdim=True)
        lead_present = (lead_w.sum(dim=1, keepdim=True) > 0.5).to(dtype=neigh_tok.dtype)
        follow_present = (follow_w.sum(dim=1, keepdim=True) > 0.5).to(dtype=neigh_tok.dtype)
        lead_tok = (lead_tok + self.role_emb.weight[1]) * lead_present.unsqueeze(-1)
        follow_tok = (follow_tok + self.role_emb.weight[2]) * follow_present.unsqueeze(-1)

        mem = torch.cat([neigh_tok, lead_tok, follow_tok, map_tok], dim=1)
        pad = torch.cat([
            ~neighbor_valid,
            lead_present <= 0.5,
            follow_present <= 0.5,
            ~map_valid,
        ], dim=1)
        all_pad = pad.all(dim=1)
        if all_pad.any():
            pad = pad.clone()
            pad[all_pad, 0] = False
        ctx, _ = self.cross_attn(target.unsqueeze(1), mem, mem, key_padding_mask=pad)
        ctx = self.attn_norm(ctx.squeeze(1) + target)
        signal = self.signal_encoder(signals).mean(dim=1)
        return self.fusion(torch.cat([target, ctx, signal, self.type_emb(type_idx)], dim=-1))

    def decode_heads(self, h: torch.Tensor):
        b = h.shape[0]
        goals = self.goal_head(h).view(b, self.modes, 2)
        deltas = self.delta_head(h).view(b, self.modes, 80, 2)
        raw = torch.cumsum(deltas, dim=2)
        time = torch.linspace(1 / 80, 1, 80, device=h.device, dtype=h.dtype).view(1, 1, 80, 1)
        trajectories = raw + time * (goals.unsqueeze(2) - raw[:, :, -1:, :])
        return trajectories, goals, self.mode_head(h)

    def forward(self, target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx):
        h = self.encode_context(
            target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
        )
        return self.decode_heads(h)


def load_compatible_state(model: nn.Module, state: dict[str, Any]) -> tuple[list[str], list[str]]:
    current = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    return model.load_state_dict(filtered, strict=False)
