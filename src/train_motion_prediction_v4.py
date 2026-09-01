#!/usr/bin/env python3
"""Motion Prediction V4: polyline HD map + lane-anchored K=6 goals.

Replaces V3's 64 nearest map points with resampled lane polylines, and ties each
mode to a map polyline so classification is 'which lane' rather than a free goal.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.map_polylines import POLYLINE_POINTS
from src.train_motion_prediction_v1 import TYPE_TO_INDEX, local_vec, local_xy, select_neighbor_indices
from src.train_motion_prediction_v3 import (
    HIST_STEPS,
    NEIGHBOR_HIST_DIM,
    SIGNAL_K,
    load_compatible_state,
    pack_neighbor_history,
)

POLY_K = 16
POLY_FEAT_DIM = 8
MODES = 6
MAP_RANGE_M = 80.0


def _empty_polylines(t: int, poly_k: int, modes: int, ref: torch.Tensor):
    feat = ref.new_zeros(t, poly_k, POLYLINE_POINTS, POLY_FEAT_DIM)
    valid = torch.zeros(t, poly_k, dtype=torch.bool)
    goals = ref.new_zeros(t, modes, 2)
    goals[:, :, 0] = 24.0
    gvalid = torch.zeros(t, modes, dtype=torch.bool)
    gvalid[:, 0] = True
    idx = torch.zeros(t, modes, dtype=torch.long)
    return feat, valid, goals, gvalid, idx


def _gather_pts(xy: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    # xy: [T, K, 20, 2], index: [T, K]
    t, k, _, _ = xy.shape
    gather_t = torch.arange(t, device=xy.device).view(t, 1).expand(t, k)
    gather_k = torch.arange(k, device=xy.device).view(1, k).expand(t, k)
    return xy[gather_t, gather_k, index.clamp(0, xy.shape[2] - 1)]


def encode_map_polylines(
    static: dict,
    origins: torch.Tensor,
    yaws: torch.Tensor,
    speed: torch.Tensor,
    poly_k: int = POLY_K,
    modes: int = MODES,
    max_range: float = MAP_RANGE_M,
):
    """Select nearby polylines and 6 lane-anchored 8s goals in the target frame."""
    t = int(origins.shape[0])
    if "map_polyline_xy" not in static or static["map_polyline_xy"].numel() == 0:
        return _empty_polylines(t, poly_k, modes, origins)

    xy_w = static["map_polyline_xy"].float()
    dir_w = static["map_polyline_dir"].float()
    pvalid = static["map_polyline_valid"].bool()
    ptype = static["map_polyline_type"].float()
    m = int(xy_w.shape[0])
    local_xy_pts = local_xy(xy_w.unsqueeze(0), origins[:, None, None, :], yaws[:, None, None])
    local_dir = local_vec(dir_w.unsqueeze(0), yaws[:, None, None])
    dist = torch.linalg.vector_norm(local_xy_pts, dim=-1)
    dist = dist.masked_fill(~pvalid.unsqueeze(0), float("inf"))
    min_d = dist.min(dim=-1).values
    ahead = (local_xy_pts[..., 0] > 0.5) & pvalid.unsqueeze(0)
    heading = (local_dir[..., 0] * pvalid.unsqueeze(0).float()).sum(dim=-1) / pvalid.unsqueeze(0).float().sum(dim=-1).clamp_min(1.0)
    lat = local_xy_pts[..., 1].abs().masked_fill(~pvalid.unsqueeze(0), 1.0e6).min(dim=-1).values
    is_lane = ((ptype == 1) | (ptype == 2) | (ptype == 3)).float().unsqueeze(0)
    in_range = min_d <= max_range
    score = (
        -min_d
        + 8.0 * ahead.any(dim=-1).float()
        + 4.0 * heading.clamp(-1.0, 1.0)
        + 3.0 * is_lane
        - 0.15 * lat.clamp(max=40.0)
    )
    score = score.masked_fill(~in_range, -1.0e9)
    k = min(poly_k, m)
    nidx = torch.topk(score, k=k, largest=True).indices
    gt = torch.arange(t).unsqueeze(1)
    sel_xy = local_xy_pts[gt, nidx]
    sel_dir = local_dir[gt, nidx]
    sel_valid_pts = pvalid.unsqueeze(0).expand(t, m, POLYLINE_POINTS)[gt, nidx]
    sel_type = ptype[nidx]
    sel_min_d = min_d[gt, nidx]
    sel_score = score[gt, nidx]
    poly_ok = sel_score > -1.0e8
    is_lane_sel = ((sel_type == 1) | (sel_type == 2) | (sel_type == 3)).to(sel_xy.dtype)
    feat = torch.cat([
        sel_xy,
        sel_dir,
        sel_valid_pts.unsqueeze(-1).to(sel_xy.dtype),
        (sel_type / 20.0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, POLYLINE_POINTS, 1),
        is_lane_sel.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, POLYLINE_POINTS, 1),
        (sel_min_d.clamp(max=max_range) / max_range).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, POLYLINE_POINTS, 1),
    ], dim=-1)
    feat = feat * poly_ok.view(t, k, 1, 1).to(feat.dtype)
    if k < poly_k:
        feat = torch.cat([feat, feat.new_zeros(t, poly_k - k, POLYLINE_POINTS, POLY_FEAT_DIM)], dim=1)
        poly_ok = torch.cat([poly_ok, torch.zeros(t, poly_k - k, dtype=torch.bool)], dim=1)
        sel_xy = torch.cat([sel_xy, sel_xy.new_zeros(t, poly_k - k, POLYLINE_POINTS, 2)], dim=1)
        sel_valid_pts = torch.cat([sel_valid_pts, torch.zeros(t, poly_k - k, POLYLINE_POINTS, dtype=torch.bool)], dim=1)
        sel_score = torch.cat([sel_score, sel_score.new_full((t, poly_k - k), -1.0e9)], dim=1)

    horizon = (speed * 8.0).clamp(12.0, 72.0)
    seg = torch.linalg.vector_norm(sel_xy[:, :, 1:, :] - sel_xy[:, :, :-1, :], dim=-1)
    seg = seg * sel_valid_pts[:, :, 1:].float() * sel_valid_pts[:, :, :-1].float()
    arc = torch.cumsum(seg, dim=-1)
    arc = torch.cat([arc.new_zeros(t, poly_k, 1), arc], dim=-1)
    nearest = dist.new_zeros(t, poly_k, dtype=torch.long)
    # nearest among the already-selected polylines
    dsel = torch.linalg.vector_norm(sel_xy, dim=-1).masked_fill(~sel_valid_pts, float("inf"))
    nearest = dsel.argmin(dim=-1)
    s0 = _gather_pts(arc.unsqueeze(-1), nearest).squeeze(-1)
    s_query = s0 + horizon.view(t, 1)
    s_end = arc[:, :, -1].clamp_min(1e-3)
    s_query = torch.minimum(s_query, s_end)
    idx0 = (arc <= s_query.unsqueeze(-1)).sum(dim=-1).clamp(min=1) - 1
    idx0 = idx0.clamp(0, POLYLINE_POINTS - 2)
    idx1 = idx0 + 1
    s_a = _gather_pts(arc.unsqueeze(-1), idx0).squeeze(-1)
    s_b = _gather_pts(arc.unsqueeze(-1), idx1).squeeze(-1)
    w = ((s_query - s_a) / (s_b - s_a).clamp_min(1e-3)).unsqueeze(-1)
    p0 = _gather_pts(sel_xy, idx0)
    p1 = _gather_pts(sel_xy, idx1)
    along = (1.0 - w) * p0 + w * p1
    along = along.clone()
    along[..., 0] = along[..., 0].clamp(-MAP_RANGE_M, MAP_RANGE_M)
    along[..., 1] = along[..., 1].clamp(-40.0, 40.0)
    kin = torch.stack([horizon, torch.zeros_like(horizon)], dim=-1).unsqueeze(1)
    along = torch.where((along[..., 0] < 2.0).unsqueeze(-1), kin.expand_as(along), along)

    mode_score = sel_score + 12.0 * (along[..., 0] > 4.0).float() - 8.0 * (along[..., 0] < 0.0).float()
    mode_idx = torch.topk(mode_score, k=min(modes, poly_k), largest=True).indices
    if mode_idx.shape[1] < modes:
        pad = mode_idx.new_zeros(t, modes - mode_idx.shape[1])
        mode_idx = torch.cat([mode_idx, pad], dim=1)
    gather_t = torch.arange(t).unsqueeze(1)
    lane_goals = along[gather_t, mode_idx]
    lane_valid = poly_ok[gather_t, mode_idx]
    kin = origins.new_zeros(t, modes, 2)
    kin[:, :, 0] = horizon.unsqueeze(1)
    lane_goals = torch.where(lane_valid.unsqueeze(-1), lane_goals, kin)
    return feat, poly_ok, lane_goals, lane_valid, mode_idx


def _scene_targets_v4(static, window, max_targets, neighbor_k, signal_k, poly_k, train: bool):
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
    speed = torch.linalg.vector_norm(target_hist[:, -1, 3:5], dim=-1)

    if n <= 1:
        nfeat = target_hist.new_zeros(t, neighbor_k, HIST_STEPS, NEIGHBOR_HIST_DIM)
        nvalid = torch.zeros(t, neighbor_k, dtype=torch.bool)
    else:
        cur_local = local_xy(hist[:, -1, :2].unsqueeze(0), origins[:, None, :], yaws[:, None])
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
        sig_local = local_xy(sig[:, -1, :2].unsqueeze(0), origins[:, None, :], yaws[:, None])
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

    poly_feat, poly_valid, lane_goals, lane_valid, mode_idx = encode_map_polylines(
        static, origins, yaws, speed, poly_k=poly_k,
    )
    mapped = torch.empty(t, dtype=torch.long)
    selected_types = target_types[keep_idx]
    for src, dst in TYPE_TO_INDEX.items():
        mapped[selected_types == src] = dst
    future = local_xy(future_world[keep_idx], origins[:, None, :], yaws[:, None])
    return (
        target_hist, nfeat, nvalid, poly_feat, poly_valid, sfeat, mapped,
        future, future_valid[keep_idx], lane_goals, lane_valid, mode_idx,
    )


def window_to_samples_v4(
    batch: list[tuple[dict, dict]],
    max_targets: int = 24,
    neighbor_k: int = 16,
    signal_k: int = SIGNAL_K,
    poly_k: int = POLY_K,
    train: bool = False,
):
    keys = (
        "target_hist", "neighbors", "neighbor_valid", "map_feat", "map_valid",
        "signals", "type_idx", "future", "future_valid",
        "lane_goals", "lane_goal_valid", "mode_poly_idx",
    )
    buckets = {k: [] for k in keys}
    for static, window in batch:
        packed = _scene_targets_v4(static, window, max_targets, neighbor_k, signal_k, poly_k, train)
        if packed is None:
            continue
        (
            target_hist, nfeat, nvalid, poly_feat, poly_valid, sfeat, mapped,
            future, future_valid, lane_goals, lane_valid, mode_idx,
        ) = packed
        buckets["target_hist"].append(target_hist)
        buckets["neighbors"].append(nfeat)
        buckets["neighbor_valid"].append(nvalid)
        buckets["map_feat"].append(poly_feat)
        buckets["map_valid"].append(poly_valid)
        buckets["signals"].append(sfeat)
        buckets["type_idx"].append(mapped)
        buckets["future"].append(future)
        buckets["future_valid"].append(future_valid)
        buckets["lane_goals"].append(lane_goals)
        buckets["lane_goal_valid"].append(lane_valid)
        buckets["mode_poly_idx"].append(mode_idx)
    if not buckets["future"]:
        return None
    return {key: torch.cat(value, dim=0) for key, value in buckets.items()}


class WindowSampleCollateV4:
    def __init__(self, max_targets=24, neighbor_k=16, signal_k=SIGNAL_K, poly_k=POLY_K, train=False):
        self.max_targets = max_targets
        self.neighbor_k = neighbor_k
        self.signal_k = signal_k
        self.poly_k = poly_k
        self.train = train

    def __call__(self, batch):
        samples = window_to_samples_v4(
            batch, self.max_targets, self.neighbor_k, self.signal_k, self.poly_k, self.train,
        )
        if samples is not None:
            return samples
        return {
            "target_hist": torch.zeros(0, HIST_STEPS, 5),
            "neighbors": torch.zeros(0, self.neighbor_k, HIST_STEPS, NEIGHBOR_HIST_DIM),
            "neighbor_valid": torch.zeros(0, self.neighbor_k, dtype=torch.bool),
            "map_feat": torch.zeros(0, self.poly_k, POLYLINE_POINTS, POLY_FEAT_DIM),
            "map_valid": torch.zeros(0, self.poly_k, dtype=torch.bool),
            "signals": torch.zeros(0, self.signal_k, 4),
            "type_idx": torch.zeros(0, dtype=torch.long),
            "future": torch.zeros(0, 80, 2),
            "future_valid": torch.zeros(0, 80, dtype=torch.bool),
            "lane_goals": torch.zeros(0, MODES, 2),
            "lane_goal_valid": torch.zeros(0, MODES, dtype=torch.bool),
            "mode_poly_idx": torch.zeros(0, MODES, dtype=torch.long),
        }


class MotionPredictorV4(nn.Module):
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
        self.poly_step = nn.Sequential(
            nn.Linear(POLY_FEAT_DIM, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden),
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
        self.goal_res = nn.Sequential(
            nn.Linear(hidden * 2 + 2, hidden), nn.GELU(), nn.Linear(hidden, 2),
        )
        self.delta_head = nn.Linear(hidden, modes * 80 * 2)
        self.mode_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 2, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )

    def _pool_neighbors(self, neighbors: torch.Tensor) -> torch.Tensor:
        step_h = self.neigh_step(neighbors)
        step_valid = neighbors[..., 5:6]
        step_h = step_h * step_valid
        denom = step_valid.sum(dim=2).clamp_min(1.0)
        return step_h.sum(dim=2) / denom

    def _pool_polylines(self, poly_feat: torch.Tensor) -> torch.Tensor:
        # poly_feat: [B, P, 20, 8], valid channel is index 4
        step_h = self.poly_step(poly_feat)
        step_valid = poly_feat[..., 4:5]
        step_h = step_h * step_valid
        denom = step_valid.sum(dim=2).clamp_min(1.0)
        return step_h.sum(dim=2) / denom

    def encode_context(
        self, target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
    ):
        target = self.target_encoder(target_hist)
        neigh_tok = self._pool_neighbors(neighbors)
        poly_tok = self._pool_polylines(map_feat)

        lead_w = neighbors[:, :, -1, 12].clamp(min=0.0)
        follow_w = neighbors[:, :, -1, 13].clamp(min=0.0)
        lead_tok = (neigh_tok * lead_w.unsqueeze(-1)).sum(dim=1, keepdim=True)
        follow_tok = (neigh_tok * follow_w.unsqueeze(-1)).sum(dim=1, keepdim=True)
        lead_present = (lead_w.sum(dim=1, keepdim=True) > 0.5).to(dtype=neigh_tok.dtype)
        follow_present = (follow_w.sum(dim=1, keepdim=True) > 0.5).to(dtype=neigh_tok.dtype)
        lead_tok = (lead_tok + self.role_emb.weight[1]) * lead_present.unsqueeze(-1)
        follow_tok = (follow_tok + self.role_emb.weight[2]) * follow_present.unsqueeze(-1)

        mem = torch.cat([neigh_tok, lead_tok, follow_tok, poly_tok], dim=1)
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
        h = self.fusion(torch.cat([target, ctx, signal, self.type_emb(type_idx)], dim=-1))
        return h, poly_tok

    def decode_heads(self, h, poly_tok, lane_goals, lane_goal_valid, mode_poly_idx):
        b = h.shape[0]
        gather_b = torch.arange(b, device=h.device).unsqueeze(1).expand(b, self.modes)
        mode_poly = poly_tok[gather_b, mode_poly_idx.clamp(min=0, max=poly_tok.shape[1] - 1)]
        mode_in = torch.cat([
            h.unsqueeze(1).expand(-1, self.modes, -1),
            mode_poly,
            lane_goals,
        ], dim=-1)
        residual = torch.tanh(self.goal_res(mode_in)) * 12.0
        goals = lane_goals + residual * lane_goal_valid.unsqueeze(-1).to(dtype=h.dtype)
        logits = self.mode_mlp(mode_in).squeeze(-1)
        logits = logits.masked_fill(~lane_goal_valid, -1.0e4)
        deltas = self.delta_head(h).view(b, self.modes, 80, 2)
        raw = torch.cumsum(deltas, dim=2)
        time = torch.linspace(1 / 80, 1, 80, device=h.device, dtype=h.dtype).view(1, 1, 80, 1)
        trajectories = raw + time * (goals.unsqueeze(2) - raw[:, :, -1:, :])
        return trajectories, goals, logits

    def forward(
        self, target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
        lane_goals, lane_goal_valid, mode_poly_idx,
    ):
        h, poly_tok = self.encode_context(
            target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
        )
        return self.decode_heads(h, poly_tok, lane_goals, lane_goal_valid, mode_poly_idx)
