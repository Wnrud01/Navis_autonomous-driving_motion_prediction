#!/usr/bin/env python3
"""Motion Prediction V8: mixed interaction tokens on V6.

Lane polylines are bound to the nearest traffic light (lane + signal).
Target kinematics (speed, accel, yaw-rate / steer proxy) mix with neighbors
in a second attention stream. New projections start at 0 so a loaded V6
checkpoint is unchanged at step 0.
"""
from __future__ import annotations

import torch
from torch import nn

from src.train_motion_prediction_v1 import local_vec, local_xy
from src.train_motion_prediction_v3 import (
    HIST_STEPS,
    MAP_K,
    NEIGHBOR_HIST_DIM,
    SIGNAL_K,
    MotionPredictorV3,
    WindowSampleCollateV3,
    _scene_targets_v3,
    wrap_angle,
)
from src.train_motion_prediction_v6 import MotionPredictorV6

POLY_K = 12
LANE_SIG_DIM = 16
KIN_DIM = 8
MAP_RANGE_M = 80.0


def target_kinematics(target_hist: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """speed / accel / yaw-rate (steer proxy) / size in the target frame."""
    xy = target_hist[:, :, :2]
    yaw = target_hist[:, :, 2]
    vel = target_hist[:, :, 3:5]
    speed = torch.linalg.vector_norm(vel, dim=-1)
    dt = 1.0
    accel = (speed[:, -1] - speed[:, 0]) / dt
    yaw_rate = wrap_angle(yaw[:, -1] - yaw[:, 0]) / dt
    hist_disp = torch.linalg.vector_norm(xy[:, 0], dim=-1)
    length = sizes[:, 0] if sizes.ndim == 2 else sizes.new_zeros(target_hist.shape[0])
    width = sizes[:, 1] if sizes.ndim == 2 else sizes.new_zeros(target_hist.shape[0])
    return torch.stack([
        (speed[:, -1] / 15.0).clamp(-2.0, 2.0),
        (accel / 3.0).clamp(-2.0, 2.0),
        (yaw_rate / 0.5).clamp(-2.0, 2.0),
        (hist_disp / 20.0).clamp(-2.0, 2.0),
        (vel[:, -1, 0] / 15.0).clamp(-2.0, 2.0),
        (vel[:, -1, 1] / 8.0).clamp(-2.0, 2.0),
        (length / 8.0).clamp(0.0, 2.0),
        (width / 3.0).clamp(0.0, 2.0),
    ], dim=-1)


def encode_lane_signals(
    static: dict,
    origins: torch.Tensor,
    yaws: torch.Tensor,
    sig: torch.Tensor,
    sig_valid: torch.Tensor,
    poly_k: int = POLY_K,
    max_range: float = MAP_RANGE_M,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearby polylines fused with the nearest traffic light. [T, P, 16]."""
    t = int(origins.shape[0])
    empty = origins.new_zeros(t, poly_k, LANE_SIG_DIM)
    empty_valid = torch.zeros(t, poly_k, dtype=torch.bool)
    if "map_polyline_xy" not in static or static["map_polyline_xy"].numel() == 0:
        return empty, empty_valid

    xy = static["map_polyline_xy"].float()
    direc = static["map_polyline_dir"].float()
    pvalid = static["map_polyline_valid"].bool()
    ptype = static["map_polyline_type"].float()
    m = int(xy.shape[0])
    local_pts = local_xy(xy.unsqueeze(0), origins[:, None, None, :], yaws[:, None, None])
    local_dir = local_vec(direc.unsqueeze(0), yaws[:, None, None])
    dist = torch.linalg.vector_norm(local_pts, dim=-1).masked_fill(~pvalid.unsqueeze(0), float("inf"))
    min_d = dist.min(dim=-1).values
    w = pvalid.unsqueeze(0).to(local_pts.dtype)
    centroid = (local_pts * w.unsqueeze(-1)).sum(dim=2) / w.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean_dir = (local_dir * w.unsqueeze(-1)).sum(dim=2)
    mean_dir = mean_dir / torch.linalg.vector_norm(mean_dir, dim=-1, keepdim=True).clamp_min(1e-6)
    last_idx = (pvalid.float().sum(dim=-1).long() - 1).clamp(min=0)
    gather_m = torch.arange(m)
    end_xy = local_pts[:, gather_m, last_idx]
    ahead = (centroid[..., 0] > 0.5).float()
    is_lane = ((ptype == 1) | (ptype == 2) | (ptype == 3)).float().unsqueeze(0)
    score = -min_d + 6.0 * ahead + 3.0 * is_lane
    score = score.masked_fill(min_d > max_range, -1.0e9)
    k = min(poly_k, m)
    nidx = torch.topk(score, k=k, largest=True).indices
    gt = torch.arange(t).unsqueeze(1)
    sel_c = centroid[gt, nidx]
    sel_end = end_xy[gt, nidx]
    sel_dir = mean_dir[gt, nidx]
    sel_d = min_d[gt, nidx]
    sel_type = ptype[nidx]
    sel_ok = score[gt, nidx] > -1.0e8
    sel_lane = ((sel_type == 1) | (sel_type == 2) | (sel_type == 3)).to(sel_c.dtype)

    sig_xy = sel_c.new_zeros(t, k, 2)
    sig_state = sel_c.new_zeros(t, k, 1)
    sig_dist = sel_c.new_full((t, k, 1), 1.0)
    sig_ahead = sel_c.new_zeros(t, k, 1)
    sig_hit = sel_c.new_zeros(t, k, 1)
    n_sig = int(sig.shape[0]) if sig is not None else 0
    if n_sig > 0:
        s_local = local_xy(sig[:, -1, :2].unsqueeze(0), origins[:, None, :], yaws[:, None])
        sdist = torch.linalg.vector_norm(s_local, dim=-1)
        if sig_valid is not None and sig_valid.numel():
            sdist = sdist.masked_fill(~sig_valid[:, -1].unsqueeze(0), float("inf"))
        d_ps = torch.linalg.vector_norm(sel_c.unsqueeze(2) - s_local.unsqueeze(1), dim=-1)
        if sig_valid is not None and sig_valid.numel():
            d_ps = d_ps.masked_fill(~sig_valid[:, -1].view(1, 1, n_sig), float("inf"))
        nearest = d_ps.argmin(dim=-1)
        gather_s = nearest
        sig_xy = s_local[gt, gather_s]
        raw_state = sig[:, -1, 2][gather_s]
        sig_state = (raw_state / 8.0).unsqueeze(-1)
        sig_dist = (d_ps.min(dim=-1).values.clamp(max=max_range) / max_range).unsqueeze(-1)
        sig_ahead = (sig_xy[..., 0] > 0.5).to(sel_c.dtype).unsqueeze(-1)
        sig_hit = torch.isfinite(d_ps.min(dim=-1).values).to(sel_c.dtype).unsqueeze(-1)

    feat = torch.cat([
        sel_c,
        sel_end,
        sel_dir,
        (sel_type / 20.0).unsqueeze(-1),
        sel_lane.unsqueeze(-1),
        (sel_d.clamp(max=max_range) / max_range).unsqueeze(-1),
        (sel_c[..., 0] > 0).to(sel_c.dtype).unsqueeze(-1),
        sig_xy,
        sig_state,
        sig_dist,
        sig_ahead,
        sig_hit,
    ], dim=-1)
    feat = feat * sel_ok.unsqueeze(-1).to(feat.dtype)
    if k < poly_k:
        feat = torch.cat([feat, feat.new_zeros(t, poly_k - k, LANE_SIG_DIM)], dim=1)
        sel_ok = torch.cat([sel_ok, torch.zeros(t, poly_k - k, dtype=torch.bool)], dim=1)
    return feat, sel_ok


def window_to_samples_v8(batch, max_targets=24, neighbor_k=16, signal_k=SIGNAL_K, map_k=MAP_K, poly_k=POLY_K, train=False):
    keys = (
        "target_hist", "neighbors", "neighbor_valid", "map_feat", "map_valid",
        "signals", "type_idx", "future", "future_valid", "kin_feat", "lane_sig", "lane_sig_valid",
    )
    buckets = {k: [] for k in keys}
    for static, window in batch:
        packed = _scene_targets_v3(static, window, max_targets, neighbor_k, signal_k, map_k, train)
        if packed is None:
            continue
        target_hist, nfeat, nvalid, map_feat, map_valid, sfeat, mapped, future, future_valid, origins, yaws, sizes = packed
        kin = target_kinematics(target_hist, sizes)
        sig = window["inputs"]["signal_history_world_state"].float()
        sig_valid = window["inputs"]["signal_history_valid"].bool()
        lane_sig, lane_ok = encode_lane_signals(static, origins, yaws, sig, sig_valid, poly_k=poly_k)
        buckets["target_hist"].append(target_hist)
        buckets["neighbors"].append(nfeat)
        buckets["neighbor_valid"].append(nvalid)
        buckets["map_feat"].append(map_feat)
        buckets["map_valid"].append(map_valid)
        buckets["signals"].append(sfeat)
        buckets["type_idx"].append(mapped)
        buckets["future"].append(future)
        buckets["future_valid"].append(future_valid)
        buckets["kin_feat"].append(kin)
        buckets["lane_sig"].append(lane_sig)
        buckets["lane_sig_valid"].append(lane_ok)
    if not buckets["future"]:
        return None
    return {key: torch.cat(value, dim=0) for key, value in buckets.items()}


class WindowSampleCollateV8:
    def __init__(self, max_targets=24, neighbor_k=16, signal_k=SIGNAL_K, map_k=MAP_K, poly_k=POLY_K, train=False):
        self.max_targets = max_targets
        self.neighbor_k = neighbor_k
        self.signal_k = signal_k
        self.map_k = map_k
        self.poly_k = poly_k
        self.train = train

    def __call__(self, batch):
        samples = window_to_samples_v8(
            batch, self.max_targets, self.neighbor_k, self.signal_k, self.map_k, self.poly_k, self.train,
        )
        if samples is not None:
            return samples
        return {
            "target_hist": torch.zeros(0, HIST_STEPS, 5),
            "neighbors": torch.zeros(0, self.neighbor_k, HIST_STEPS, NEIGHBOR_HIST_DIM),
            "neighbor_valid": torch.zeros(0, self.neighbor_k, dtype=torch.bool),
            "map_feat": torch.zeros(0, self.map_k, 8),
            "map_valid": torch.zeros(0, self.map_k, dtype=torch.bool),
            "signals": torch.zeros(0, self.signal_k, 4),
            "type_idx": torch.zeros(0, dtype=torch.long),
            "future": torch.zeros(0, 80, 2),
            "future_valid": torch.zeros(0, 80, dtype=torch.bool),
            "kin_feat": torch.zeros(0, KIN_DIM),
            "lane_sig": torch.zeros(0, self.poly_k, LANE_SIG_DIM),
            "lane_sig_valid": torch.zeros(0, self.poly_k, dtype=torch.bool),
        }


class MotionPredictorV8(MotionPredictorV6):
    def __init__(self, hidden: int = 256, modes: int = 6, nhead: int = 4):
        super().__init__(hidden=hidden, modes=modes, nhead=nhead)
        self.kin_encoder = nn.Sequential(nn.Linear(KIN_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.kin_skip = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.kin_skip.weight)
        nn.init.zeros_(self.kin_skip.bias)
        self.lane_sig_encoder = nn.Sequential(
            nn.Linear(LANE_SIG_DIM, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden),
        )
        self.mix_attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.mix_norm = nn.LayerNorm(hidden)
        self.mix_gate = nn.Parameter(torch.tensor([1.0]))

    def encode_target(self, target_hist: torch.Tensor, kin_feat: torch.Tensor | None = None) -> torch.Tensor:
        target = MotionPredictorV6.encode_target(self, target_hist)
        if kin_feat is None:
            return target
        return target + self.kin_skip(self.kin_encoder(kin_feat))

    def encode_context(
        self, target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
        kin_feat=None, lane_sig=None, lane_sig_valid=None,
    ):
        target = self.encode_target(target_hist, kin_feat)
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
        h = self.fusion(torch.cat([target, ctx, signal, self.type_emb(type_idx)], dim=-1))

        extra_toks = [self.signal_encoder(signals)]
        extra_pad = [torch.zeros(signals.shape[0], signals.shape[1], dtype=torch.bool, device=signals.device)]
        if lane_sig is not None:
            extra_toks.append(self.lane_sig_encoder(lane_sig))
            extra_pad.append(~lane_sig_valid if lane_sig_valid is not None else torch.zeros(lane_sig.shape[:2], dtype=torch.bool, device=lane_sig.device))
        extra_mem = torch.cat(extra_toks, dim=1)
        extra_mask = torch.cat(extra_pad, dim=1)
        if extra_mask.all(dim=1).any():
            extra_mask = extra_mask.clone()
            extra_mask[extra_mask.all(dim=1), 0] = False
        mix, _ = self.mix_attn(target.unsqueeze(1), extra_mem, extra_mem, key_padding_mask=extra_mask)
        h = h + self.mix_gate * self.mix_norm(mix.squeeze(1))
        return h

    def forward(
        self, target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
        kin_feat=None, lane_sig=None, lane_sig_valid=None,
    ):
        h = self.encode_context(
            target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
            kin_feat=kin_feat, lane_sig=lane_sig, lane_sig_valid=lane_sig_valid,
        )
        return self.decode_heads(h)
