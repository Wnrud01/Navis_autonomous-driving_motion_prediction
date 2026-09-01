#!/usr/bin/env python3
"""V10: from-scratch motion prediction on v2 packs.

Tokens: AGENT / LANE (type {1,2} k/N) / MAP (TL, xwalk 18, stop 17, edges) / INTER.
History in packs is [N,11,6] = x,y,yaw,vx,vy,valid. Future GT is labels only.
"""
from __future__ import annotations

import torch
from torch import nn

from src.lane_index import lane_index_from_polylines
from src.map_polylines import KEEP_TYPES, LANE_TYPES
from src.train_motion_prediction_v1 import TYPE_TO_INDEX, local_vec, local_xy, wrap_angle
from src.train_motion_prediction_v3 import (
    HIST_STEPS,
    MAP_K,
    NEIGHBOR_HIST_DIM,
    NEIGHBOR_K,
    SIGNAL_K,
    encode_map_points,
    pack_neighbor_history,
    select_neighbor_indices,
)

AGENT_DIM = 16
LANE_DIM = 8
MAP_DIM = 16
INTER_DIM = 16


def split_hist(hist: torch.Tensor, hist_valid: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    if hist.shape[-1] >= 6:
        xy5 = hist[..., :5]
        valid = hist[..., 5] > 0.5
        if hist_valid is not None:
            valid = valid & hist_valid.bool()
        return xy5, valid
    valid = hist_valid.bool() if hist_valid is not None else torch.ones(hist.shape[:2], dtype=torch.bool)
    return hist[..., :5], valid


def last_true_index(valid: torch.Tensor) -> torch.Tensor:
    t = valid.shape[-1]
    idx = torch.arange(t, device=valid.device).view(1, t).expand_as(valid)
    return torch.where(valid, idx, idx.new_zeros(())).max(dim=-1).values


def agent_token(target_hist: torch.Tensor, sizes: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
    xy, yaw, vel = target_hist[:, :, :2], target_hist[:, :, 2], target_hist[:, :, 3:5]
    speed = torch.linalg.vector_norm(vel, dim=-1)
    accel = speed[:, -1] - speed[:, 0]
    yaw_rate = wrap_angle(yaw[:, -1] - yaw[:, 0])
    vel_yaw = torch.atan2(vel[:, -1, 1], vel[:, -1, 0])
    steer = wrap_angle(vel_yaw - yaw[:, -1])
    disp = torch.linalg.vector_norm(xy[:, 0], dim=-1)
    onehot = torch.nn.functional.one_hot(type_idx.clamp(0, 2), 3).to(target_hist.dtype)
    return torch.cat([
        (speed[:, -1] / 15.0).clamp(-2, 2).unsqueeze(-1),
        (accel / 3.0).clamp(-2, 2).unsqueeze(-1),
        (yaw_rate / 0.5).clamp(-2, 2).unsqueeze(-1),
        (steer / 0.5).clamp(-2, 2).unsqueeze(-1),
        (vel[:, -1, 0] / 15.0).clamp(-2, 2).unsqueeze(-1),
        (vel[:, -1, 1] / 8.0).clamp(-2, 2).unsqueeze(-1),
        (disp / 20.0).clamp(-2, 2).unsqueeze(-1),
        (sizes[:, 0] / 8.0).clamp(0, 2).unsqueeze(-1),
        (sizes[:, 1] / 3.0).clamp(0, 2).unsqueeze(-1),
        onehot,
        target_hist.new_zeros(target_hist.shape[0], 4),
    ], dim=-1)


def lane_token_np(static, origins, yaws) -> tuple[torch.Tensor, torch.Tensor]:
    t = origins.shape[0]
    feat = origins.new_zeros(t, LANE_DIM)
    valid = torch.zeros(t, dtype=torch.bool, device=origins.device)
    if "map_polyline_xy" not in static or static["map_polyline_xy"].numel() == 0:
        return feat, valid
    xy = static["map_polyline_xy"].numpy()
    direc = static["map_polyline_dir"].numpy()
    pvalid = static["map_polyline_valid"].numpy()
    ptype = static["map_polyline_type"].numpy()
    for i in range(t):
        info = lane_index_from_polylines(
            xy, direc, pvalid, ptype,
            float(origins[i, 0]), float(origins[i, 1]), float(yaws[i]),
        )
        if not info["valid"]:
            continue
        n = max(1, info["n_lanes"])
        feat[i, 0] = info["lane_idx"] / n
        feat[i, 1] = info["n_lanes"] / 6.0
        feat[i, 2] = 1.0 if info["has_left"] else 0.0
        feat[i, 3] = 1.0 if info["has_right"] else 0.0
        feat[i, 4] = max(-2.0, min(2.0, info["lat"] / 4.0))
        feat[i, 5] = max(-2.0, min(2.0, info["yaw_err"] / 0.5))
        feat[i, 6] = 1.0
        valid[i] = True
    return feat, valid


def map_token(static, origins, yaws, tl_xy, tl_state, tl_valid) -> torch.Tensor:
    t = origins.shape[0]
    feat = origins.new_zeros(t, MAP_DIM)
    # traffic lights (mask invalid; xy may be -1)
    if tl_xy is not None and tl_xy.numel():
        xy = tl_xy[:, -1]
        ok = tl_valid[:, -1] if tl_valid is not None and tl_valid.numel() else torch.ones(xy.shape[0], dtype=torch.bool)
        if int(ok.sum()):
            loc = local_xy(xy.unsqueeze(0), origins[:, None, :], yaws[:, None])
            dist = torch.linalg.vector_norm(loc, dim=-1)
            dist = dist.masked_fill(~ok.unsqueeze(0), float("inf"))
            cone = (loc[..., 0] > 0.5) & (loc[..., 1].abs() < 12.0)
            dist = dist.masked_fill(~cone, float("inf"))
            j = dist.argmin(-1)
            hit = torch.isfinite(dist[torch.arange(t), j])
            loc_j = loc[torch.arange(t), j]
            st = tl_state[:, -1].float()[j] / 8.0
            nchg = origins.new_zeros(t)
            if tl_state.shape[1] > 1:
                chg = (tl_state[:, 1:] != tl_state[:, :-1]).float()
                vm = tl_valid[:, 1:].float() if tl_valid is not None else 1.0
                nchg = (chg * vm).sum(-1)[j]
            loc_j = torch.where(hit.unsqueeze(-1), loc_j, loc_j.new_zeros(loc_j.shape))
            feat[:, 0:2] = (loc_j / 40.0).clamp(-2, 2)
            feat[:, 2] = torch.where(hit, st.clamp(0, 2), st.new_zeros(()))
            feat[:, 3] = torch.where(hit, (nchg / 5.0).clamp(0, 2), nchg.new_zeros(()))
            feat[:, 4] = hit.float()
            feat[:, 5] = ((loc_j[:, 0] > 0.5) & hit).float()

    rg_xyz = static["roadgraph_xyz_world"].float()
    rg_type = static["roadgraph_type"].long()
    rg_valid = static["roadgraph_valid"].bool()
    if rg_xyz.ndim == 2 and rg_xyz.shape[0]:
        loc = local_xy(rg_xyz[:, :2].unsqueeze(0), origins[:, None, :], yaws[:, None])
        for slot, types, col in ((18, (18,), 6), (17, (17,), 9), (15, (15, 16), 12)):
            mask_t = torch.zeros(rg_type.shape[0], dtype=torch.bool)
            for tid in types:
                mask_t |= rg_type == tid
            useful = rg_valid & mask_t
            d = torch.linalg.vector_norm(loc, dim=-1)
            ahead = (loc[..., 0] > 0.5) & (loc[..., 1].abs() < 10.0)
            d = d.masked_fill(~useful.unsqueeze(0) | ~ahead, float("inf"))
            j = d.argmin(-1)
            hit = torch.isfinite(d[torch.arange(t), j])
            pt = loc[torch.arange(t), j]
            pt = torch.where(hit.unsqueeze(-1), pt, pt.new_zeros(pt.shape))
            feat[:, col:col + 2] = (pt / 40.0).clamp(-2, 2)
            feat[:, col + 2] = hit.float()
    return feat


def inter_token(neighbors, neighbor_valid, origins, yaws, hist, hist_valid, is_sdc) -> torch.Tensor:
    t = neighbors.shape[0]
    feat = neighbors.new_zeros(t, INTER_DIM)
    lead_w = neighbors[:, :, -1, 12].clamp(min=0) * neighbor_valid.float()
    has = lead_w.sum(1) > 0.5
    lw = lead_w.unsqueeze(-1)
    lead_xy = (neighbors[:, :, -1, :2] * lw).sum(1) / lw.sum(1).clamp_min(1e-6)
    lead_vel = (neighbors[:, :, -1, 3:5] * lw).sum(1) / lw.sum(1).clamp_min(1e-6)
    nspd = torch.linalg.vector_norm(neighbors[:, :, :, 3:5], dim=-1)
    lead_acc = ((nspd[:, :, -1] - nspd[:, :, 0]) * lead_w).sum(1) / lead_w.sum(1).clamp_min(1e-6)
    feat[:, 0] = has.float()
    feat[:, 1] = (lead_xy[:, 0] / 40.0).clamp(-2, 2)
    feat[:, 2] = (lead_vel[:, 0] / 15.0).clamp(-2, 2)
    feat[:, 3] = (lead_acc / 3.0).clamp(-2, 2)
    nxy = neighbors[:, :, -1, :2]
    for col, side in ((4, 1.0), (7, -1.0)):
        lat = nxy[..., 1] * side
        adj = neighbor_valid & (lat > 2.0) & (lat < 6.5) & (nxy[..., 0] > -15) & (nxy[..., 0] < 50)
        feat[:, col] = (adj.float().sum(1) / 5.0).clamp(0, 2)
        feat[:, col + 1] = adj.any(1).float()
    if is_sdc is not None and bool((is_sdc[: hist.shape[0]] & hist_valid[:, -1]).any()):
        ei = int((is_sdc[: hist.shape[0]] & hist_valid[:, -1]).nonzero()[0])
        ego_xy = local_xy(hist[ei, -1, :2].unsqueeze(0).expand(t, 2), origins, yaws)
        feat[:, 10:12] = (ego_xy / 50.0).clamp(-2, 2)
        feat[:, 12] = 1.0
    return feat


def _scene_v10(static, window, max_targets, train):
    hist_raw = window["inputs"]["agent_history_world"].float()
    hist_valid_in = window["inputs"].get("agent_history_valid")
    hist, valid = split_hist(hist_raw, hist_valid_in)
    sizes = window["inputs"]["agent_size_m"].float().clone()
    sizes[:, 0] = sizes[:, 0].clamp(0.0, 30.0)
    sizes[:, 1] = sizes[:, 1].clamp(0.0, 5.0)
    if sizes.shape[-1] >= 3:
        sizes[:, 2] = 0.0
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
    map_feat, map_valid = encode_map_points(
        static, origins, yaws, map_k=MAP_K, keep_types=KEEP_TYPES,
    )
    mapped = torch.zeros(t, dtype=torch.long)
    for src, dst in TYPE_TO_INDEX.items():
        mapped[types[rows] == src] = dst
    future = local_xy(future_world[rows, :, :2], origins[:, None, :], yaws[:, None])
    fv = future_valid[rows]
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
        "map_feat": map_feat,
        "map_valid": map_valid,
        "signals": dummy_sig,
        "type_idx": mapped,
        "future": future,
        "future_valid": fv,
        "agent_tok": agent,
        "lane_tok": lane,
        "lane_valid": lane_ok,
        "map_tok": mmap,
        "inter_tok": inter,
    }


class WindowSampleCollateV10:
    def __init__(self, max_targets=24, train=False):
        self.max_targets = max_targets
        self.train = train

    def __call__(self, batch):
        buckets = None
        for static, window in batch:
            packed = _scene_v10(static, window, self.max_targets, self.train)
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
                "map_feat": torch.zeros(0, MAP_K, 8),
                "map_valid": torch.zeros(0, MAP_K, dtype=torch.bool),
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
        out = {k: torch.cat(v, 0) for k, v in buckets.items()}
        for k, v in out.items():
            if torch.is_floating_point(v):
                out[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        return out


class MotionPredictorV10(nn.Module):
    def __init__(self, hidden=256, modes=6, nhead=4):
        super().__init__()
        self.modes = modes
        self.hidden = hidden
        self.type_emb = nn.Embedding(3, 16)
        self.target_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(55, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.hist_gru = nn.GRU(5, hidden, batch_first=True)
        self.hist_skip = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.hist_skip.weight)
        nn.init.zeros_(self.hist_skip.bias)
        self.neigh_step = nn.Sequential(nn.Linear(NEIGHBOR_HIST_DIM, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden))
        self.map_encoder = nn.Sequential(nn.Linear(8, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden))
        self.cross_attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.agent_enc = nn.Sequential(nn.Linear(AGENT_DIM, hidden), nn.GELU())
        self.lane_enc = nn.Sequential(nn.Linear(LANE_DIM, hidden), nn.GELU())
        self.map_tok_enc = nn.Sequential(nn.Linear(MAP_DIM, hidden), nn.GELU())
        self.inter_enc = nn.Sequential(nn.Linear(INTER_DIM, hidden), nn.GELU())
        self.mix_attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.mix_norm = nn.LayerNorm(hidden)
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3 + 16, hidden * 2), nn.GELU(), nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden), nn.GELU(),
        )
        self.goal_head = nn.Linear(hidden, modes * 2)
        self.delta_head = nn.Linear(hidden, modes * 80 * 2)
        self.mode_head = nn.Linear(hidden, modes)
        self.refine_gru = nn.GRU(2, hidden // 4, batch_first=True)
        self.refine_out = nn.Linear(hidden // 4, 2)
        nn.init.zeros_(self.refine_out.weight)
        nn.init.zeros_(self.refine_out.bias)

    def _pool_neighbors(self, neighbors):
        step_h = self.neigh_step(neighbors)
        step_valid = neighbors[..., 5:6]
        step_h = step_h * step_valid
        return step_h.sum(2) / step_valid.sum(2).clamp_min(1.0)

    def forward(self, target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
                agent_tok=None, lane_tok=None, lane_valid=None, map_tok=None, inter_tok=None):
        b = target_hist.shape[0]
        target = self.target_encoder(target_hist)
        gru_out, _ = self.hist_gru(target_hist)
        target = target + self.hist_skip(gru_out[:, -1])
        neigh_tok = self._pool_neighbors(neighbors)
        map_pts = self.map_encoder(map_feat)
        mem = torch.cat([neigh_tok, map_pts], dim=1)
        pad = torch.cat([~neighbor_valid, ~map_valid], dim=1)
        if pad.all(1).any():
            pad = pad.clone()
            pad[pad.all(1), 0] = False
        ctx, _ = self.cross_attn(target.unsqueeze(1), mem, mem, key_padding_mask=pad)
        ctx = self.attn_norm(ctx.squeeze(1) + target)
        extra = []
        extra_pad = []
        if agent_tok is not None:
            extra.append(self.agent_enc(agent_tok).unsqueeze(1))
            extra_pad.append(torch.zeros(b, 1, dtype=torch.bool, device=target.device))
        if lane_tok is not None:
            extra.append(self.lane_enc(lane_tok).unsqueeze(1))
            extra_pad.append(~lane_valid.unsqueeze(1) if lane_valid is not None else torch.zeros(b, 1, dtype=torch.bool, device=target.device))
        if map_tok is not None:
            extra.append(self.map_tok_enc(map_tok).unsqueeze(1))
            extra_pad.append(torch.zeros(b, 1, dtype=torch.bool, device=target.device))
        if inter_tok is not None:
            extra.append(self.inter_enc(inter_tok).unsqueeze(1))
            extra_pad.append(torch.zeros(b, 1, dtype=torch.bool, device=target.device))
        if extra:
            em = torch.cat(extra, 1)
            ep = torch.cat(extra_pad, 1)
            if ep.all(1).any():
                ep = ep.clone()
                ep[ep.all(1), 0] = False
            mix, _ = self.mix_attn(target.unsqueeze(1), em, em, key_padding_mask=ep)
            target = target + self.mix_norm(mix.squeeze(1))
        h = self.fusion(torch.cat([target, ctx, target, self.type_emb(type_idx)], dim=-1))
        goals = self.goal_head(h).view(b, self.modes, 2)
        deltas = self.delta_head(h).view(b, self.modes, 80, 2)
        raw = torch.cumsum(deltas, dim=2)
        time = torch.linspace(1 / 80, 1, 80, device=h.device, dtype=h.dtype).view(1, 1, 80, 1)
        traj = raw + time * (goals.unsqueeze(2) - raw[:, :, -1:])
        flat = traj.reshape(b * self.modes, 80, 2)
        ref, _ = self.refine_gru(flat)
        traj = traj + self.refine_out(ref).view(b, self.modes, 80, 2)
        traj = traj + time * (goals.unsqueeze(2) - traj[:, :, -1:])
        return traj, goals, self.mode_head(h)
