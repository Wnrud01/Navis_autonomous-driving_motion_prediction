#!/usr/bin/env python3
"""V9 interaction tokens on V6: lane+signal+lead, signal history, adjacent cut-in,
stop/crosswalk, ego. Height is not used. New layers zero-gated so V6 is unchanged
at step 0.
"""
from __future__ import annotations

import torch
from torch import nn

from src.train_motion_prediction_v1 import local_vec, local_xy
from src.train_motion_prediction_v3 import SIGNAL_K, wrap_angle, _scene_targets_v3
from src.train_motion_prediction_v8 import (
    HIST_STEPS,
    KIN_DIM,
    LANE_SIG_DIM,
    MAP_K,
    NEIGHBOR_HIST_DIM,
    POLY_K,
    MotionPredictorV8,
    encode_lane_signals,
    target_kinematics,
)

INTER_DIM = 24
INTER_N = 14  # 6 specials + 8 signal-history
SIG_HIST_K = 8
MAP_RANGE = 80.0


def _clip(x, s, cap=2.0):
    return (x / s).clamp(-cap, cap)


def encode_interact(
    static: dict,
    window: dict,
    origins: torch.Tensor,
    yaws: torch.Tensor,
    neighbors: torch.Tensor,
    neighbor_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build mixed interaction tokens in the target frame. [T, 14, 24]."""
    t = int(origins.shape[0])
    device = origins.device
    dtype = origins.dtype
    tokens = origins.new_zeros(t, INTER_N, INTER_DIM)
    valid = torch.zeros(t, INTER_N, dtype=torch.bool, device=device)

    hist = window["inputs"]["agent_history_world"].float()
    hist_valid = window["inputs"]["agent_history_valid"].bool()
    types = static["agent_types"].long()
    is_sdc = static["agent_is_sdc"].bool()
    n_agents = hist.shape[0]
    sig = window["inputs"]["signal_history_world_state"].float()
    sig_valid = window["inputs"]["signal_history_valid"].bool()

    # --- 1. my lane + forward light + lead ---
    my = tokens[:, 0]
    if "map_polyline_xy" in static and static["map_polyline_xy"].numel():
        xy = static["map_polyline_xy"].float()
        direc = static["map_polyline_dir"].float()
        pvalid = static["map_polyline_valid"].bool()
        ptype = static["map_polyline_type"].float()
        m = int(xy.shape[0])
        local_pts = local_xy(xy.unsqueeze(0), origins[:, None, None, :], yaws[:, None, None])
        local_dir = local_vec(direc.unsqueeze(0), yaws[:, None, None])
        w = pvalid.unsqueeze(0).to(dtype)
        dist = torch.linalg.vector_norm(local_pts, dim=-1).masked_fill(~pvalid.unsqueeze(0), float("inf"))
        lat = local_pts[..., 1].abs().masked_fill(~pvalid.unsqueeze(0), 1.0e6)
        min_lat = lat.min(dim=-1).values
        min_d = dist.min(dim=-1).values
        mean_dir = (local_dir * w.unsqueeze(-1)).sum(dim=2)
        mean_dir = mean_dir / torch.linalg.vector_norm(mean_dir, dim=-1, keepdim=True).clamp_min(1e-6)
        yaw_err = wrap_angle(torch.atan2(mean_dir[..., 1], mean_dir[..., 0]))
        is_lane = ((ptype == 1) | (ptype == 2) | (ptype == 3)).float().unsqueeze(0)
        ahead = (local_pts[..., 0] > 0.5).any(dim=-1).float()
        score = -min_lat + 4.0 * is_lane - 2.0 * yaw_err.abs() + 2.0 * ahead
        score = score.masked_fill(min_d > MAP_RANGE, -1.0e9)
        best = score.argmax(dim=-1)
        rows = torch.arange(t, device=device)
        hit = score[rows, best] > -1.0e8
        sel_pts = local_pts[rows, best]
        sel_valid = pvalid[best]
        centroid = (sel_pts * sel_valid.unsqueeze(-1).to(dtype)).sum(dim=1) / sel_valid.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
        end_idx = (sel_valid.float().sum(dim=-1).long() - 1).clamp(min=0)
        end_xy = sel_pts[rows, end_idx]
        sdir = mean_dir[rows, best]
        remain = torch.linalg.vector_norm(end_xy, dim=-1)
        i0 = dist[rows, best].argmin(dim=-1)
        j = (i0 + 5).clamp(max=sel_pts.shape[1] - 1)
        h0 = torch.atan2(local_dir[rows, best, i0, 1], local_dir[rows, best, i0, 0])
        h1 = torch.atan2(local_dir[rows, best, j, 1], local_dir[rows, best, j, 0])
        curv = wrap_angle(h1 - h0) / 5.0

        my[:, 0:2] = _clip(centroid, 40.0)
        my[:, 2:4] = _clip(end_xy, 40.0)
        my[:, 4:6] = sdir
        my[:, 6] = _clip(min_lat[rows, best], 4.0)
        my[:, 7] = _clip(yaw_err[rows, best], 1.0)
        my[:, 8] = _clip(remain, 80.0)
        my[:, 9] = _clip(curv, 0.3)
        valid[hit, 0] = True

        n_sig = int(sig.shape[0]) if sig is not None else 0
        if n_sig > 0:
            s_local = local_xy(sig[:, -1, :2].unsqueeze(0), origins[:, None, :], yaws[:, None])
            s_ok = sig_valid[:, -1].unsqueeze(0) if sig_valid.numel() else torch.ones(1, n_sig, dtype=torch.bool, device=device)
            cone = (s_local[..., 0] > 0.5) & (s_local[..., 1].abs() < 12.0) & s_ok
            d_lane = torch.linalg.vector_norm(s_local - centroid.unsqueeze(1), dim=-1)
            d_lane = d_lane.masked_fill(~cone, float("inf"))
            jsig = d_lane.argmin(dim=-1)
            sh = torch.isfinite(d_lane[rows, jsig])
            my[:, 10:12] = _clip(s_local[rows, jsig], 40.0)
            last_st = sig[:, -1, 2][jsig]
            my[:, 12] = (last_st / 8.0).clamp(0, 2)
            if sig.shape[1] > 1:
                chg = (sig[:, 1:, 2] != sig[:, :-1, 2]).float()
                nchg = (chg * sig_valid[:, 1:].float()).sum(dim=1)
                last_chg = chg[:, -1]
                my[:, 13] = last_chg[jsig]
                my[:, 14] = _clip(nchg[jsig], 5.0)
            my[:, 15] = (s_local[rows, jsig, 0] > 0.5).to(dtype)
            valid[hit & sh, 0] = True

    lead_w = neighbors[:, :, -1, 12].clamp(min=0.0) * neighbor_valid.float()
    has_lead = lead_w.sum(dim=1) > 0.5
    lw = lead_w.unsqueeze(-1)
    lead_xy = (neighbors[:, :, -1, 0:2] * lw).sum(dim=1) / lw.sum(dim=1).clamp_min(1e-6)
    lead_vel = (neighbors[:, :, -1, 3:5] * lw).sum(dim=1) / lw.sum(dim=1).clamp_min(1e-6)
    n_speed = torch.linalg.vector_norm(neighbors[:, :, :, 3:5], dim=-1)
    lead_acc = ((n_speed[:, :, -1] - n_speed[:, :, 0]) * lead_w).sum(dim=1) / lead_w.sum(dim=1).clamp_min(1e-6)
    my[:, 16] = has_lead.to(dtype)
    my[:, 17] = _clip(lead_xy[:, 0], 40.0)
    my[:, 18] = _clip(lead_vel[:, 0], 15.0)
    my[:, 19] = _clip(lead_acc, 3.0)
    my[:, 20] = _clip(lead_xy[:, 1], 4.0)
    valid[:, 0] = valid[:, 0] | has_lead

    # --- 3. adjacent occupancy + cut-in (left=1, right=2) ---
    nxy = neighbors[:, :, -1, 0:2]
    nvel = neighbors[:, :, -1, 3:5]
    nv = neighbor_valid
    for slot, side in ((1, 1.0), (2, -1.0)):
        lat = nxy[..., 1] * side
        adj = nv & (lat > 2.0) & (lat < 6.0) & (nxy[..., 0] > -15.0) & (nxy[..., 0] < 50.0)
        n_occ = adj.float().sum(dim=1)
        dist = torch.linalg.vector_norm(nxy, dim=-1).masked_fill(~adj, float("inf"))
        jn = dist.argmin(dim=-1)
        hit_n = torch.isfinite(dist[torch.arange(t, device=device), jn])
        gather_t = torch.arange(t, device=device)
        cutin = (-nvel[gather_t, jn, 1] * side)  # toward ego
        tok = tokens[:, slot]
        tok[:, 0] = _clip(n_occ, 5.0)
        tok[:, 1] = _clip(nxy[gather_t, jn, 0], 40.0)
        tok[:, 2] = _clip(nxy[gather_t, jn, 1], 6.0)
        tok[:, 3] = _clip(nvel[gather_t, jn, 0], 15.0)
        tok[:, 4] = _clip(cutin, 4.0)
        tok[:, 5] = hit_n.to(dtype)
        tok[:, 6] = _clip(torch.linalg.vector_norm(nvel[gather_t, jn], dim=-1), 15.0)
        valid[:, slot] = n_occ > 0

    # --- 4. stop line / crosswalk ---
    rg_xyz = static["roadgraph_xyz_world"].float()
    rg_type = static["roadgraph_type"].long()
    rg_valid = static["roadgraph_valid"].bool()
    if rg_xyz.ndim == 2 and rg_xyz.shape[0]:
        local_rg = local_xy(rg_xyz[:, :2].unsqueeze(0), origins[:, None, :], yaws[:, None])
        for slot, type_ids in ((3, (15, 16)), (4, (19,))):
            ok_t = torch.zeros(rg_type.shape[0], dtype=torch.bool)
            for tid in type_ids:
                ok_t |= rg_type == tid
            useful = rg_valid & ok_t
            d = torch.linalg.vector_norm(local_rg, dim=-1)
            ahead = (local_rg[..., 0] > 0.5) & (local_rg[..., 1].abs() < 8.0)
            d = d.masked_fill(~useful.unsqueeze(0) | ~ahead, float("inf"))
            if int(useful.sum()) == 0:
                continue
            j = d.argmin(dim=-1)
            hit = torch.isfinite(d[torch.arange(t, device=device), j])
            pt = local_rg[torch.arange(t, device=device), j]
            tok = tokens[:, slot]
            tok[:, 0:2] = _clip(pt, 40.0)
            tok[:, 2] = _clip(d[torch.arange(t, device=device), j].clamp(max=MAP_RANGE), 80.0)
            tok[:, 3] = hit.to(dtype)
            tok[:, 4] = (pt[:, 0] > 0.5).to(dtype)
            valid[:, slot] = hit

    # --- 5. ego (no height) ---
    sdc = is_sdc[:n_agents] & hist_valid[:, -1]
    if bool(sdc.any()):
        ego_i = int(sdc.nonzero()[0])
        ego_xy = local_xy(hist[ego_i, -1, :2].unsqueeze(0).expand(t, 2), origins, yaws)
        ego_vel = local_vec(hist[ego_i, -1, 3:5].unsqueeze(0).expand(t, 2), yaws)
        dx, dy = ego_xy[:, 0], ego_xy[:, 1]
        bearing = wrap_angle(torch.atan2(dy, dx))
        tok = tokens[:, 5]
        tok[:, 0:2] = _clip(ego_xy, 50.0)
        tok[:, 2:4] = _clip(ego_vel, 15.0)
        tok[:, 4] = _clip(torch.linalg.vector_norm(ego_vel, dim=-1), 15.0)
        tok[:, 5] = _clip(bearing, 3.14)
        tok[:, 6] = 1.0
        valid[:, 5] = True

    # --- 2. signal 11-step history (8 lights) ---
    n_sig = int(sig.shape[0]) if sig is not None else 0
    if n_sig > 0:
        s_local = local_xy(sig[:, -1, :2].unsqueeze(0), origins[:, None, :], yaws[:, None])
        sdist = torch.linalg.vector_norm(s_local, dim=-1)
        if sig_valid.numel():
            sdist = sdist.masked_fill(~sig_valid[:, -1].unsqueeze(0), float("inf"))
        k = min(SIG_HIST_K, n_sig)
        nidx = torch.topk(sdist, k=k, largest=False).indices
        gt = torch.arange(t, device=device).unsqueeze(1)
        sel_xy = s_local[gt, nidx]
        sel_d = sdist[gt, nidx]
        sel_ok = torch.isfinite(sel_d)
        last_st = sig[:, -1, 2][nidx]
        if sig.shape[1] > 1:
            chg = (sig[:, 1:, 2] != sig[:, :-1, 2]).float()
            nchg = (chg * (sig_valid[:, 1:].float() if sig_valid.numel() else 1.0)).sum(dim=1)
            last_chg = chg[:, -1]
            prev_st = sig[:, -2, 2][nidx] if sig.shape[1] > 1 else last_st
        else:
            nchg = sig.new_zeros(n_sig)
            last_chg = sig.new_zeros(n_sig)
            prev_st = last_st
        blk = tokens[:, 6:6 + k]
        blk[:, :, 0:2] = _clip(sel_xy, 40.0)
        blk[:, :, 2] = (last_st / 8.0).clamp(0, 2)
        blk[:, :, 3] = (prev_st / 8.0).clamp(0, 2)
        blk[:, :, 4] = last_chg[nidx]
        blk[:, :, 5] = _clip(nchg[nidx], 5.0)
        blk[:, :, 6] = _clip(sel_d.clamp(max=MAP_RANGE), 80.0)
        blk[:, :, 7] = (sel_xy[..., 0] > 0.5).to(dtype)
        valid[:, 6:6 + k] = sel_ok

    tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)
    return tokens, valid


def window_to_samples_v9(batch, max_targets=24, neighbor_k=16, signal_k=SIGNAL_K, map_k=MAP_K, poly_k=POLY_K, train=False):
    keys = (
        "target_hist", "neighbors", "neighbor_valid", "map_feat", "map_valid",
        "signals", "type_idx", "future", "future_valid",
        "kin_feat", "lane_sig", "lane_sig_valid", "interact", "interact_valid",
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
        interact, interact_ok = encode_interact(static, window, origins, yaws, nfeat, nvalid)
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
        buckets["interact"].append(interact)
        buckets["interact_valid"].append(interact_ok)
    if not buckets["future"]:
        return None
    return {key: torch.cat(value, dim=0) for key, value in buckets.items()}


class WindowSampleCollateV9:
    def __init__(self, max_targets=24, neighbor_k=16, signal_k=SIGNAL_K, map_k=MAP_K, poly_k=POLY_K, train=False):
        self.max_targets = max_targets
        self.neighbor_k = neighbor_k
        self.signal_k = signal_k
        self.map_k = map_k
        self.poly_k = poly_k
        self.train = train

    def __call__(self, batch):
        samples = window_to_samples_v9(
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
            "interact": torch.zeros(0, INTER_N, INTER_DIM),
            "interact_valid": torch.zeros(0, INTER_N, dtype=torch.bool),
        }


class MotionPredictorV9(MotionPredictorV8):
    def __init__(self, hidden: int = 256, modes: int = 6, nhead: int = 4):
        super().__init__(hidden=hidden, modes=modes, nhead=nhead)
        self.interact_encoder = nn.Sequential(
            nn.Linear(INTER_DIM, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, hidden),
        )

    def encode_context(
        self, target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
        kin_feat=None, lane_sig=None, lane_sig_valid=None, interact=None, interact_valid=None,
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
        extra_pad = [torch.zeros(signals.shape[0], signals.shape[1], dtype=torch.bool, device=h.device)]
        if lane_sig is not None:
            extra_toks.append(self.lane_sig_encoder(lane_sig))
            extra_pad.append(~lane_sig_valid if lane_sig_valid is not None else torch.zeros(lane_sig.shape[:2], dtype=torch.bool, device=h.device))
        if interact is not None:
            extra_toks.append(self.interact_encoder(interact))
            extra_pad.append(~interact_valid if interact_valid is not None else torch.zeros(interact.shape[:2], dtype=torch.bool, device=h.device))
        extra_mem = torch.cat(extra_toks, dim=1)
        extra_mask = torch.cat(extra_pad, dim=1)
        if extra_mask.all(dim=1).any():
            extra_mask = extra_mask.clone()
            extra_mask[extra_mask.all(dim=1), 0] = False
        mix, _ = self.mix_attn(target.unsqueeze(1), extra_mem, extra_mem, key_padding_mask=extra_mask)
        return h + self.mix_gate * self.mix_norm(mix.squeeze(1))

    def forward(
        self, target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
        kin_feat=None, lane_sig=None, lane_sig_valid=None, interact=None, interact_valid=None,
    ):
        h = self.encode_context(
            target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
            kin_feat=kin_feat, lane_sig=lane_sig, lane_sig_valid=lane_sig_valid,
            interact=interact, interact_valid=interact_valid,
        )
        return self.decode_heads(h)
