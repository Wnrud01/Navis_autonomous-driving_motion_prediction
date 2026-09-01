#!/usr/bin/env python3
"""V12: ordered token fusion + per-type decoder heads.

Human-like stages (not a permutation of one attention):
  speed → lane → lead → signal → roadside → adjacent
Each stage is its own gated cross-attn, so earlier context conditions later tokens.

Type-specific goal/delta/mode heads (vehicle / pedestrian / cyclist) so a
pedestrian is not trained by vehicle-scale deltas.
"""
from __future__ import annotations

import torch
from torch import nn

from src.train_motion_prediction_v11 import MotionPredictorV11


class MotionPredictorV12(MotionPredictorV11):
    def __init__(self, hidden=256, modes=6, nhead=4):
        super().__init__(hidden=hidden, modes=modes, nhead=nhead)
        self.n_stages = 6
        self.speed_enc = nn.Sequential(nn.Linear(6, hidden), nn.GELU())
        self.lead_enc = nn.Sequential(nn.Linear(4, hidden), nn.GELU())
        self.sig_enc = nn.Sequential(nn.Linear(6, hidden), nn.GELU())
        self.road_enc = nn.Sequential(nn.Linear(10, hidden), nn.GELU())
        self.adj_enc = nn.Sequential(nn.Linear(6, hidden), nn.GELU())
        self.stage_attn = nn.ModuleList(
            [nn.MultiheadAttention(hidden, nhead, batch_first=True) for _ in range(self.n_stages)]
        )
        self.stage_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(self.n_stages)])
        self.stage_gate = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid()) for _ in range(self.n_stages)]
        )
        self.goal_heads = nn.ModuleList([nn.Linear(hidden, modes * 2) for _ in range(3)])
        self.delta_heads = nn.ModuleList([nn.Linear(hidden, modes * 80 * 2) for _ in range(3)])
        self.mode_heads = nn.ModuleList([nn.Linear(hidden, modes) for _ in range(3)])

    def _fuse_stage(self, i, h, tok, valid):
        b = h.shape[0]
        if valid is None:
            valid = torch.ones(b, dtype=torch.bool, device=h.device)
        pad = (~valid).unsqueeze(1)
        if bool(pad.all()):
            return h
        if bool(pad.all(1).any()):
            pad = pad.clone()
            pad[pad.all(1), 0] = False
        out, _ = self.stage_attn[i](h.unsqueeze(1), tok, tok, key_padding_mask=pad)
        g = self.stage_gate[i](h)
        return h + g * self.stage_norm[i](out.squeeze(1))

    def _type_head(self, heads, h, type_idx):
        stacked = torch.stack([head(h) for head in heads], dim=1)
        idx = type_idx.clamp(0, 2)
        return stacked[torch.arange(h.shape[0], device=h.device), idx]

    def forward(self, target_hist, neighbors, neighbor_valid, map_feat, map_valid, signals, type_idx,
                agent_tok=None, lane_tok=None, lane_valid=None, map_tok=None, inter_tok=None):
        b = target_hist.shape[0]
        target = self.target_encoder(target_hist)
        gru_out, _ = self.hist_gru(target_hist)
        h = target + self.hist_skip(gru_out[:, -1])
        neigh_tok = self._pool_neighbors(neighbors)
        map_pts = self.map_encoder(map_feat)
        mem = torch.cat([neigh_tok, map_pts], dim=1)
        pad = torch.cat([~neighbor_valid, ~map_valid], dim=1)
        if pad.all(1).any():
            pad = pad.clone()
            pad[pad.all(1), 0] = False
        ctx, _ = self.cross_attn(h.unsqueeze(1), mem, mem, key_padding_mask=pad)
        ctx = ctx.squeeze(1)
        h = self.attn_norm(ctx + h)

        zeros_b = torch.zeros(b, dtype=torch.bool, device=h.device)
        ones_b = torch.ones(b, dtype=torch.bool, device=h.device)
        if agent_tok is None:
            agent_tok = h.new_zeros(b, 16)
        if lane_tok is None:
            lane_tok = h.new_zeros(b, 8)
            lane_valid = zeros_b
        if map_tok is None:
            map_tok = h.new_zeros(b, 16)
        if inter_tok is None:
            inter_tok = h.new_zeros(b, 16)
        if lane_valid is None:
            lane_valid = zeros_b

        speed = self.speed_enc(agent_tok[:, :6]).unsqueeze(1)
        lane = self.lane_enc(lane_tok).unsqueeze(1)
        lead = self.lead_enc(inter_tok[:, :4]).unsqueeze(1)
        sig = self.sig_enc(map_tok[:, :6]).unsqueeze(1)
        road = self.road_enc(map_tok[:, 6:16]).unsqueeze(1)
        adj = self.adj_enc(torch.cat([inter_tok[:, 4:8], inter_tok[:, 10:12]], dim=-1)).unsqueeze(1)
        lead_ok = inter_tok[:, 0] > 0.5
        sig_ok = map_tok[:, 4] > 0.5
        road_ok = (map_tok[:, 8] + map_tok[:, 11] + map_tok[:, 14]) > 0.5
        adj_ok = (inter_tok[:, 5] + inter_tok[:, 8] + inter_tok[:, 12]) > 0.5

        # speed → lane → lead → signal → roadside → adjacent
        h = self._fuse_stage(0, h, speed, ones_b)
        h = self._fuse_stage(1, h, lane, lane_valid)
        h = self._fuse_stage(2, h, lead, lead_ok)
        h = self._fuse_stage(3, h, sig, sig_ok)
        h = self._fuse_stage(4, h, road, road_ok)
        h = self._fuse_stage(5, h, adj, adj_ok)

        h = self.fusion(torch.cat([h, ctx, h, self.type_emb(type_idx)], dim=-1))
        goals = self._type_head(self.goal_heads, h, type_idx).view(b, self.modes, 2)
        deltas = self._type_head(self.delta_heads, h, type_idx).view(b, self.modes, 80, 2)
        traj = torch.cumsum(deltas, dim=2)
        flat = traj.reshape(b * self.modes, 80, 2)
        ref, _ = self.refine_gru(flat)
        traj = traj + self.refine_out(ref).view(b, self.modes, 80, 2)
        logits = self._type_head(self.mode_heads, h, type_idx)
        return traj, goals, logits
