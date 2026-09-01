#!/usr/bin/env python3
"""Motion Prediction V13: Polyline Map Encoder + Mode Query Cross-Attention Decoder.

Key Innovations:
1. Method 4: VectorNet Subgraph Polyline Map Encoder (16 polylines x 20 points).
2. Method 1: Mode Query Cross-Attention Decoder (6 learnable queries with self/cross attention).
3. Type-specific decoder heads (Vehicle, Pedestrian, Cyclist).
4. Full temporal rollout with cumsum + Refine GRU.
"""
from __future__ import annotations

import torch
from torch import nn

from src.train_motion_prediction_v12 import MotionPredictorV12

POLYLINE_POINTS = 20
POLY_K = 16
POLY_FEAT_DIM = 8
MODES = 6


class MotionPredictorV13(MotionPredictorV12):
    def __init__(self, hidden: int = 256, modes: int = 6, nhead: int = 4):
        super().__init__(hidden=hidden, modes=modes, nhead=nhead)
        self.modes = modes
        self.hidden = hidden

        # 1. Polyline PointNet Subgraph Encoder
        self.poly_pt_mlp = nn.Sequential(
            nn.Linear(POLY_FEAT_DIM, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
        )
        self.poly_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.poly_norm = nn.LayerNorm(hidden)

        # 2. Learnable Mode Queries
        self.mode_queries = nn.Parameter(torch.randn(1, modes, hidden) * 0.02)

        # 3. Mode Query Transformer Decoder Layers
        self.mode_self_attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.mode_self_norm = nn.LayerNorm(hidden)
        self.mode_cross_attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.mode_cross_norm = nn.LayerNorm(hidden)
        self.mode_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )

        # 4. Mode-specific Type Heads (Operating on [B, 6, hidden])
        # Each head maps from [B, 6, hidden] -> [B, 6, OutputDim]
        self.goal_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 2))
            for _ in range(3)
        ])
        self.delta_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 80 * 2))
            for _ in range(3)
        ])
        self.mode_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
            for _ in range(3)
        ])

    def _encode_polylines(self, polylines: torch.Tensor, poly_valid: torch.Tensor) -> torch.Tensor:
        """VectorNet Subgraph Encoder:
        polylines: [B, 16, 20, 8] or fallback [B, 64, 8]
        poly_valid: [B, 16] or [B, 64]
        Returns: [B, 16, hidden]
        """
        b = polylines.shape[0]
        if polylines.dim() == 3:
            # Fallback for point-cloud map_feat [B, 64, 8]
            pts_h = self.poly_pt_mlp(polylines)
            if pts_h.shape[1] >= POLY_K:
                pts_h = pts_h[:, :POLY_K]
            else:
                pad_k = POLY_K - pts_h.shape[1]
                pts_h = torch.cat([pts_h, pts_h.new_zeros(b, pad_k, self.hidden)], dim=1)
            return self.poly_norm(self.poly_mlp(pts_h))

        # Standard [B, 16, 20, 8]
        b, k, p, d = polylines.shape
        flat_pts = polylines.reshape(b * k * p, d)
        pt_feat = self.poly_pt_mlp(flat_pts).reshape(b, k, p, self.hidden)

        # Mask invalid points (column 4 is sel_valid_pts)
        if d >= 5:
            pt_valid = polylines[..., 4:5]
            pt_feat = pt_feat * pt_valid

        # MaxPool across the 20 points in each polyline
        poly_tok = pt_feat.max(dim=2).values  # [B, 16, hidden]
        poly_tok = self.poly_norm(self.poly_mlp(poly_tok))

        # Mask out completely invalid polylines
        poly_tok = poly_tok * poly_valid.unsqueeze(-1).to(poly_tok.dtype)
        return poly_tok

    def _type_mode_output(self, heads: nn.ModuleList, h_modes: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
        """h_modes: [B, 6, hidden]
        Returns: [B, 6, OutputDim]
        """
        b = h_modes.shape[0]
        idx = type_idx.clamp(0, 2)
        # Compute for each of 3 types
        stacked = torch.stack([head(h_modes) for head in heads], dim=1)  # [B, 3, 6, OutDim]
        gather_idx = idx[:, None, None, None].expand(b, 1, self.modes, stacked.shape[-1])
        return stacked.gather(1, gather_idx).squeeze(1)

    def forward(
        self,
        target_hist: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_valid: torch.Tensor,
        map_feat: torch.Tensor,
        map_valid: torch.Tensor,
        signals: torch.Tensor,
        type_idx: torch.Tensor,
        agent_tok: torch.Tensor | None = None,
        lane_tok: torch.Tensor | None = None,
        lane_valid: torch.Tensor | None = None,
        map_tok: torch.Tensor | None = None,
        inter_tok: torch.Tensor | None = None,
    ):
        b = target_hist.shape[0]

        # 1. Target Encoding
        target = self.target_encoder(target_hist)
        gru_out, _ = self.hist_gru(target_hist)
        h = target + self.hist_skip(gru_out[:, -1])

        # 2. Neighbor and Polyline Encoding
        neigh_tok = self._pool_neighbors(neighbors)  # [B, 16, hidden]
        poly_tok = self._encode_polylines(map_feat, map_valid)  # [B, 16, hidden]

        # 3. Target-Centric Cross-Attention with Neighbors + Polylines
        mem = torch.cat([neigh_tok, poly_tok], dim=1)  # [B, 32, hidden]
        poly_pad = ~map_valid
        if poly_pad.shape[1] != POLY_K:
            poly_pad = poly_pad[:, :POLY_K] if poly_pad.shape[1] > POLY_K else torch.cat([poly_pad, poly_pad.new_ones(b, POLY_K - poly_pad.shape[1])], dim=1)
        pad = torch.cat([~neighbor_valid, poly_pad], dim=1)  # [B, 32]
        if pad.all(1).any():
            pad = pad.clone()
            pad[pad.all(1), 0] = False

        ctx, _ = self.cross_attn(h.unsqueeze(1), mem, mem, key_padding_mask=pad)
        ctx = ctx.squeeze(1)
        h = self.attn_norm(ctx + h)

        # 4. 6-Stage Human-like Cognitive Gating
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

        h = self._fuse_stage(0, h, speed, ones_b)
        h = self._fuse_stage(1, h, lane, lane_valid)
        h = self._fuse_stage(2, h, lead, lead_ok)
        h = self._fuse_stage(3, h, sig, sig_ok)
        h = self._fuse_stage(4, h, road, road_ok)
        h = self._fuse_stage(5, h, adj, adj_ok)

        # 5. Scene Representation
        scene_feat = self.fusion(torch.cat([h, ctx, h, self.type_emb(type_idx)], dim=-1))  # [B, hidden]

        # 6. Mode Query Cross-Attention Decoder (Method 1)
        queries = self.mode_queries.expand(b, -1, -1)  # [B, 6, hidden]

        # Mode Self-Attention (modes communicate and enforce multi-modality)
        q_self, _ = self.mode_self_attn(queries, queries, queries)
        queries = self.mode_self_norm(queries + q_self)

        # Mode Cross-Attention into Multi-Token Scene Memory
        scene_mem = torch.cat([scene_feat.unsqueeze(1), target.unsqueeze(1), neigh_tok, poly_tok], dim=1)  # [B, 34, hidden]
        scene_mem_pad = torch.cat([zeros_b.unsqueeze(1), zeros_b.unsqueeze(1), ~neighbor_valid, poly_pad], dim=1)  # [B, 34]
        if scene_mem_pad.all(1).any():
            scene_mem_pad = scene_mem_pad.clone()
            scene_mem_pad[scene_mem_pad.all(1), 0] = False

        q_cross, _ = self.mode_cross_attn(queries, scene_mem, scene_mem, key_padding_mask=scene_mem_pad)
        h_modes = self.mode_cross_norm(queries + q_cross)  # [B, 6, hidden]
        h_modes = h_modes + self.mode_ffn(h_modes)  # [B, 6, hidden]

        # 7. Type-Specific Decoding of 6 Modes
        goals = self._type_mode_output(self.goal_heads, h_modes, type_idx).view(b, self.modes, 2)
        deltas = self._type_mode_output(self.delta_heads, h_modes, type_idx).view(b, self.modes, 80, 2)
        traj = torch.cumsum(deltas, dim=2)  # [B, 6, 80, 2]

        flat = traj.reshape(b * self.modes, 80, 2)
        ref, _ = self.refine_gru(flat)
        traj = traj + self.refine_out(ref).view(b, self.modes, 80, 2)

        logits = self._type_mode_output(self.mode_heads, h_modes, type_idx).squeeze(-1)  # [B, 6]
        return traj, goals, logits
