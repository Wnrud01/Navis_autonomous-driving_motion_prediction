#!/usr/bin/env python3
"""Motion Prediction V16: Two-Stage Proposal-to-Refinement Decoder.

Architecture:
1. Stage 1 (Proposal Generator - V13 Baseline):
   - VectorNet Polyline Subgraph Encoder (16 polylines x 20 points x 8 dim)
   - 6-Stage Human-like Cognitive Gating
   - 6 Learnable Mode Queries with Self-Attention & Cross-Attention into Scene Memory
   - Outputs: traj_1 [B, 6, 80, 2], goals_1 [B, 6, 2] (Diverse multi-modal proposals)

2. Stage 2 (Local Scene-Interactive Refinement & Re-Scoring):
   - Waypoint Feature Extractor: Samples 10 key waypoints along traj_1
   - Refine Query = h_modes_1 + traj_feat_1
   - Refine Transformer Layer (Self-Attn + Cross-Attn into Scene Memory)
   - Residual Offset Heads: d_traj [B, 6, 80, 2], d_goals [B, 6, 2]
   - Final Trajectory: traj_2 = traj_1 + d_traj, goals_2 = goals_1 + d_goals
   - Stage 2 Mode Scoring Heads: Outputs high-confidence logits [B, 6]
"""
from __future__ import annotations

import torch
from torch import nn

from src.train_motion_prediction_v13 import (
    MotionPredictorV13,
    POLY_FEAT_DIM,
    POLY_K,
    POLYLINE_POINTS,
    MODES,
)

WAYPOINT_INDICES = (7, 15, 23, 31, 39, 47, 55, 63, 71, 79)
N_WAYPOINTS = len(WAYPOINT_INDICES)
TRAJ_PT_DIM = 7  # x, y, dx, dy, speed, cos_heading, sin_heading


class MotionPredictorV16(MotionPredictorV13):
    def __init__(self, hidden: int = 256, modes: int = 6, nhead: int = 4):
        super().__init__(hidden=hidden, modes=modes, nhead=nhead)

        # ---------------------------------------------------------------------
        # Stage 2: Refinement & Re-Scoring Modules
        # ---------------------------------------------------------------------
        # 1. Waypoint PointNet Extractor on Stage 1 Trajectory
        self.wp_mlp = nn.Sequential(
            nn.Linear(TRAJ_PT_DIM, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
        )
        self.wp_norm = nn.LayerNorm(hidden)

        # 2. Stage 2 Refinement Transformer Decoder Layer
        self.refine_self_attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.refine_self_norm = nn.LayerNorm(hidden)
        self.refine_cross_attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.refine_cross_norm = nn.LayerNorm(hidden)
        self.refine_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )

        # 3. Stage 2 Type-Specific Residual Offset Heads
        self.stage2_goal_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 2),
            )
            for _ in range(3)
        ])
        self.stage2_delta_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 80 * 2),
            )
            for _ in range(3)
        ])

        # 4. Stage 2 Mode Scoring Heads (Evaluates refined representation)
        self.stage2_mode_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )
            for _ in range(3)
        ])

    def _extract_waypoint_features(self, traj: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale waypoint features along the 80-step predicted path.
        traj: [B, 6, 80, 2]
        Returns: [B, 6, hidden]
        """
        b, k, t, _ = traj.shape
        wp_indices = torch.tensor(WAYPOINT_INDICES, device=traj.device, dtype=torch.long)
        wp_xy = traj[:, :, wp_indices, :]  # [B, 6, 10, 2]

        diff = traj[:, :, 1:, :] - traj[:, :, :-1, :]
        diff_full = torch.cat([traj[:, :, :1, :], diff], dim=2)
        wp_dxy = diff_full[:, :, wp_indices, :]  # [B, 6, 10, 2]

        speed = torch.linalg.vector_norm(wp_dxy, dim=-1, keepdim=True) * 10.0
        heading = torch.atan2(wp_dxy[..., 1:2], wp_dxy[..., 0:1].clamp_min(1e-4))
        h_cos = torch.cos(heading)
        h_sin = torch.sin(heading)

        pt_raw = torch.cat([wp_xy, wp_dxy, speed / 20.0, h_cos, h_sin], dim=-1)
        pt_feat = self.wp_mlp(pt_raw.reshape(b * k * N_WAYPOINTS, TRAJ_PT_DIM))
        pt_feat = pt_feat.reshape(b, k, N_WAYPOINTS, self.hidden)

        traj_tok = pt_feat.max(dim=2).values  # [B, 6, hidden]
        return self.wp_norm(traj_tok)

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

        # ---------------------------------------------------------------------
        # 1. Target, Neighbor & Polyline Subgraph Encoders
        # ---------------------------------------------------------------------
        target = self.target_encoder(target_hist)
        gru_out, _ = self.hist_gru(target_hist)
        h = target + self.hist_skip(gru_out[:, -1])

        neigh_tok = self._pool_neighbors(neighbors)  # [B, 16, hidden]
        poly_tok = self._encode_polylines(map_feat, map_valid)  # [B, 16, hidden]

        # ---------------------------------------------------------------------
        # 2. Target-Centric Cross-Attention with Map & Neighbors
        # ---------------------------------------------------------------------
        mem = torch.cat([neigh_tok, poly_tok], dim=1)
        poly_pad = ~map_valid
        if poly_pad.shape[1] != POLY_K:
            poly_pad = poly_pad[:, :POLY_K] if poly_pad.shape[1] > POLY_K else torch.cat([poly_pad, poly_pad.new_ones(b, POLY_K - poly_pad.shape[1])], dim=1)
        pad = torch.cat([~neighbor_valid, poly_pad], dim=1)
        if pad.all(1).any():
            pad = pad.clone()
            pad[pad.all(1), 0] = False

        ctx, _ = self.cross_attn(h.unsqueeze(1), mem, mem, key_padding_mask=pad)
        ctx = ctx.squeeze(1)
        h = self.attn_norm(ctx + h)

        # ---------------------------------------------------------------------
        # 3. 6-Stage Human-like Cognitive Gating
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 4. Scene Memory Representation
        # ---------------------------------------------------------------------
        scene_feat = self.fusion(torch.cat([h, ctx, h, self.type_emb(type_idx)], dim=-1))
        scene_mem = torch.cat([scene_feat.unsqueeze(1), target.unsqueeze(1), neigh_tok, poly_tok], dim=1)  # [B, 34, hidden]
        scene_mem_pad = torch.cat([zeros_b.unsqueeze(1), zeros_b.unsqueeze(1), ~neighbor_valid, poly_pad], dim=1)
        if scene_mem_pad.all(1).any():
            scene_mem_pad = scene_mem_pad.clone()
            scene_mem_pad[scene_mem_pad.all(1), 0] = False

        # ---------------------------------------------------------------------
        # 5. Stage 1: Proposal Generator (V13 Mechanism)
        # ---------------------------------------------------------------------
        queries = self.mode_queries.expand(b, -1, -1)  # [B, 6, hidden]
        q_self, _ = self.mode_self_attn(queries, queries, queries)
        queries = self.mode_self_norm(queries + q_self)

        q_cross, _ = self.mode_cross_attn(queries, scene_mem, scene_mem, key_padding_mask=scene_mem_pad)
        h_modes_1 = self.mode_cross_norm(queries + q_cross)
        h_modes_1 = h_modes_1 + self.mode_ffn(h_modes_1)  # [B, 6, hidden]

        # Stage 1 Rollout
        goals_1 = self._type_mode_output(self.goal_heads, h_modes_1, type_idx).view(b, self.modes, 2)
        deltas_1 = self._type_mode_output(self.delta_heads, h_modes_1, type_idx).view(b, self.modes, 80, 2)
        traj_1 = torch.cumsum(deltas_1, dim=2)

        flat_1 = traj_1.reshape(b * self.modes, 80, 2)
        ref_1, _ = self.refine_gru(flat_1)
        traj_1 = traj_1 + self.refine_out(ref_1).view(b, self.modes, 80, 2)  # [B, 6, 80, 2]

        # ---------------------------------------------------------------------
        # 6. Stage 2: Local Scene-Interactive Refinement & Re-Scoring
        # ---------------------------------------------------------------------
        # Extract features of Stage 1 proposals (differentiable end-to-end)
        traj_feat_1 = self._extract_waypoint_features(traj_1)  # [B, 6, hidden]
        q_refine = h_modes_1 + traj_feat_1                     # [B, 6, hidden]

        # Stage 2 Mode Self-Attention
        r_self, _ = self.refine_self_attn(q_refine, q_refine, q_refine)
        q_refine = self.refine_self_norm(q_refine + r_self)

        # Stage 2 Cross-Attention into Scene Memory
        r_cross, _ = self.refine_cross_attn(q_refine, scene_mem, scene_mem, key_padding_mask=scene_mem_pad)
        h_modes_2 = self.refine_cross_norm(q_refine + r_cross)
        h_modes_2 = h_modes_2 + self.refine_ffn(h_modes_2)  # [B, 6, hidden]

        # Stage 2 Residual Offsets
        d_goals = self._type_mode_output(self.stage2_goal_heads, h_modes_2, type_idx).view(b, self.modes, 2)
        d_deltas = self._type_mode_output(self.stage2_delta_heads, h_modes_2, type_idx).view(b, self.modes, 80, 2)
        d_traj = torch.cumsum(d_deltas, dim=2)

        # Final Refined Trajectories & Goals
        goals_2 = goals_1 + d_goals
        traj_2 = traj_1 + d_traj

        # Stage 2 Mode Logits (Confidence scoring)
        logits_2 = self._type_mode_output(self.stage2_mode_heads, h_modes_2, type_idx).squeeze(-1)  # [B, 6]

        return traj_2, goals_2, logits_2, traj_1, goals_1
