#!/usr/bin/env python3
"""Motion Prediction V14: VectorNet Map + Mode Query Generator + Trajectory-Aware Dense Scorer.

Key Innovations:
1. Generator: V13 VectorNet Polyline Subgraph Encoder + 6-Stage Gated Context + Mode Query Cross-Attention.
2. Method 1 (Scorer): Trajectory-Aware Dense Cross-Attention Scorer.
   - Evaluates the generated 80-step rollout against actual road polylines and dynamic neighbors.
   - Extracts waypoint features along the path and performs cross-attention with scene memory.
3. Method 2 (Loss): ADE-Proportional Margin Ranking Loss in AdaptiveWTALossV14.
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

# 10 key waypoints across 80 steps (10Hz: 0.8s to 8.0s)
WAYPOINT_INDICES = (7, 15, 23, 31, 39, 47, 55, 63, 71, 79)
N_WAYPOINTS = len(WAYPOINT_INDICES)
TRAJ_PT_DIM = 7  # x, y, dx, dy, speed, heading_cos, heading_sin


class MotionPredictorV14(MotionPredictorV13):
    def __init__(self, hidden: int = 256, modes: int = 6, nhead: int = 4):
        super().__init__(hidden=hidden, modes=modes, nhead=nhead)

        # 1. Trajectory Waypoint Feature Extractor
        self.traj_pt_mlp = nn.Sequential(
            nn.Linear(TRAJ_PT_DIM, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
        )
        self.traj_norm = nn.LayerNorm(hidden)

        # 2. Trajectory-Map Dense Cross-Attention Scorer
        self.score_cross_attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.score_norm = nn.LayerNorm(hidden)
        self.score_ffn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        # 3. Type-Specific Trajectory Scoring Heads (evaluating [h_modes, z_eval])
        self.mode_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )
            for _ in range(3)
        ])

    def _extract_traj_features(self, traj: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale waypoint features along the 80-step predicted path.
        traj: [B, 6, 80, 2]
        Returns: [B, 6, hidden]
        """
        b, k, t, _ = traj.shape
        wp_indices = torch.tensor(WAYPOINT_INDICES, device=traj.device, dtype=torch.long)
        # [B, 6, 10, 2]
        wp_xy = traj[:, :, wp_indices, :]

        # Consecutive step differences
        diff = traj[:, :, 1:, :] - traj[:, :, :-1, :]  # [B, 6, 79, 2]
        diff_full = torch.cat([traj[:, :, :1, :], diff], dim=2)  # [B, 6, 80, 2]
        wp_dxy = diff_full[:, :, wp_indices, :]  # [B, 6, 10, 2]

        speed = torch.linalg.vector_norm(wp_dxy, dim=-1, keepdim=True) * 10.0  # m/s
        heading = torch.atan2(wp_dxy[..., 1:2], wp_dxy[..., 0:1].clamp_min(1e-4))
        h_cos = torch.cos(heading)
        h_sin = torch.sin(heading)

        # [B, 6, 10, 7]
        pt_raw = torch.cat([wp_xy, wp_dxy, speed / 20.0, h_cos, h_sin], dim=-1)
        pt_feat = self.traj_pt_mlp(pt_raw.reshape(b * k * N_WAYPOINTS, TRAJ_PT_DIM))
        pt_feat = pt_feat.reshape(b, k, N_WAYPOINTS, self.hidden)

        # MaxPool across 10 waypoints -> [B, 6, hidden]
        traj_tok = pt_feat.max(dim=2).values
        return self.traj_norm(traj_tok)

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

        # 2. Neighbor and Polyline Encoding (VectorNet Subgraph)
        neigh_tok = self._pool_neighbors(neighbors)  # [B, 16, hidden]
        poly_tok = self._encode_polylines(map_feat, map_valid)  # [B, 16, hidden]

        # 3. Target-Centric Cross-Attention
        mem = torch.cat([neigh_tok, poly_tok], dim=1)  # [B, 32, hidden]
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

        # 6. Mode Query Cross-Attention Generator
        queries = self.mode_queries.expand(b, -1, -1)  # [B, 6, hidden]
        q_self, _ = self.mode_self_attn(queries, queries, queries)
        queries = self.mode_self_norm(queries + q_self)

        scene_mem = torch.cat([scene_feat.unsqueeze(1), target.unsqueeze(1), neigh_tok, poly_tok], dim=1)  # [B, 34, hidden]
        scene_mem_pad = torch.cat([zeros_b.unsqueeze(1), zeros_b.unsqueeze(1), ~neighbor_valid, poly_pad], dim=1)  # [B, 34]
        if scene_mem_pad.all(1).any():
            scene_mem_pad = scene_mem_pad.clone()
            scene_mem_pad[scene_mem_pad.all(1), 0] = False

        q_cross, _ = self.mode_cross_attn(queries, scene_mem, scene_mem, key_padding_mask=scene_mem_pad)
        h_modes = self.mode_cross_norm(queries + q_cross)  # [B, 6, hidden]
        h_modes = h_modes + self.mode_ffn(h_modes)         # [B, 6, hidden]

        # 7. Rollout 6 Mode Trajectories
        goals = self._type_mode_output(self.goal_heads, h_modes, type_idx).view(b, self.modes, 2)
        deltas = self._type_mode_output(self.delta_heads, h_modes, type_idx).view(b, self.modes, 80, 2)
        traj = torch.cumsum(deltas, dim=2)  # [B, 6, 80, 2]

        flat = traj.reshape(b * self.modes, 80, 2)
        ref, _ = self.refine_gru(flat)
        traj = traj + self.refine_out(ref).view(b, self.modes, 80, 2)  # [B, 6, 80, 2]

        # 8. Method 1: Trajectory-Aware Dense Cross-Attention Scorer
        # CRITICAL: detach traj so ranking loss gradient does NOT flow back
        # into the trajectory generator (delta_heads, refine_gru).
        # Without detach, loss_rank pulls trajectory coordinates in conflicting
        # directions vs loss_reg, causing loss explosion and ADE regression.
        traj_feat = self._extract_traj_features(traj.detach())  # [B, 6, hidden]
        traj_query = traj_feat + h_modes                        # [B, 6, hidden]

        # Cross-attend the physical path with road polylines & scene elements
        traj_cross, _ = self.score_cross_attn(traj_query, scene_mem, scene_mem, key_padding_mask=scene_mem_pad)
        z_eval = self.score_norm(traj_query + traj_cross)
        z_eval = z_eval + self.score_ffn(z_eval)  # [B, 6, hidden]

        # Final Trajectory-Verified Mode Scoring: cat([Prior h_modes, Physical Match z_eval])
        score_input = torch.cat([h_modes, z_eval], dim=-1)  # [B, 6, hidden * 2]
        logits = self._type_mode_output(self.mode_heads, score_input, type_idx).squeeze(-1)  # [B, 6]

        return traj, goals, logits
