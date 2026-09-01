#!/usr/bin/env python3
"""Motion Prediction V17: X-Large 58M Architecture with Polyline & Social Graph Attention.

Key Features:
1. X-Large Representation Capacity: hidden = 768, nhead = 12, ffn_dim = 1536 (~58M parameters).
2. Polyline Graph Attention (Lane-to-Lane Self-Attention): Models road topology (merges, splits, intersections).
3. Social Graph Attention (Agent-to-Agent Self-Attention): Models dynamic traffic flow and vehicle interactions.
4. Two-Stage Proposal-to-Refinement Decoder with Deep Supervision.
5. Dropout 0.1 for regularization across all attention and MLP layers.
"""
from __future__ import annotations

import torch
from torch import nn

POLYLINE_POINTS = 20
POLY_K = 16
POLY_FEAT_DIM = 8
MODES = 6
WAYPOINT_INDICES = (7, 15, 23, 31, 39, 47, 55, 63, 71, 79)
N_WAYPOINTS = len(WAYPOINT_INDICES)
TRAJ_PT_DIM = 7  # x, y, dx, dy, speed, cos_heading, sin_heading


class MotionPredictorV17(nn.Module):
    def __init__(self, hidden: int = 768, modes: int = 6, nhead: int = 12, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.modes = modes
        self.nhead = nhead

        # ---------------------------------------------------------------------
        # 1. Target History Encoder
        # ---------------------------------------------------------------------
        self.target_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(55, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.hist_gru = nn.GRU(input_size=5, hidden_size=hidden, batch_first=True)
        self.hist_skip = nn.Linear(hidden, hidden)

        # ---------------------------------------------------------------------
        # 2. Neighbor Encoder & Social Graph Self-Attention
        # ---------------------------------------------------------------------
        self.neigh_mlp = nn.Sequential(
            nn.Linear(14, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.social_graph_attn = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.social_norm = nn.LayerNorm(hidden)

        # ---------------------------------------------------------------------
        # 3. Polyline VectorNet Subgraph Encoder & Polyline Graph Self-Attention
        # ---------------------------------------------------------------------
        self.poly_pt_mlp = nn.Sequential(
            nn.Linear(POLY_FEAT_DIM, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
        )
        self.poly_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.poly_norm = nn.LayerNorm(hidden)
        self.poly_graph_attn = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.poly_graph_norm = nn.LayerNorm(hidden)

        # ---------------------------------------------------------------------
        # 4. Target-Centric Cross-Attention with Map & Neighbors
        # ---------------------------------------------------------------------
        self.cross_attn = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)

        # ---------------------------------------------------------------------
        # 5. 6-Stage Human-like Cognitive Gating (768-dim)
        # ---------------------------------------------------------------------
        self.speed_enc = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.lane_enc = nn.Sequential(nn.Linear(8, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.lead_enc = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.sig_enc = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.road_enc = nn.Sequential(nn.Linear(10, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.adj_enc = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.LayerNorm(hidden))

        self.fuse_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
            for _ in range(6)
        ])
        self.fuse_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(6)])

        self.type_emb = nn.Embedding(3, 32)
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3 + 32, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        # ---------------------------------------------------------------------
        # 6. Stage 1: Proposal Generator (768-dim Mode Query Decoder)
        # ---------------------------------------------------------------------
        self.mode_queries = nn.Parameter(torch.randn(1, modes, hidden) * 0.02)
        self.mode_self_attn = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.mode_self_norm = nn.LayerNorm(hidden)
        self.mode_cross_attn = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.mode_cross_norm = nn.LayerNorm(hidden)
        self.mode_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )

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
        self.refine_gru = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
        self.refine_out = nn.Linear(hidden, 2)

        # ---------------------------------------------------------------------
        # 7. Stage 2: Refinement & Re-Scoring Modules (768-dim)
        # ---------------------------------------------------------------------
        self.wp_mlp = nn.Sequential(
            nn.Linear(TRAJ_PT_DIM, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
        )
        self.wp_norm = nn.LayerNorm(hidden)

        self.refine_self_attn = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.refine_self_norm = nn.LayerNorm(hidden)
        self.refine_cross_attn = nn.MultiheadAttention(hidden, nhead, dropout=dropout, batch_first=True)
        self.refine_cross_norm = nn.LayerNorm(hidden)
        self.refine_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
        )

        self.stage2_goal_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 2))
            for _ in range(3)
        ])
        self.stage2_delta_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 80 * 2))
            for _ in range(3)
        ])
        self.stage2_mode_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )
            for _ in range(3)
        ])

    def _pool_neighbors(self, neighbors: torch.Tensor) -> torch.Tensor:
        b, n, t, d = neighbors.shape
        flat = neighbors.reshape(b * n * t, d)
        feat = self.neigh_mlp(flat).reshape(b, n, t, self.hidden)
        return feat.max(dim=2).values  # [B, 16, hidden]

    def _encode_polylines(self, polylines: torch.Tensor, poly_valid: torch.Tensor) -> torch.Tensor:
        b = polylines.shape[0]
        if polylines.dim() == 3:
            pts_h = self.poly_pt_mlp(polylines)
            if pts_h.shape[1] >= POLY_K:
                pts_h = pts_h[:, :POLY_K]
            else:
                pad_k = POLY_K - pts_h.shape[1]
                pts_h = torch.cat([pts_h, pts_h.new_zeros(b, pad_k, self.hidden)], dim=1)
            return self.poly_norm(self.poly_mlp(pts_h))

        b, k, p, d = polylines.shape
        flat_pts = polylines.reshape(b * k * p, d)
        pt_feat = self.poly_pt_mlp(flat_pts).reshape(b, k, p, self.hidden)

        if d >= 5:
            pt_valid = polylines[..., 4:5]
            pt_feat = pt_feat * pt_valid

        poly_tok = pt_feat.max(dim=2).values
        poly_tok = self.poly_norm(self.poly_mlp(poly_tok))
        poly_tok = poly_tok * poly_valid.unsqueeze(-1).to(poly_tok.dtype)
        return poly_tok

    def _fuse_stage(self, idx: int, h: torch.Tensor, cue: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        gate = self.fuse_gates[idx](torch.cat([h.unsqueeze(1), cue], dim=-1))
        h_cand = self.fuse_norms[idx](h.unsqueeze(1) + gate * cue).squeeze(1)
        return torch.where(active.unsqueeze(-1), h_cand, h)

    def _type_mode_output(self, heads: nn.ModuleList, h_modes: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
        b = h_modes.shape[0]
        idx = type_idx.clamp(0, 2)
        stacked = torch.stack([head(h_modes) for head in heads], dim=1)
        gather_idx = idx[:, None, None, None].expand(b, 1, self.modes, stacked.shape[-1])
        return stacked.gather(1, gather_idx).squeeze(1)

    def _extract_waypoint_features(self, traj: torch.Tensor) -> torch.Tensor:
        b, k, t, _ = traj.shape
        wp_indices = torch.tensor(WAYPOINT_INDICES, device=traj.device, dtype=torch.long)
        wp_xy = traj[:, :, wp_indices, :]

        diff = traj[:, :, 1:, :] - traj[:, :, :-1, :]
        diff_full = torch.cat([traj[:, :, :1, :], diff], dim=2)
        wp_dxy = diff_full[:, :, wp_indices, :]

        speed = torch.linalg.vector_norm(wp_dxy, dim=-1, keepdim=True) * 10.0
        heading = torch.atan2(wp_dxy[..., 1:2], wp_dxy[..., 0:1].clamp_min(1e-4))
        h_cos = torch.cos(heading)
        h_sin = torch.sin(heading)

        pt_raw = torch.cat([wp_xy, wp_dxy, speed / 20.0, h_cos, h_sin], dim=-1)
        pt_feat = self.wp_mlp(pt_raw.reshape(b * k * N_WAYPOINTS, TRAJ_PT_DIM))
        pt_feat = pt_feat.reshape(b, k, N_WAYPOINTS, self.hidden)

        traj_tok = pt_feat.max(dim=2).values
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
        # 1. Target, Neighbor & Polyline Graph Encoders
        # ---------------------------------------------------------------------
        target = self.target_encoder(target_hist)
        gru_out, _ = self.hist_gru(target_hist)
        h = target + self.hist_skip(gru_out[:, -1])

        # Neighbor PointNet + Social Graph Self-Attention (12 Heads)
        neigh_tok = self._pool_neighbors(neighbors)  # [B, 16, hidden]
        neigh_pad = ~neighbor_valid
        if neigh_pad.all(1).any():
            neigh_pad_safe = neigh_pad.clone()
            neigh_pad_safe[neigh_pad_safe.all(1), 0] = False
        else:
            neigh_pad_safe = neigh_pad

        s_self, _ = self.social_graph_attn(neigh_tok, neigh_tok, neigh_tok, key_padding_mask=neigh_pad_safe)
        s_self = s_self * neighbor_valid.unsqueeze(-1).to(s_self.dtype)
        neigh_tok = self.social_norm(neigh_tok + s_self)

        # Polyline VectorNet + Polyline Graph Self-Attention (12 Heads)
        poly_tok = self._encode_polylines(map_feat, map_valid)  # [B, 16, hidden]
        poly_pad = ~map_valid
        if poly_pad.shape[1] != POLY_K:
            poly_pad = poly_pad[:, :POLY_K] if poly_pad.shape[1] > POLY_K else torch.cat([poly_pad, poly_pad.new_ones(b, POLY_K - poly_pad.shape[1])], dim=1)

        if poly_pad.all(1).any():
            poly_pad_safe = poly_pad.clone()
            poly_pad_safe[poly_pad_safe.all(1), 0] = False
        else:
            poly_pad_safe = poly_pad

        p_self, _ = self.poly_graph_attn(poly_tok, poly_tok, poly_tok, key_padding_mask=poly_pad_safe)
        p_self = p_self * map_valid.unsqueeze(-1).to(p_self.dtype)
        poly_tok = self.poly_graph_norm(poly_tok + p_self)

        # ---------------------------------------------------------------------
        # 2. Target-Centric Cross-Attention with Graph-Enhanced Map & Neighbors
        # ---------------------------------------------------------------------
        mem = torch.cat([neigh_tok, poly_tok], dim=1)  # [B, 32, hidden]
        pad = torch.cat([~neighbor_valid, poly_pad], dim=1)
        if pad.all(1).any():
            pad = pad.clone()
            pad[pad.all(1), 0] = False

        ctx, _ = self.cross_attn(h.unsqueeze(1), mem, mem, key_padding_mask=pad)
        ctx = ctx.squeeze(1)
        h = self.attn_norm(ctx + h)

        # ---------------------------------------------------------------------
        # 3. 6-Stage Human-like Cognitive Gating (768-dim)
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
        # 5. Stage 1: Proposal Generator (768-dim)
        # ---------------------------------------------------------------------
        queries = self.mode_queries.expand(b, -1, -1)
        q_self, _ = self.mode_self_attn(queries, queries, queries)
        queries = self.mode_self_norm(queries + q_self)

        q_cross, _ = self.mode_cross_attn(queries, scene_mem, scene_mem, key_padding_mask=scene_mem_pad)
        h_modes_1 = self.mode_cross_norm(queries + q_cross)
        h_modes_1 = h_modes_1 + self.mode_ffn(h_modes_1)

        goals_1 = self._type_mode_output(self.goal_heads, h_modes_1, type_idx).view(b, self.modes, 2)
        deltas_1 = self._type_mode_output(self.delta_heads, h_modes_1, type_idx).view(b, self.modes, 80, 2)
        traj_1 = torch.cumsum(deltas_1, dim=2)

        flat_1 = traj_1.reshape(b * self.modes, 80, 2)
        ref_1, _ = self.refine_gru(flat_1)
        traj_1 = traj_1 + self.refine_out(ref_1).view(b, self.modes, 80, 2)

        # ---------------------------------------------------------------------
        # 6. Stage 2: Local Scene-Interactive Refinement & Re-Scoring (768-dim)
        # ---------------------------------------------------------------------
        traj_feat_1 = self._extract_waypoint_features(traj_1)
        q_refine = h_modes_1 + traj_feat_1

        r_self, _ = self.refine_self_attn(q_refine, q_refine, q_refine)
        q_refine = self.refine_self_norm(q_refine + r_self)

        r_cross, _ = self.refine_cross_attn(q_refine, scene_mem, scene_mem, key_padding_mask=scene_mem_pad)
        h_modes_2 = self.refine_cross_norm(q_refine + r_cross)
        h_modes_2 = h_modes_2 + self.refine_ffn(h_modes_2)

        d_goals = self._type_mode_output(self.stage2_goal_heads, h_modes_2, type_idx).view(b, self.modes, 2)
        d_deltas = self._type_mode_output(self.stage2_delta_heads, h_modes_2, type_idx).view(b, self.modes, 80, 2)
        d_traj = torch.cumsum(d_deltas, dim=2)

        goals_2 = goals_1 + d_goals
        traj_2 = traj_1 + d_traj

        logits_2 = self._type_mode_output(self.stage2_mode_heads, h_modes_2, type_idx).squeeze(-1)

        return traj_2, goals_2, logits_2, traj_1, goals_1
