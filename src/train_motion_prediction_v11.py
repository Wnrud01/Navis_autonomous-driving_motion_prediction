#!/usr/bin/env python3
"""V11: same mixed tokens as V10, decoder does not warp 80 steps onto the 8s goal.

ADE is the mean over valid timesteps. Forcing traj[-1]=goal pulls 1–6s off the
true path and raises minADE6. V11 keeps cumsum+refine only; goal is an aux head.
"""
from __future__ import annotations

import torch
from torch import nn

from src.train_motion_prediction_v10 import MotionPredictorV10


class MotionPredictorV11(MotionPredictorV10):
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
        traj = torch.cumsum(deltas, dim=2)
        flat = traj.reshape(b * self.modes, 80, 2)
        ref, _ = self.refine_gru(flat)
        traj = traj + self.refine_out(ref).view(b, self.modes, 80, 2)
        return traj, goals, self.mode_head(h)
