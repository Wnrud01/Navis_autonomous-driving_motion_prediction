"""Residual ranker: s_k = V11 mode_head logits_k + r_theta(z_k).

Decoder and mode_head stay frozen. Only the residual is trained.
GT is never an input. Trajectories are not flattened.
"""
from __future__ import annotations

import torch
from torch import nn

AGENT_DIM = 16
LANE_DIM = 8
INTER_DIM = 16
TRAJ_SUM_DIM = 6  # end xy, 4s xy, mean yaw, lateral bow
T_4S = 39  # 10 Hz, 0-indexed step for 4.0s on an 80-step future


def _tok(x, dim, batch, ref):
    if x is None:
        return ref.new_zeros(batch, dim)
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    if x.shape[-1] == dim:
        return x
    out = ref.new_zeros(batch, dim)
    n = min(dim, x.shape[-1])
    out[:, :n] = x[:, :n]
    return out


def traj_summary(traj: torch.Tensor) -> torch.Tensor:
    """Per-mode [end_xy, t4_xy, mean_yaw, mean |cross-track to chord|]."""
    b, k, t, _ = traj.shape
    end = traj[:, :, -1, :]
    t4 = traj[:, :, min(T_4S, t - 1), :]
    dxy = traj[:, :, 1:, :] - traj[:, :, :-1, :]
    mean_yaw = torch.atan2(dxy[..., 1], dxy[..., 0]).mean(dim=2, keepdim=True)
    chord = torch.linalg.vector_norm(end, dim=-1, keepdim=True).clamp_min(1e-3)
    cross = (traj[..., 0] * end[..., 1:2] - traj[..., 1] * end[..., 0:1]) / chord
    lat = cross.abs().mean(dim=2, keepdim=True)
    return torch.cat([end, t4, mean_yaw, lat], dim=-1)


class HypRanker(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.traj_enc = nn.Sequential(
            nn.Linear(TRAJ_SUM_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.ctx_enc = nn.Sequential(
            nn.Linear(AGENT_DIM + LANE_DIM + INTER_DIM, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, traj, agent_tok=None, type_idx=None, lane_tok=None, inter_tok=None, prior_logits=None):
        del type_idx  # type is already inside agent_tok
        b, k, _, _ = traj.shape
        te = self.traj_enc(traj_summary(traj))
        ctx = self.ctx_enc(torch.cat([
            _tok(agent_tok, AGENT_DIM, b, traj),
            _tok(lane_tok, LANE_DIM, b, traj),
            _tok(inter_tok, INTER_DIM, b, traj),
        ], dim=-1))
        ctx = ctx.unsqueeze(1).expand(b, k, -1)
        residual = self.residual(torch.cat([te, ctx], dim=-1)).squeeze(-1)
        if prior_logits is None:
            prior_logits = residual.new_zeros(b, k)
        else:
            prior_logits = prior_logits.to(dtype=residual.dtype)
        return prior_logits + residual
