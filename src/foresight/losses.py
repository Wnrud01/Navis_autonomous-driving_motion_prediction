"""Soft CE / WTA ranker losses and ADE stand-ins from the Foresight README."""
from __future__ import annotations

import torch
import torch.nn.functional as F

TAU = 1.2
L2 = 0.01
ALIGN = 0.5


def target_dist(ades: torch.Tensor, kind: str = "soft-ce", tau: float = TAU) -> torch.Tensor:
    if kind == "wta":
        k_star = ades.argmin(dim=-1)
        q = torch.zeros_like(ades)
        q.scatter_(1, k_star.unsqueeze(1), 1.0)
        return q
    if kind in ("soft-ce", "soft-wta"):
        clipped = ades.clamp(max=12.0)
        return torch.softmax(-clipped / tau, dim=-1)
    raise ValueError(f"unknown loss kind {kind}")


def ranker_loss(
    logits: torch.Tensor,
    ades: torch.Tensor,
    kind: str = "soft-wta",
    tau: float = TAU,
    weights: torch.Tensor | None = None,
    l2: float = L2,
    align: float = ALIGN,
) -> dict[str, torch.Tensor]:
    p = torch.softmax(logits, dim=-1)
    q = target_dist(ades, kind=kind, tau=tau)
    ce = -(q * (p.clamp_min(1e-8).log())).sum(dim=-1).mean()
    l2_pen = 0.5 * l2 * (weights * weights).sum() if weights is not None else logits.new_zeros(())
    pred = p.argmax(dim=-1)
    k_star = ades.argmin(dim=-1)
    acc = (pred == k_star).float().mean()
    b = torch.arange(logits.size(0), device=logits.device)
    ade1 = ades[b, pred]
    ade6 = ades[b, k_star]
    gap = (ade1 - ade6).clamp_min(0.0)
    logp = p.clamp_min(1e-8).log()
    align_term = (gap.detach() * (-logp[b, k_star])).mean()
    loss = ce + l2_pen
    if kind == "soft-wta":
        loss = loss + align * align_term
    elif kind == "wta":
        loss = (-logp[b, k_star]).mean() + l2_pen
    return dict(loss=loss, ce=ce, l2=l2_pen, align=align_term, acc=acc, gap=gap.mean(), p=p, q=q)


def batched_ade(pred_xy: torch.Tensor, gt_xy: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
    squeeze = pred_xy.dim() == 3
    if squeeze:
        pred_xy = pred_xy.unsqueeze(1)
    dist = (pred_xy - gt_xy[:, None, :, :]).square().sum(-1).clamp_min(1e-9).sqrt()
    dist = dist * gt_valid[:, None, :]
    denom = gt_valid.sum(-1).clamp_min(1.0)[:, None]
    ade = dist.sum(-1) / denom
    return ade.squeeze(1) if squeeze else ade


def metric_loss(ades: torch.Tensor, logits: torch.Tensor, tau: float = TAU) -> dict[str, torch.Tensor]:
    p = torch.softmax(logits, dim=-1)
    q_softmin = torch.softmax(-ades / tau, dim=-1)
    ade1 = (p * ades).sum(-1).mean()
    ade6 = (q_softmin * ades).sum(-1).mean()
    return dict(ade1=ade1, ade6=ade6, loss=ade1 + ade6)
