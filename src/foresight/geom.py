"""Kinematics, CV/CTRA rollouts, ADE/FDE. Ported from the Foresight README package."""
from __future__ import annotations

import numpy as np

FUTURE_STEPS = 80
DT = 0.1
EPS = 1e-6


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def history_kinematics(hist: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """hist: (T, 5) x,y,yaw,vx,vy. valid: (T,). Last valid row is current."""
    seq = hist[valid > 0.5]
    if len(seq) == 0:
        return dict(x=0, y=0, yaw=0, vx=0, vy=0, speed=0, accel=0, yaw_rate=0)
    last = seq[-1]
    speed = float(np.hypot(last[3], last[4]))
    yaw = float(last[2])
    accel = 0.0
    yaw_rate = 0.0
    if len(seq) >= 2:
        speeds = np.hypot(seq[:, 3], seq[:, 4])
        dspeed = np.diff(speeds) / DT
        dyaw = wrap_angle(np.diff(seq[:, 2])) / DT
        accel = float(dspeed.mean())
        yaw_rate = float(np.asarray(dyaw).mean())
    return dict(
        x=float(last[0]),
        y=float(last[1]),
        yaw=yaw,
        vx=float(last[3]),
        vy=float(last[4]),
        speed=speed,
        accel=accel,
        yaw_rate=yaw_rate,
    )


def rollout_cv(x: float, y: float, vx: float, vy: float, steps: int = FUTURE_STEPS, dt: float = DT) -> np.ndarray:
    t = np.arange(1, steps + 1, dtype=np.float32) * dt
    return np.stack([x + vx * t, y + vy * t], axis=1).astype(np.float32)


def rollout_ctra(
    x: float,
    y: float,
    yaw: float,
    v: float,
    accel: float,
    yaw_rate: float,
    steps: int = FUTURE_STEPS,
    dt: float = DT,
) -> np.ndarray:
    path = np.zeros((steps, 2), dtype=np.float32)
    px, py, th, vel = x, y, yaw, max(0.0, v)
    w = yaw_rate
    for i in range(steps):
        vel = max(0.0, vel + accel * dt)
        if abs(w) < 1e-4:
            px += vel * np.cos(th) * dt
            py += vel * np.sin(th) * dt
        else:
            dth = w * dt
            px += vel / w * (np.sin(th + dth) - np.sin(th))
            py += -vel / w * (np.cos(th + dth) - np.cos(th))
            th = th + dth
        path[i] = (px, py)
    return path


def ade(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> float:
    if valid is not None:
        m = valid > 0.5
        if int(m.sum()) == 0:
            return float("inf")
        d = pred[m] - gt[m]
        return float(np.sqrt((d * d).sum(-1)).mean())
    n = min(len(pred), len(gt))
    if n == 0:
        return float("inf")
    d = pred[:n] - gt[:n]
    return float(np.sqrt((d * d).sum(-1)).mean())
