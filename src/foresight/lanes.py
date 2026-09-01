"""Nearest-lane lookup and centerline following on packed polylines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.foresight.geom import DT, FUTURE_STEPS, wrap_angle

LANE_TYPES = (1, 2, 3)


@dataclass
class Polyline:
    rid: int
    type: int
    xy: np.ndarray
    heading: np.ndarray


def polylines_from_pack(static: dict) -> list[Polyline]:
    if "map_polyline_xy" not in static or static["map_polyline_xy"].numel() == 0:
        return []
    xy = static["map_polyline_xy"].numpy()
    direc = static["map_polyline_dir"].numpy()
    valid = static["map_polyline_valid"].numpy().astype(bool)
    types = static["map_polyline_type"].numpy().astype(np.int32)
    ids = static["map_polyline_id"].numpy().astype(np.int64)
    out: list[Polyline] = []
    for i in range(xy.shape[0]):
        if int(types[i]) not in LANE_TYPES:
            continue
        m = valid[i]
        if int(m.sum()) < 2:
            continue
        pts = xy[i][m].astype(np.float32)
        dxy = direc[i][m].astype(np.float32)
        heading = np.arctan2(dxy[:, 1], dxy[:, 0]).astype(np.float32)
        out.append(Polyline(rid=int(ids[i]), type=int(types[i]), xy=pts, heading=heading))
    return out


def nearest_lane(lanes: list[Polyline], x: float, y: float, yaw: float):
    best = None
    best_d = 1e9
    p = np.array([x, y], dtype=np.float32)
    for lane in lanes:
        dxy = lane.xy - p
        dist = np.sqrt((dxy * dxy).sum(-1))
        i = int(dist.argmin())
        d = float(dist[i])
        heading = float(lane.heading[i])
        yaw_err = abs(float(wrap_angle(heading - yaw)))
        score = d + 0.8 * yaw_err
        if score < best_d:
            best_d = score
            best = dict(lane=lane, idx=i, d=d, heading=heading, yaw_err=float(wrap_angle(heading - yaw)))
    return best


def parallel_lanes(lanes: list[Polyline], hit: dict, x: float, y: float, yaw: float, max_n: int = 2):
    if hit is None:
        return []
    base = hit["lane"]
    hx, hy = float(np.cos(hit["heading"])), float(np.sin(hit["heading"]))
    lx, ly = -hy, hx
    p = np.array([x, y], dtype=np.float32)
    scored = []
    for lane in lanes:
        if lane.rid == base.rid:
            continue
        dxy = lane.xy - p
        dist = np.sqrt((dxy * dxy).sum(-1))
        i = int(dist.argmin())
        heading = float(lane.heading[i])
        if abs(float(wrap_angle(heading - yaw))) > 0.7:
            continue
        lat = float((lane.xy[i, 0] - x) * lx + (lane.xy[i, 1] - y) * ly)
        if abs(lat) < 1.5 or abs(lat) > 6.5:
            continue
        scored.append((abs(lat), i, lane))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [s[2] for s in scored[:max_n]]


def follow_lane(lane: Polyline, speed: float, start_xy: np.ndarray, steps: int = FUTURE_STEPS, dt: float = DT) -> np.ndarray:
    xy = lane.xy
    dxy = xy - start_xy
    i0 = int(np.sqrt((dxy * dxy).sum(-1)).argmin())
    seg = xy[1:] - xy[:-1]
    slen = np.sqrt((seg * seg).sum(-1))
    slen = np.concatenate([[0], slen])
    s = np.cumsum(slen)
    s = s - s[i0]
    path = np.zeros((steps, 2), dtype=np.float32)
    for t in range(steps):
        target = speed * (t + 1) * dt
        j = int(np.searchsorted(s, target))
        if j <= 0:
            path[t] = xy[0]
        elif j >= len(xy):
            extra = target - float(s[-1])
            h = float(lane.heading[-1])
            path[t] = xy[-1] + extra * np.array([np.cos(h), np.sin(h)], dtype=np.float32)
        else:
            s0, s1 = float(s[j - 1]), float(s[j])
            a = 0.0 if s1 <= s0 else (target - s0) / (s1 - s0)
            path[t] = (1 - a) * xy[j - 1] + a * xy[j]
    return path
