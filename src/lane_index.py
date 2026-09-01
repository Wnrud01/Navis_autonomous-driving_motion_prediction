"""Reconstruct N-lane index (left=1) from type {1,2} centerlines at an agent pose.

Merge / split (meters of lateral offset in the agent frame):
  |Δlat| < ~1.4 m     → same-lane fragments, merge at cluster time
  ~1.4–2.5 m          → still same-lane leftover; skip, do not break
  ~2.5–6.5 m          → adjacent lane, count separately in n_lanes
  > ~6.5 m            → stop growing that side
3.2–3.8 m is typical lane width, NOT the merge threshold.
"""
from __future__ import annotations

import numpy as np

from src.map_polylines import LANE_ADJ_LAT_MAX_M, LANE_ADJ_LAT_MIN_M, LANE_MERGE_LAT_M, LANE_TYPES

HEADING_MAX_RAD = 0.5


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def lane_index_from_polylines(
    poly_xy: np.ndarray,
    poly_dir: np.ndarray,
    poly_valid: np.ndarray,
    poly_type: np.ndarray,
    x: float,
    y: float,
    yaw: float,
    merge_lat: float = LANE_MERGE_LAT_M,
    adj_min: float = LANE_ADJ_LAT_MIN_M,
    adj_max: float = LANE_ADJ_LAT_MAX_M,
    heading_max: float = HEADING_MAX_RAD,
) -> dict:
    """Return lane_idx (1-based from left), n_lanes, lateral offset, has_left/right."""
    empty = dict(lane_idx=0, n_lanes=0, lat=0.0, yaw_err=0.0, has_left=False, has_right=False, valid=False)
    if poly_xy.size == 0:
        return empty
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    p = np.array([x, y], dtype=np.float32)
    cands: list[tuple[float, float, float]] = []
    for i in range(poly_xy.shape[0]):
        if int(poly_type[i]) not in LANE_TYPES:
            continue
        m = poly_valid[i]
        if int(m.sum()) < 2:
            continue
        pts = poly_xy[i][m]
        dxy = pts - p
        dist = np.sqrt((dxy * dxy).sum(-1))
        j = int(dist.argmin())
        dx, dy = float(dxy[j, 0]), float(dxy[j, 1])
        lat = -s * dx + c * dy
        heading = float(np.arctan2(poly_dir[i][m][j, 1], poly_dir[i][m][j, 0]))
        yaw_err = float(abs(wrap_angle(heading - yaw)))
        if yaw_err > heading_max or float(dist[j]) > 40.0:
            continue
        cands.append((lat, yaw_err, float(dist[j])))
    if not cands:
        return empty
    cands.sort(key=lambda t: t[0])
    clusters: list[list[tuple[float, float, float]]] = []
    for item in cands:
        if clusters and abs(item[0] - float(np.mean([c[0] for c in clusters[-1]]))) < merge_lat:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    means: list[tuple[float, float]] = []
    for cl in clusters:
        means.append((
            float(np.mean([c[0] for c in cl])),
            float(np.mean([c[1] for c in cl])),
        ))
    seed = int(np.argmin([abs(m[0]) for m in means]))
    taken_right = [means[seed]]
    right = seed
    for i in range(seed + 1, len(means)):
        gap = means[i][0] - means[right][0]
        if gap < adj_min:
            continue
        if gap <= adj_max:
            taken_right.append(means[i])
            right = i
        else:
            break
    taken_left: list[tuple[float, float]] = []
    left = seed
    for i in range(seed - 1, -1, -1):
        gap = means[left][0] - means[i][0]
        if gap < adj_min:
            continue
        if gap <= adj_max:
            taken_left.append(means[i])
            left = i
        else:
            break
    lanes = list(reversed(taken_left)) + taken_right
    if not lanes:
        return empty
    lats = np.array([ln[0] for ln in lanes], dtype=np.float32)
    k = int(np.argmin(np.abs(lats)))
    return dict(
        lane_idx=k + 1,
        n_lanes=len(lanes),
        lat=float(lats[k]),
        yaw_err=float(lanes[k][1]),
        has_left=k > 0,
        has_right=k < len(lanes) - 1,
        valid=True,
    )
