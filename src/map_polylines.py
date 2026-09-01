"""Group flat roadgraph samples into fixed-length lane polylines by roadgraph id."""
from __future__ import annotations

import numpy as np
import torch

POLYLINE_POINTS = 20
# Driving lanes: freeway=1, surface=2. Bike=3 is not a vehicle lane.
LANE_TYPES = (1, 2)
EDGE_TYPES = (15, 16)
STOP_TYPES = (17,)
XWALK_TYPES = (18,)  # 19 is speed bump, not crosswalk
KEEP_TYPES = LANE_TYPES + EDGE_TYPES + STOP_TYPES + XWALK_TYPES
# Same-lane stitch vs adjacent-lane split (meters of lateral offset).
LANE_MERGE_LAT_M = 1.4
LANE_ADJ_LAT_MIN_M = 2.5
LANE_ADJ_LAT_MAX_M = 6.5


def order_polyline_points(xy: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort shuffled samples of one (id, type) along the centerline.

    Prefer mean sample direction; fall back to PCA if dir is degenerate.
    """
    if xy.shape[0] <= 2:
        return xy, direction
    axis = np.mean(np.asarray(direction, dtype=np.float64), axis=0)
    nrm = float(np.linalg.norm(axis))
    if nrm < 1e-6:
        centered = np.asarray(xy, dtype=np.float64) - np.mean(xy, axis=0)
        cov = centered.T @ centered
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        nrm = float(np.linalg.norm(axis))
        if nrm < 1e-6:
            return xy, direction
    axis = axis / nrm
    order = np.argsort(xy[:, 0] * axis[0] + xy[:, 1] * axis[1])
    return xy[order], direction[order]


def resample_polyline(xy: np.ndarray, direction: np.ndarray, n_out: int = POLYLINE_POINTS):
    if xy.shape[0] == 1:
        xy_out = np.repeat(xy, n_out, axis=0)
        dir_out = np.repeat(direction, n_out, axis=0)
        valid = np.zeros((n_out,), dtype=bool)
        valid[0] = True
        return xy_out.astype(np.float32), dir_out.astype(np.float32), valid
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-3:
        xy_out = np.repeat(xy[:1], n_out, axis=0)
        dir_out = np.repeat(direction[:1], n_out, axis=0)
        valid = np.zeros((n_out,), dtype=bool)
        valid[0] = True
        return xy_out.astype(np.float32), dir_out.astype(np.float32), valid
    t = np.linspace(0.0, total, n_out)
    x = np.interp(t, s, xy[:, 0])
    y = np.interp(t, s, xy[:, 1])
    dx = np.interp(t, s, direction[:, 0])
    dy = np.interp(t, s, direction[:, 1])
    nrm = np.sqrt(dx * dx + dy * dy) + 1e-6
    xy_out = np.stack([x, y], axis=-1).astype(np.float32)
    dir_out = np.stack([dx / nrm, dy / nrm], axis=-1).astype(np.float32)
    return xy_out, dir_out, np.ones((n_out,), dtype=bool)


def build_map_polylines(
    xyz: np.ndarray,
    direction: np.ndarray,
    types: np.ndarray,
    valid: np.ndarray,
    ids: np.ndarray,
    n_points: int = POLYLINE_POINTS,
    keep_types: tuple[int, ...] = KEEP_TYPES,
) -> dict[str, torch.Tensor]:
    """Pack variable-length roadgraph polylines into [M, n_points, ...] tensors."""
    empty = {
        "map_polyline_xy": torch.zeros(0, n_points, 2, dtype=torch.float32),
        "map_polyline_dir": torch.zeros(0, n_points, 2, dtype=torch.float32),
        "map_polyline_valid": torch.zeros(0, n_points, dtype=torch.bool),
        "map_polyline_type": torch.zeros(0, dtype=torch.int16),
        "map_polyline_id": torch.zeros(0, dtype=torch.int32),
    }
    if xyz.size == 0 or ids.size == 0:
        return empty
    mask = valid.astype(bool) & np.isin(types, keep_types)
    if not np.any(mask):
        return empty
    xy = np.asarray(xyz[mask, :2], dtype=np.float32)
    direc = np.asarray(direction[mask, :2], dtype=np.float32)
    typ = np.asarray(types[mask], dtype=np.int32)
    pid = np.asarray(ids[mask], dtype=np.int64)
    groups: dict[tuple[int, int], list[int]] = {}
    for i, polyline_id in enumerate(pid.tolist()):
        groups.setdefault((int(polyline_id), int(typ[i])), []).append(i)
    xs, ds, vs, ts, pids = [], [], [], [], []
    for (polyline_id, ptype), idxs in groups.items():
        if len(idxs) < 1:
            continue
        idx = np.asarray(idxs, dtype=np.int64)
        ordered_xy, ordered_dir = order_polyline_points(xy[idx], direc[idx])
        pts, dirs, pvalid = resample_polyline(ordered_xy, ordered_dir, n_out=n_points)
        xs.append(pts)
        ds.append(dirs)
        vs.append(pvalid)
        ts.append(int(ptype))
        pids.append(polyline_id)
    if not xs:
        return empty
    return {
        "map_polyline_xy": torch.from_numpy(np.stack(xs, axis=0)),
        "map_polyline_dir": torch.from_numpy(np.stack(ds, axis=0)),
        "map_polyline_valid": torch.from_numpy(np.stack(vs, axis=0)),
        "map_polyline_type": torch.tensor(ts, dtype=torch.int16),
        "map_polyline_id": torch.tensor(pids, dtype=torch.int32),
    }
