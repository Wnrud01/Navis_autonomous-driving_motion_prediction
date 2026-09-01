"""Linear mode scorer. z_k = h_k + w·φ_k + b. φ is 32-D."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.foresight.geom import history_kinematics, wrap_angle
from src.foresight.lanes import Polyline, nearest_lane, parallel_lanes
from src.foresight.mixture import MODE_KINDS, Mode

FEAT_DIM = 32
KIND_INDEX = {k: i for i, k in enumerate(MODE_KINDS)}
N_MODES = 6


def heuristic_logit(type_id: int, mode: Mode, ctx: dict) -> float:
    is_ped = int(type_id) == 2
    speed = ctx["speed"]
    table_ped = {"cv": 1.2, "ctra": 0.95, "brake": 0.4, "lane": -0.4, "lane-alt": 0.15, "yield": 0.55}
    table_veh = {"cv": 0.85, "ctra": 1.45, "brake": 0.1, "lane": 0.55, "lane-alt": 0.25, "yield": 0.05}
    z = table_ped[mode.kind] if is_ped else table_veh[mode.kind]
    if not is_ped:
        if ctx["hit_dist"] < 1.8 and abs(ctx["hit_yaw_err"]) < 0.25 and mode.kind == "lane":
            z += 0.9
        if ctx["hit_dist"] > 3.5 and mode.kind in ("lane", "lane-alt"):
            z -= 1.4
    if speed < 0.45 and mode.kind in ("brake", "yield", "cv"):
        z += 0.8
    if speed > 6 and mode.kind == "ctra":
        z += 0.2
    if abs(ctx["yaw_rate"]) > 0.15 and mode.kind == "ctra":
        z += 0.4
    if ctx.get("to_predict", True) and mode.kind == "ctra":
        z += 0.1
    return float(z)


def _clip01(v: float, scale: float, cap: float = 2.0) -> float:
    return float(np.clip(v / scale, -cap, cap))


def _path_shape(path: np.ndarray, x: float, y: float, yaw: float, nb_xy: np.ndarray, hit) -> tuple[float, ...]:
    end = path[-1]
    dx, dy = float(end[0] - x), float(end[1] - y)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    fwd = dx * c + dy * s
    lat = -dx * s + dy * c
    seg = path[1:] - path[:-1]
    arc = float(np.sqrt((seg * seg).sum(-1)).sum())
    head = np.arctan2(seg[:, 1], seg[:, 0])
    turn = float(np.abs(wrap_angle(np.diff(head))).mean()) if len(head) > 1 else 0.0
    nb_min = 8.0
    if nb_xy is not None and len(nb_xy):
        sl = path[::10]
        d = sl[:, None, :] - nb_xy[None, :, :]
        nb_min = float(np.sqrt((d * d).sum(-1)).min())
    lane_dev = 8.0
    if hit is not None:
        lane = hit["lane"]
        dxy = lane.xy - end
        lane_dev = float(np.sqrt((dxy * dxy).sum(-1)).min())
    return fwd, lat, arc, turn, nb_min, lane_dev


def mode_features(type_id: int, length: float, width: float, to_predict: bool, mode: Mode, ctx: dict) -> np.ndarray:
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    f[0] = 1.0 if int(type_id) == 1 else 0.0
    f[1] = 1.0 if int(type_id) == 2 else 0.0
    f[2] = 1.0 if int(type_id) == 3 else 0.0
    f[3] = 1.0 if to_predict else 0.0
    f[4] = _clip01(ctx["speed"], 15.0)
    f[5] = _clip01(ctx["accel"], 3.0)
    f[6] = _clip01(ctx["yaw_rate"], 0.5)
    f[7] = _clip01(length, 8.0)
    f[8] = _clip01(width, 3.0)
    f[9] = _clip01(ctx["hist_disp"], 20.0)
    f[10] = _clip01(abs(ctx["vel_yaw_err"]), 1.0)
    f[11] = _clip01(ctx["hit_dist"], 8.0)
    f[12] = _clip01(abs(ctx["hit_yaw_err"]), 1.0)
    f[13] = _clip01(ctx["lane_curv"], 0.2)
    f[14] = _clip01(ctx["n_alt"], 3.0)
    f[15] = _clip01(ctx["neighbors"], 10.0)
    f[16] = _clip01(ctx["nb_dist"], 20.0)
    f[17] = _clip01(ctx["nb_rel_speed"], 10.0)
    f[18] = _clip01(ctx["ego_dist"], 50.0)
    f[19] = _clip01(ctx["ego_bearing"], np.pi)
    f[20 + KIND_INDEX[mode.kind]] = 1.0
    fwd, lat, arc, turn, nb_min, lane_dev = _path_shape(
        mode.path, ctx["x"], ctx["y"], ctx["yaw"], ctx.get("nb_xy"), ctx.get("hit")
    )
    f[26] = _clip01(fwd, 40.0)
    f[27] = _clip01(lat, 10.0)
    f[28] = _clip01(arc, 40.0)
    f[29] = _clip01(turn, 0.3)
    f[30] = _clip01(nb_min, 20.0)
    f[31] = _clip01(lane_dev, 8.0)
    return f


def score_context(
    hist: np.ndarray,
    hist_valid: np.ndarray,
    lanes: list[Polyline],
    nb_xy: np.ndarray,
    nb_speed: np.ndarray,
    ego_xy: np.ndarray | None,
    ego_yaw: float,
) -> dict:
    k = history_kinematics(hist, hist_valid)
    hit = nearest_lane(lanes, k["x"], k["y"], k["yaw"])
    alts = parallel_lanes(lanes, hit, k["x"], k["y"], k["yaw"]) if hit else []
    seq = hist[hist_valid > 0.5]
    if len(seq) >= 2:
        hist_disp = float(np.hypot(seq[-1, 0] - seq[0, 0], seq[-1, 1] - seq[0, 1]))
    else:
        hist_disp = 0.0
    vel_yaw = float(np.arctan2(k["vy"], k["vx"])) if k["speed"] > 0.2 else k["yaw"]
    vel_yaw_err = float(wrap_angle(vel_yaw - k["yaw"]))

    lane_curv = 0.0
    if hit is not None:
        lane = hit["lane"]
        i = hit["idx"]
        j = min(len(lane.heading) - 1, i + 5)
        if j > i:
            lane_curv = float(wrap_angle(lane.heading[j] - lane.heading[i]) / max(j - i, 1))

    nb_dist, nb_rel_speed = 30.0, 0.0
    if nb_xy is not None and len(nb_xy):
        d = np.hypot(nb_xy[:, 0] - k["x"], nb_xy[:, 1] - k["y"])
        j = int(d.argmin())
        nb_dist = float(d[j])
        nb_rel_speed = float(nb_speed[j] - k["speed"])

    ego_dist, ego_bearing = 50.0, 0.0
    if ego_xy is not None:
        dx, dy = k["x"] - float(ego_xy[0]), k["y"] - float(ego_xy[1])
        ego_dist = float(np.hypot(dx, dy))
        ego_bearing = float(wrap_angle(np.arctan2(dy, dx) - ego_yaw))

    return dict(
        x=k["x"], y=k["y"], yaw=k["yaw"], speed=k["speed"], accel=k["accel"], yaw_rate=k["yaw_rate"],
        hist_disp=hist_disp, vel_yaw_err=vel_yaw_err, hit=hit,
        hit_dist=hit["d"] if hit else 8.0,
        hit_yaw_err=hit["yaw_err"] if hit else 0.0,
        lane_curv=lane_curv, n_alt=len(alts),
        neighbors=int(len(nb_xy)) if nb_xy is not None else 0,
        nb_xy=nb_xy if nb_xy is not None else np.zeros((0, 2), dtype=np.float32),
        nb_dist=nb_dist, nb_rel_speed=nb_rel_speed,
        ego_dist=ego_dist, ego_bearing=ego_bearing,
        to_predict=True,
    )


class LinearScorer(nn.Module):
    def __init__(self, dim: int = FEAT_DIM):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor, heuristics: torch.Tensor) -> torch.Tensor:
        return heuristics + features @ self.w + self.b
