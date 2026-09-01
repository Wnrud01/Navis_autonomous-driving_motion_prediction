"""Six frozen motion hypotheses. Trajectories are not learned here."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.foresight.geom import history_kinematics, rollout_ctra, rollout_cv
from src.foresight.lanes import Polyline, follow_lane, nearest_lane, parallel_lanes

MODE_KINDS = ("cv", "ctra", "brake", "lane", "lane-alt", "yield")
MODE_LABELS = ("CV", "CTRA", "감속", "차로추종", "옆차로", "양보")


@dataclass
class Mode:
    kind: str
    label: str
    path: np.ndarray


def predict_agent(
    hist: np.ndarray,
    hist_valid: np.ndarray,
    type_id: int,
    lanes: list[Polyline],
) -> list[Mode]:
    k = history_kinematics(hist, hist_valid)
    is_ped = int(type_id) == 2
    yaw, speed = k["yaw"], max(0.0, k["speed"])
    hit = nearest_lane(lanes, k["x"], k["y"], yaw)
    alts = parallel_lanes(lanes, hit, k["x"], k["y"], yaw) if hit else []

    cv = rollout_cv(k["x"], k["y"], k["vx"], k["vy"])
    ctra = rollout_ctra(k["x"], k["y"], yaw, speed, k["accel"], k["yaw_rate"])
    brake_acc = -1.2 if is_ped else -2.4
    brake = rollout_ctra(k["x"], k["y"], yaw, speed, brake_acc, k["yaw_rate"] * 0.4)

    def _sane(path: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        if path.shape != fallback.shape or not np.isfinite(path).all():
            return fallback
        end = path[-1]
        dist = float(np.hypot(end[0] - k["x"], end[1] - k["y"]))
        cap = max(speed, 0.4) * 8.0 * 1.8 + 20.0
        return fallback if dist > cap else path

    if is_ped:
        lane_path = rollout_ctra(k["x"], k["y"], yaw + 0.35, speed * 0.9, 0.0, 0.0)
        alt_path = rollout_ctra(k["x"], k["y"], yaw - 0.40, speed * 0.7, -0.3, -0.15)
    else:
        use_lane = hit is not None and hit["d"] < 3.5 and abs(hit["yaw_err"]) < 0.5
        start = np.array([k["x"], k["y"]], dtype=np.float32)
        lane_path = follow_lane(hit["lane"], max(speed, 0.4), start_xy=start) if use_lane else ctra
        lane_path = _sane(lane_path, ctra)
        alt_path = (
            follow_lane(alts[0], max(speed, 0.4), start_xy=start)
            if alts
            else rollout_ctra(k["x"], k["y"], yaw - 0.28, speed * 0.85, 0.0, k["yaw_rate"])
        )
        alt_path = _sane(alt_path, rollout_ctra(k["x"], k["y"], yaw - 0.28, speed * 0.85, 0.0, k["yaw_rate"]))

    yield_acc = -2.2 if is_ped else -1.4
    yield_path = rollout_ctra(k["x"], k["y"], yaw, speed, yield_acc, 0.0)

    paths = (cv, ctra, brake, lane_path, alt_path, yield_path)
    return [Mode(kind=kind, label=lab, path=p) for kind, lab, p in zip(MODE_KINDS, MODE_LABELS, paths)]
