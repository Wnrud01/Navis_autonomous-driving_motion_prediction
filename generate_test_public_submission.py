"""Generate a challenge-compatible submission.npz from Motion Prediction Model V1.

This adapter consumes the organizer's test_public PKL files, runs the existing
V1 model for every manifest target agent, restores world-frame trajectories,
and writes the organizer's required (N, 6, 80, 2) NPZ schema.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.train_motion_prediction_v1 import (
    MotionPredictor,
    local_vec,
    local_xy,
    pack_neighbor_features,
    select_neighbor_indices,
    expand_neighbor_encoder_state,
    NEIGHBOR_FEAT_DIM,
)

CURRENT_STEP = 10
HISTORICAL_STEPS = 11
FUTURE_STEPS = 80
K_MODES = 6
NEIGHBOR_K = 16
SIGNAL_K = 4


def tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    out = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return out.to(dtype=dtype) if dtype is not None else out


def local_to_world(local_xy_values: torch.Tensor, origin: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert (B, K, T, 2) target-local coordinates to organizer world xy."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    x = origin[:, None, None, 0] + local_xy_values[..., 0] * c[:, None, None] - local_xy_values[..., 1] * s[:, None, None]
    y = origin[:, None, None, 1] + local_xy_values[..., 0] * s[:, None, None] + local_xy_values[..., 1] * c[:, None, None]
    return torch.stack([x, y], dim=-1)


def organizer_type_to_training_feature(agent_types: torch.Tensor) -> torch.Tensor:
    """Map organizer vehicle/pedestrian/cyclist ids 0/1/2 to V1's raw neighbor feature ids 1/2/3."""
    out = torch.zeros_like(agent_types, dtype=torch.float32)
    dynamic = (agent_types >= 0) & (agent_types <= 2)
    out[dynamic] = agent_types[dynamic].to(torch.float32) + 1.0
    return out


def load_scene(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_features(scene: dict[str, Any], target_ids: list[int]) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce V1 target-centric features using only public t<=10 state."""
    agent = scene["agent"]
    position = tensor(agent["position"], torch.float32)[:, :HISTORICAL_STEPS, :2]
    heading = tensor(agent["heading"], torch.float32)[:, :HISTORICAL_STEPS]
    velocity = tensor(agent["velocity"], torch.float32)[:, :HISTORICAL_STEPS, :2]
    valid = tensor(agent["valid_mask"], torch.bool)[:, :HISTORICAL_STEPS]
    shape = tensor(agent["shape"], torch.float32)[:, CURRENT_STEP, :3]
    raw_type = tensor(agent["type"], torch.long)

    position = torch.nan_to_num(position)
    heading = torch.nan_to_num(heading)
    velocity = torch.nan_to_num(velocity)
    shape = torch.nan_to_num(shape)
    training_type_feature = organizer_type_to_training_feature(raw_type)

    target_histories, neighbor_features, signal_features, type_indices = [], [], [], []
    origins, yaws, current_world_velocities = [], [], []
    num_agents = int(position.shape[0])

    for target_id in target_ids:
        if target_id < 0 or target_id >= num_agents:
            raise IndexError(f"target agent index out of range: {target_id} / {num_agents}")
        target_type = int(raw_type[target_id])
        if target_type not in (0, 1, 2):
            raise ValueError(f"manifest target {target_id} has unsupported organizer type {target_type}")

        origin = position[target_id, CURRENT_STEP]
        yaw = heading[target_id, CURRENT_STEP]
        target_pos = local_xy(position[target_id], origin, yaw)
        target_vel = local_vec(velocity[target_id], yaw)
        target_history = torch.cat([target_pos, (heading[target_id] - yaw).unsqueeze(-1), target_vel], dim=-1)
        target_history = torch.where(valid[target_id].unsqueeze(-1), target_history, torch.zeros_like(target_history))

        current_local = local_xy(position[:, CURRENT_STEP], origin, yaw)
        current_vel_local = local_vec(velocity[:, CURRENT_STEP], yaw)
        candidates = valid[:, CURRENT_STEP].clone()
        candidates[target_id] = False
        if num_agents <= 1:
            neighbors = torch.zeros((NEIGHBOR_K, NEIGHBOR_FEAT_DIM), dtype=torch.float32)
        else:
            nidx, is_lane, is_dir, valid_pick = select_neighbor_indices(
                current_local, heading[:, CURRENT_STEP], yaw, candidates, neighbor_k=NEIGHBOR_K,
            )
            neighbors = pack_neighbor_features(
                current_local, current_vel_local, shape, training_type_feature,
                nidx, is_lane, is_dir, valid_pick, NEIGHBOR_K,
            )

        # V1 was designed to tolerate missing signal data. The public PKL schema has
        # map point light metadata but no pre-built signal-track tensor used in training,
        # so retain the training code's no-signal fallback rather than infer a mismatched feature.
        signals = torch.zeros((SIGNAL_K, 4), dtype=torch.float32)

        target_histories.append(target_history)
        neighbor_features.append(neighbors)
        signal_features.append(signals)
        type_indices.append(target_type)  # organizer 0/1/2 == V1 embedding vehicle/pedestrian/cyclist indices
        origins.append(origin)
        yaws.append(yaw)
        current_world_velocities.append(velocity[target_id, CURRENT_STEP])

    return (
        {
            "target_hist": torch.stack(target_histories),
            "neighbors": torch.stack(neighbor_features),
            "signals": torch.stack(signal_features),
            "type_idx": torch.tensor(type_indices, dtype=torch.long),
        },
        torch.stack(origins),
        torch.stack(yaws),
        torch.stack(current_world_velocities),
    )


def run_model(
    model: MotionPredictor,
    features: dict[str, torch.Tensor],
    origins: torch.Tensor,
    yaws: torch.Tensor,
    current_world_velocities: torch.Tensor,
    device: torch.device,
    target_batch: int,
    timed: bool = False,
) -> tuple[np.ndarray, float]:
    """Run all targets in one scene. Timing excludes feature construction and host-to-device preparation."""
    outputs: list[torch.Tensor] = []
    total_ms = 0.0
    count = int(features["target_hist"].shape[0])
    for start in range(0, count, target_batch):
        stop = min(count, start + target_batch)
        batch = {name: value[start:stop].to(device, non_blocking=True) for name, value in features.items()}
        origin = origins[start:stop].to(device, non_blocking=True)
        yaw = yaws[start:stop].to(device, non_blocking=True)
        current_velocity = current_world_velocities[start:stop].to(device, non_blocking=True)
        if timed and device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_local, _, logits = model(batch["target_hist"], batch["neighbors"], batch["signals"], batch["type_idx"])
            mode_order = torch.argsort(logits, dim=1, descending=True)
            pred_local = torch.gather(pred_local, 1, mode_order[:, :, None, None].expand(-1, -1, FUTURE_STEPS, 2))
            pred_world = local_to_world(pred_local, origin, yaw)
            bad_rows = ~torch.isfinite(pred_world).all(dim=(1, 2, 3))
            if bad_rows.any():
                horizon_s = torch.arange(1, FUTURE_STEPS + 1, device=device, dtype=pred_world.dtype) * 0.1
                fallback = origin[:, None, None, :] + current_velocity[:, None, None, :] * horizon_s[None, None, :, None]
                pred_world[bad_rows] = fallback[bad_rows].expand(-1, K_MODES, -1, -1)
        if timed and device.type == "cuda":
            torch.cuda.synchronize(device)
        if timed:
            total_ms += (time.perf_counter() - t0) * 1000.0
        outputs.append(pred_world.detach().cpu())
    output = torch.cat(outputs, dim=0)
    if not torch.isfinite(output).all():
        raise RuntimeError("Fallback did not resolve non-finite predictions")
    return output.numpy().astype(np.float32), total_ms


def profile_flops(
    model: MotionPredictor,
    features: dict[str, torch.Tensor],
    origins: torch.Tensor,
    yaws: torch.Tensor,
    current_world_velocities: torch.Tensor,
    device: torch.device,
    target_batch: int,
) -> float:
    """Profile a full scene's K=6 output calculation using torch.profiler with FLOPs enabled."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=False) as profiler:
        _ = run_model(model, features, origins, yaws, current_world_velocities, device, target_batch, timed=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return float(sum(float(getattr(event, "flops", 0) or 0) for event in profiler.key_averages()) / 1e9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-batch", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="0 means every manifest scene")
    parser.add_argument("--time-warmup-scenes", type=int, default=10)
    parser.add_argument("--time-sample-scenes", type=int, default=100)
    parser.add_argument("--flops-samples", type=int, default=5)
    parser.add_argument("--skip-measurements", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.target_batch < 1:
        raise ValueError("--target-batch must be >= 1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with args.manifest.open(encoding="utf-8") as handle:
        manifest_raw = json.load(handle)
    manifest = {str(scene): [int(agent_id) for agent_id in agent_ids] for scene, agent_ids in manifest_raw.items()}
    scene_names = sorted(manifest)
    if args.limit:
        scene_names = scene_names[:args.limit]
    if not scene_names:
        raise ValueError("No manifest scenes selected")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hidden = int(checkpoint.get("args", {}).get("hidden", 256))
    model = MotionPredictor(hidden=hidden, modes=K_MODES).to(device)
    model.load_state_dict(expand_neighbor_encoder_state(checkpoint["model_state"]), strict=False)
    model.eval()
    torch.set_float32_matmul_precision("high")

    arrays: dict[str, np.ndarray] = {}
    timed_ms: list[float] = []
    profiled_gflops: list[float] = []
    started = time.time()

    for scene_number, scene_name in enumerate(scene_names):
        pkl_path = args.data_dir / f"{scene_name}.pkl"
        if not pkl_path.is_file():
            raise FileNotFoundError(f"Missing test scene: {pkl_path}")
        target_ids = manifest[scene_name]
        scene = load_scene(pkl_path)
        features, origins, yaws, current_world_velocities = build_features(scene, target_ids)
        do_time = (not args.skip_measurements and scene_number >= args.time_warmup_scenes and len(timed_ms) < args.time_sample_scenes)
        predictions, elapsed_ms = run_model(model, features, origins, yaws, current_world_velocities, device, args.target_batch, timed=do_time)
        if predictions.shape != (len(target_ids), K_MODES, FUTURE_STEPS, 2):
            raise RuntimeError(f"Unexpected prediction shape for {scene_name}: {predictions.shape}")
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"NaN/Inf prediction in {scene_name}")
        if do_time:
            timed_ms.append(elapsed_ms)
        if not args.skip_measurements and len(profiled_gflops) < args.flops_samples:
            profiled_gflops.append(profile_flops(model, features, origins, yaws, current_world_velocities, device, args.target_batch))

        arrays[f"{scene_name}||ids"] = np.asarray(target_ids, dtype=np.int64)
        arrays[f"{scene_name}||traj"] = predictions
        if (scene_number + 1) % 50 == 0 or scene_number + 1 == len(scene_names):
            elapsed = time.time() - started
            print(f"[{scene_number + 1}/{len(scene_names)}] {elapsed:.1f}s elapsed | targets={len(target_ids)}", flush=True)

    meta: dict[str, Any] = {
        "format_version": "1.0",
        "seed": args.seed,
        "k": K_MODES,
        "ckpt": args.checkpoint.name,
        "generator": "motion_prediction_model_v1_test_public_adapter",
        "device": str(device),
        "target_batch": args.target_batch,
        "scene_count": len(scene_names),
        "note": "Mode 0 is reordered to V1's highest-logit representative trajectory; modes 1-5 are remaining V1 modes.",
    }
    if timed_ms:
        meta["t_infer_ms"] = round(float(np.mean(timed_ms)), 3)
        meta["t_infer_sample_scenes"] = len(timed_ms)
        meta["t_infer_warmup_scenes"] = args.time_warmup_scenes
    if profiled_gflops:
        meta["flops_gflops"] = round(float(np.mean(profiled_gflops)), 3)
        meta["flops_sample_scenes"] = len(profiled_gflops)

    arrays["__scenes__"] = np.asarray(scene_names, dtype=str)
    arrays["__meta_json__"] = np.asarray(json.dumps(meta, ensure_ascii=False), dtype=str)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"SAVED={args.out}")
    print("META=" + json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
