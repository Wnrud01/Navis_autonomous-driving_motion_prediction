#!/usr/bin/env python3
"""Rideflux TFRecord → prediction packs v2.

Fixes vs v1:
  - traffic lights from .../x,y (not .../xyz); past+current, no future
  - history [N,11,6] = x,y,yaw,vx,vy,valid
  - polylines KEEP type {1,2} lanes, 15/16 edges, 17 stop, 18 crosswalk
  - keep every currently-valid agent + ego in the pack
"""
from __future__ import annotations

import argparse, glob, hashlib, os, struct, sys, time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raw_tfrecord_type_probe import parse_example
from src.map_polylines import KEEP_TYPES, build_map_polylines

PAST_STEPS = 10
FUTURE_STEPS = 80
OBS_STEPS = 11
SCHEMA = "target_centric_prediction_v2"


def iter_tfrecord_records(path: str):
    with open(path, "rb") as handle:
        while True:
            header = handle.read(12)
            if not header:
                return
            if len(header) != 12:
                return
            length = struct.unpack("<Q", header[:8])[0]
            payload = handle.read(length)
            if len(payload) != length:
                return
            data_crc = handle.read(4)
            if len(data_crc) != 4:
                return
            yield payload


def to_numpy(features: dict, key: str, dtype: np.dtype) -> np.ndarray:
    kind_values = features.get(key)
    if kind_values is None:
        return np.asarray([], dtype=dtype)
    _, values = kind_values
    if np.dtype(dtype) == np.dtype(np.int64):
        values = [
            int(v) - (1 << 64) if isinstance(v, (int, np.integer)) and int(v) >= (1 << 63) else v
            for v in values
        ]
    return np.asarray(values, dtype=dtype)


def _reshape_time(arr: np.ndarray, n: int, steps: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, steps), dtype=arr.dtype if arr.size else np.float32)
    if arr.size == n * steps:
        return arr.reshape(n, steps)
    if arr.size == n and steps == 1:
        return arr.reshape(n, 1)
    if arr.size == 0:
        return np.zeros((n, steps), dtype=np.float32 if arr.dtype == np.float32 else arr.dtype)
    out = np.zeros((n, steps), dtype=arr.dtype if arr.size else np.float32)
    flat = arr.reshape(-1)
    copy_n = min(flat.size, n * steps)
    out.reshape(-1)[:copy_n] = flat[:copy_n]
    return out


def pack_agent_sizes(features: dict, n_tracks: int) -> np.ndarray:
    """state/length does not exist. Sizes live at state/{past,current,future}/{length,width,height}."""
    out = np.zeros((n_tracks, 3), dtype=np.float32)
    for j, stem in enumerate(("length", "width", "height")):
        cur = _reshape_time(to_numpy(features, f"state/current/{stem}", np.float32), n_tracks, 1)[:, 0]
        out[:, j] = cur
        missing = out[:, j] <= 0
        if missing.any():
            past = _reshape_time(to_numpy(features, f"state/past/{stem}", np.float32), n_tracks, PAST_STEPS)
            if past.size:
                out[missing, j] = past[missing, -1]
            missing = out[:, j] <= 0
        if missing.any():
            fut = _reshape_time(to_numpy(features, f"state/future/{stem}", np.float32), n_tracks, FUTURE_STEPS)
            if fut.size:
                out[missing, j] = fut[missing, 0]
    return out


def pack_traffic_lights(features: dict) -> dict[str, torch.Tensor]:
    """past 10 + current 1. Future TL is not stored. Invalid xy must be masked at train."""
    empty = {
        "tl_xy": torch.zeros(0, OBS_STEPS, 2, dtype=torch.float32),
        "tl_state": torch.zeros(0, OBS_STEPS, dtype=torch.int16),
        "tl_valid": torch.zeros(0, OBS_STEPS, dtype=torch.bool),
        "tl_id": torch.zeros(0, OBS_STEPS, dtype=torch.int32),
    }
    cur_x = to_numpy(features, "traffic_light_state/current/x", np.float32)
    n = int(cur_x.size)
    if n == 0:
        return empty
    cur_y = _reshape_time(to_numpy(features, "traffic_light_state/current/y", np.float32), n, 1)
    cur_x = _reshape_time(cur_x, n, 1)
    cur_s = _reshape_time(to_numpy(features, "traffic_light_state/current/state", np.int64), n, 1)
    cur_v = _reshape_time(to_numpy(features, "traffic_light_state/current/valid", np.int64), n, 1)
    cur_id = _reshape_time(to_numpy(features, "traffic_light_state/current/id", np.int64), n, 1)

    past_x = _reshape_time(to_numpy(features, "traffic_light_state/past/x", np.float32), n, PAST_STEPS)
    past_y = _reshape_time(to_numpy(features, "traffic_light_state/past/y", np.float32), n, PAST_STEPS)
    past_s = _reshape_time(to_numpy(features, "traffic_light_state/past/state", np.int64), n, PAST_STEPS)
    past_v = _reshape_time(to_numpy(features, "traffic_light_state/past/valid", np.int64), n, PAST_STEPS)
    past_id = _reshape_time(to_numpy(features, "traffic_light_state/past/id", np.int64), n, PAST_STEPS)

    xy = np.stack(
        [np.concatenate([past_x, cur_x], axis=1), np.concatenate([past_y, cur_y], axis=1)],
        axis=-1,
    ).astype(np.float32)
    state = np.concatenate([past_s, cur_s], axis=1).astype(np.int16)
    valid = np.concatenate([past_v, cur_v], axis=1) > 0
    ids = np.concatenate([past_id, cur_id], axis=1).astype(np.int32)
    return {
        "tl_xy": torch.from_numpy(xy),
        "tl_state": torch.from_numpy(state),
        "tl_valid": torch.from_numpy(valid),
        "tl_id": torch.from_numpy(ids),
    }


def process_single_file(args_tuple):
    raw_path, raw_root, output_dir, val_percent, overwrite = args_tuple
    source_rel = os.path.relpath(raw_path, raw_root).replace(os.sep, "/")
    digest = hashlib.sha1(source_rel.encode("utf-8")).hexdigest()
    split = "val" if (int(digest[:8], 16) % 100) < val_percent else "train"
    written_count = 0
    targets_count = 0
    tl_valid_count = 0
    try:
        for record_index, record_bytes in enumerate(iter_tfrecord_records(raw_path)):
            content_key = hashlib.sha1(f"{source_rel}:{record_index}".encode("utf-8")).hexdigest()[:16]
            out_file = os.path.join(output_dir, split, f"{content_key}_r{record_index:03d}.pt")
            if os.path.exists(out_file) and not overwrite:
                written_count += 1
                continue

            features = parse_example(record_bytes)
            sid_item = features.get("scenario/id", ("", [b""]))
            scenario_id = str(sid_item[1][0]) if sid_item[1] else ""
            agent_ids = to_numpy(features, "state/id", np.int64)
            n_tracks = int(agent_ids.size)
            if n_tracks == 0:
                continue

            agent_types = to_numpy(features, "state/type", np.int64)
            agent_is_sdc = to_numpy(features, "state/is_sdc", np.int64)
            agent_sizes = pack_agent_sizes(features, n_tracks)

            def track(prefix, stem, dtype, steps):
                return _reshape_time(to_numpy(features, f"state/{prefix}/{stem}", dtype), n_tracks, steps)

            past_x = track("past", "x", np.float32, PAST_STEPS)
            past_y = track("past", "y", np.float32, PAST_STEPS)
            past_yaw = track("past", "bbox_yaw", np.float32, PAST_STEPS)
            past_vx = track("past", "velocity_x", np.float32, PAST_STEPS)
            past_vy = track("past", "velocity_y", np.float32, PAST_STEPS)
            past_v = track("past", "valid", np.int64, PAST_STEPS) > 0

            cur_x = track("current", "x", np.float32, 1)
            cur_y = track("current", "y", np.float32, 1)
            cur_yaw = track("current", "bbox_yaw", np.float32, 1)
            cur_vx = track("current", "velocity_x", np.float32, 1)
            cur_vy = track("current", "velocity_y", np.float32, 1)
            cur_v = track("current", "valid", np.int64, 1) > 0

            fut_x = track("future", "x", np.float32, FUTURE_STEPS)
            fut_y = track("future", "y", np.float32, FUTURE_STEPS)
            fut_yaw = track("future", "bbox_yaw", np.float32, FUTURE_STEPS)
            fut_vx = track("future", "velocity_x", np.float32, FUTURE_STEPS)
            fut_vy = track("future", "velocity_y", np.float32, FUTURE_STEPS)
            fut_v = track("future", "valid", np.int64, FUTURE_STEPS) > 0

            hist_valid = np.concatenate([past_v, cur_v], axis=1)
            hist_world = np.stack(
                [
                    np.concatenate([past_x, cur_x], axis=1),
                    np.concatenate([past_y, cur_y], axis=1),
                    np.concatenate([past_yaw, cur_yaw], axis=1),
                    np.concatenate([past_vx, cur_vx], axis=1),
                    np.concatenate([past_vy, cur_vy], axis=1),
                    hist_valid.astype(np.float32),
                ],
                axis=-1,
            )
            fut_world = np.stack([fut_x, fut_y, fut_yaw, fut_vx, fut_vy], axis=-1)

            # Keep every currently-valid object in the pack (neighbors + ego).
            # Loss-target filter (not ego, type 1/2/3, future >= 2s) is collate-only.
            target_rows = np.where(hist_valid[:, -1])[0]
            if target_rows.size == 0:
                continue

            tl = pack_traffic_lights(features)
            tl_valid_count += int(tl["tl_valid"][:, -1].sum()) if tl["tl_valid"].numel() else 0

            rg_xyz = to_numpy(features, "roadgraph_samples/xyz", np.float32)
            if rg_xyz.size:
                rg_xyz = rg_xyz.reshape(-1, 3)
            else:
                rg_xyz = np.zeros((0, 3), dtype=np.float32)
            rg_dir = to_numpy(features, "roadgraph_samples/dir", np.float32)
            rg_dir = rg_dir.reshape(-1, 3) if rg_dir.size else np.zeros((0, 3), dtype=np.float32)
            rg_type = to_numpy(features, "roadgraph_samples/type", np.int64).reshape(-1)
            rg_valid = to_numpy(features, "roadgraph_samples/valid", np.int64).reshape(-1) > 0
            rg_id = to_numpy(features, "roadgraph_samples/id", np.int64).reshape(-1)
            if rg_id.size != rg_xyz.shape[0]:
                rg_id = np.zeros((rg_xyz.shape[0],), dtype=np.int64)

            window = {
                "inputs": {
                    "agent_history_world": torch.from_numpy(hist_world).float(),
                    "agent_history_valid": torch.from_numpy(hist_valid).bool(),
                    "agent_size_m": torch.from_numpy(agent_sizes).float(),
                    **tl,
                },
                "targets": {
                    "target_rows": torch.from_numpy(target_rows.astype(np.int64)),
                    "agent_future_world": torch.from_numpy(fut_world).float(),
                    "agent_future_valid": torch.from_numpy(fut_v).bool(),
                },
            }
            pack = {
                "schema_version": SCHEMA,
                "scenario_id": scenario_id,
                "source_tfrecord": source_rel,
                "static": {
                    "agent_ids_raw": torch.from_numpy(agent_ids).long(),
                    "agent_types": torch.from_numpy(agent_types).long(),
                    "agent_is_sdc": torch.from_numpy(agent_is_sdc).long(),
                    "roadgraph_xyz_world": torch.from_numpy(rg_xyz).float(),
                    "roadgraph_dir_world": torch.from_numpy(rg_dir).float(),
                    "roadgraph_type": torch.from_numpy(rg_type).long(),
                    "roadgraph_valid": torch.from_numpy(rg_valid).bool(),
                    "roadgraph_id": torch.from_numpy(rg_id.astype(np.int32)),
                    **build_map_polylines(rg_xyz, rg_dir, rg_type, rg_valid, rg_id, keep_types=KEEP_TYPES),
                },
                "windows": [window],
            }
            torch.save(pack, out_file)
            written_count += 1
            targets_count += int(target_rows.size)
        return (True, source_rel, written_count, targets_count, tl_valid_count)
    except Exception as e:
        return (False, source_rel, str(e), 0, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=r"E:\motion_data\rideflux_91f_full\rideflux")
    parser.add_argument("--out-dir", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2")
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "val"), exist_ok=True)

    print("=" * 80)
    print(" 85k V2 PREPROCESS — TL x/y, hist 11x6, lanes {1,2}, xwalk 18")
    print(f" Raw: {args.raw_dir}")
    print(f" Out: {args.out_dir}")
    print("=" * 80, flush=True)

    raw_files = sorted(glob.glob(os.path.join(args.raw_dir, "**", "*.tfrecord"), recursive=True))
    if args.max_files > 0:
        raw_files = raw_files[: args.max_files]
    total_files = len(raw_files)
    print(f"-> TFRecord files: {total_files:,}", flush=True)
    tasks = [(f, args.raw_dir, args.out_dir, args.val_percent, args.overwrite) for f in raw_files]

    t0 = time.time()
    ok_n = fail_n = packs = targets = tl_now = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_single_file, t): t[0] for t in tasks}
        for idx, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            ok, rel, w_cnt, t_cnt, tl_cnt = res
            if ok:
                ok_n += 1
                packs += int(w_cnt) if isinstance(w_cnt, (int, np.integer)) else 0
                targets += int(t_cnt) if isinstance(t_cnt, (int, np.integer)) else 0
                tl_now += int(tl_cnt) if isinstance(tl_cnt, (int, np.integer)) else 0
            else:
                fail_n += 1
                print(f" FAIL {rel}: {w_cnt}", flush=True)
            if idx % 1000 == 0 or idx == total_files:
                elapsed = time.time() - t0
                fps = idx / max(0.1, elapsed)
                eta = (total_files - idx) / max(0.1, fps)
                print(
                    f" [{idx:05d}/{total_files:05d}] packs {packs:,} targets {targets:,} "
                    f"tl_valid_now {tl_now:,} | {fps:.1f} files/s ETA {eta/60:.1f}m",
                    flush=True,
                )
    print(
        f"\nDONE files {ok_n}/{total_files} fail {fail_n} packs {packs:,} "
        f"targets {targets:,} tl_valid_now {tl_now:,}  { (time.time()-t0)/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
