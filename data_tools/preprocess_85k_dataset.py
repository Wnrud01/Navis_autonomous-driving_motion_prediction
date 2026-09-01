#!/usr/bin/env python3
"""High-Speed Multi-Core Preprocessor for 85k Rideflux 91-frame TFRecord Scenarios.
Converts 85,126 raw .tfrecord files into PyTorch target-centric .pt packs.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, struct, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raw_tfrecord_type_probe import parse_example
from src.map_polylines import build_map_polylines

PAST_STEPS = 10
FUTURE_STEPS = 80
OBS_STEPS = 11
PRED_STEPS = 80
ANCHOR_STEP = 10
TYPE_NAMES = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 5: "misc"}

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
        values = [int(v) - (1 << 64) if isinstance(v, (int, np.integer)) and int(v) >= (1 << 63) else v for v in values]
    return np.asarray(values, dtype=dtype)

def process_single_file(args_tuple):
    raw_path, raw_root, output_dir, val_percent, overwrite = args_tuple
    source_rel = os.path.relpath(raw_path, raw_root).replace(os.sep, "/")
    
    # Deterministic Train/Val Split Hash
    digest = hashlib.sha1(source_rel.encode("utf-8")).hexdigest()
    split = "val" if (int(digest[:8], 16) % 100) < val_percent else "train"
    
    written_count = 0
    targets_count = 0

    try:
        for record_index, record_bytes in enumerate(iter_tfrecord_records(raw_path)):
            content_key = hashlib.sha1(f"{source_rel}:{record_index}".encode("utf-8")).hexdigest()[:16]
            file_name = f"{content_key}_r{record_index:03d}.pt"
            out_file = os.path.join(output_dir, split, file_name)

            if os.path.exists(out_file) and not overwrite:
                written_count += 1
                continue

            features = parse_example(record_bytes)
            
            # Extract basic static features
            scenario_id = str(features.get("scenario/id", ("", [b""]))[1][0])
            agent_ids = to_numpy(features, "state/id", np.int64)
            n_tracks = agent_ids.size
            if n_tracks == 0:
                continue

            agent_types = to_numpy(features, "state/type", np.int64)
            agent_is_sdc = to_numpy(features, "state/is_sdc", np.int64)

            # Agent lengths, widths, heights
            lengths = to_numpy(features, "state/length", np.float32)
            widths = to_numpy(features, "state/width", np.float32)
            heights = to_numpy(features, "state/height", np.float32)
            agent_sizes = np.stack([lengths, widths, heights], axis=-1) if lengths.size == n_tracks else np.zeros((n_tracks, 3), dtype=np.float32)

            # Track positions & velocities: past (10) + current (1) + future (80) = 91
            past_x = to_numpy(features, "state/past/x", np.float32).reshape(n_tracks, PAST_STEPS)
            past_y = to_numpy(features, "state/past/y", np.float32).reshape(n_tracks, PAST_STEPS)
            past_yaw = to_numpy(features, "state/past/bbox_yaw", np.float32).reshape(n_tracks, PAST_STEPS)
            past_vx = to_numpy(features, "state/past/velocity_x", np.float32).reshape(n_tracks, PAST_STEPS)
            past_vy = to_numpy(features, "state/past/velocity_y", np.float32).reshape(n_tracks, PAST_STEPS)
            past_v = to_numpy(features, "state/past/valid", np.int64).reshape(n_tracks, PAST_STEPS)

            cur_x = to_numpy(features, "state/current/x", np.float32).reshape(n_tracks, 1)
            cur_y = to_numpy(features, "state/current/y", np.float32).reshape(n_tracks, 1)
            cur_yaw = to_numpy(features, "state/current/bbox_yaw", np.float32).reshape(n_tracks, 1)
            cur_vx = to_numpy(features, "state/current/velocity_x", np.float32).reshape(n_tracks, 1)
            cur_vy = to_numpy(features, "state/current/velocity_y", np.float32).reshape(n_tracks, 1)
            cur_v = to_numpy(features, "state/current/valid", np.int64).reshape(n_tracks, 1)

            fut_x = to_numpy(features, "state/future/x", np.float32).reshape(n_tracks, FUTURE_STEPS)
            fut_y = to_numpy(features, "state/future/y", np.float32).reshape(n_tracks, FUTURE_STEPS)
            fut_yaw = to_numpy(features, "state/future/bbox_yaw", np.float32).reshape(n_tracks, FUTURE_STEPS)
            fut_vx = to_numpy(features, "state/future/velocity_x", np.float32).reshape(n_tracks, FUTURE_STEPS)
            fut_vy = to_numpy(features, "state/future/velocity_y", np.float32).reshape(n_tracks, FUTURE_STEPS)
            fut_v = to_numpy(features, "state/future/valid", np.int64).reshape(n_tracks, FUTURE_STEPS)

            hist_x = np.concatenate([past_x, cur_x], axis=1) # [N, 11]
            hist_y = np.concatenate([past_y, cur_y], axis=1)
            hist_yaw = np.concatenate([past_yaw, cur_yaw], axis=1)
            hist_vx = np.concatenate([past_vx, cur_vx], axis=1)
            hist_vy = np.concatenate([past_vy, cur_vy], axis=1)
            hist_valid = np.concatenate([past_v, cur_v], axis=1) > 0

            hist_world = np.stack([hist_x, hist_y, hist_yaw, hist_vx, hist_vy], axis=-1) # [N, 11, 5]
            fut_world = np.stack([fut_x, fut_y, fut_yaw, fut_vx, fut_vy], axis=-1)       # [N, 80, 5]
            fut_valid = (fut_v > 0)                                                      # [N, 80]

            # Signal states
            sig_xyz = to_numpy(features, "traffic_light_state/current/xyz", np.float32).reshape(-1, 3) if "traffic_light_state/current/xyz" in features else np.zeros((0, 3), dtype=np.float32)
            n_signals = sig_xyz.shape[0]
            if n_signals > 0:
                sig_past_state = to_numpy(features, "traffic_light_state/past/state", np.int64).reshape(n_signals, PAST_STEPS)
                sig_cur_state = to_numpy(features, "traffic_light_state/current/state", np.int64).reshape(n_signals, 1)
                sig_past_v = to_numpy(features, "traffic_light_state/past/valid", np.int64).reshape(n_signals, PAST_STEPS)
                sig_cur_v = to_numpy(features, "traffic_light_state/current/valid", np.int64).reshape(n_signals, 1)

                sig_state = np.concatenate([sig_past_state, sig_cur_state], axis=1)
                sig_v = np.concatenate([sig_past_v, sig_cur_v], axis=1) > 0
                sig_xyz_11 = np.repeat(sig_xyz[:, None, :2], OBS_STEPS, axis=1)
                sig_world_state = np.concatenate([sig_xyz_11, sig_state[:, :, None]], axis=-1) # [S, 11, 3]
            else:
                sig_world_state = np.zeros((0, OBS_STEPS, 3), dtype=np.float32)
                sig_v = np.zeros((0, OBS_STEPS), dtype=bool)

            # Roadgraph
            rg_xyz = to_numpy(features, "roadgraph_samples/xyz", np.float32).reshape(-1, 3)
            rg_dir = to_numpy(features, "roadgraph_samples/dir", np.float32).reshape(-1, 3)
            rg_type = to_numpy(features, "roadgraph_samples/type", np.int64).reshape(-1)
            rg_valid = to_numpy(features, "roadgraph_samples/valid", np.int64).reshape(-1) > 0
            rg_id = to_numpy(features, "roadgraph_samples/id", np.int64).reshape(-1)
            if rg_id.size != rg_xyz.shape[0]:
                rg_id = np.zeros((rg_xyz.shape[0],), dtype=np.int64)

            # Target eligibility (valid at t=1.0s and active future)
            target_mask = hist_valid[:, -1] & (fut_valid.sum(axis=1) > 0)
            target_rows = np.where(target_mask)[0]

            if len(target_rows) == 0:
                continue

            window = {
                "inputs": {
                    "agent_history_world": torch.from_numpy(hist_world).float(),
                    "agent_history_valid": torch.from_numpy(hist_valid).bool(),
                    "agent_size_m": torch.from_numpy(agent_sizes).float(),
                    "signal_history_world_state": torch.from_numpy(sig_world_state).float(),
                    "signal_history_valid": torch.from_numpy(sig_v).bool(),
                },
                "targets": {
                    "target_rows": torch.from_numpy(target_rows).long(),
                    "agent_future_world": torch.from_numpy(fut_world).float(),
                    "agent_future_valid": torch.from_numpy(fut_valid).bool(),
                }
            }

            pack = {
                "schema_version": "target_centric_prediction_v1",
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
                    **build_map_polylines(rg_xyz, rg_dir, rg_type, rg_valid, rg_id),
                },
                "windows": [window]
            }

            torch.save(pack, out_file)
            written_count += 1
            targets_count += len(target_rows)

        return (True, source_rel, written_count, targets_count)
    except Exception as e:
        return (False, source_rel, str(e), 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=r"E:\motion_data\rideflux_91f_full\rideflux")
    parser.add_argument("--out-dir", default=r"E:\motion_planning\data\processed\prediction_pt_85k")
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "val"), exist_ok=True)

    print("=" * 80)
    print(" 🚀 85k RIDEFLUX DATASET TARGET-CENTRIC PREDICTION PREPROCESSOR")
    print(f" Raw TFRecord Root: {args.raw_dir}")
    print(f" Output PT Root:    {args.out_dir}")
    print(f" Train/Val Split:   {100 - args.val_percent}% Train / {args.val_percent}% Val")
    print(f" Worker Processes:  {args.workers}")
    print("=" * 80, flush=True)

    print("-> Discovering raw .tfrecord files...", flush=True)
    raw_files = sorted(glob.glob(os.path.join(args.raw_dir, "**", "*.tfrecord"), recursive=True))
    if args.max_files > 0:
        raw_files = raw_files[:args.max_files]
    
    total_files = len(raw_files)
    print(f"-> Total TFRecord files discovered: {total_files:,}", flush=True)

    tasks = [
        (fpath, args.raw_dir, args.out_dir, args.val_percent, args.overwrite)
        for fpath in raw_files
    ]

    start_time = time.time()
    success_count = 0
    fail_count = 0
    total_written = 0
    total_targets = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_file, t): t[0] for t in tasks}
        
        for idx, future in enumerate(as_completed(futures), start=1):
            res = future.result()
            ok, s_rel, w_cnt, t_cnt = res
            if ok:
                success_count += 1
                total_written += w_cnt
                total_targets += t_cnt
            else:
                fail_count += 1

            if idx % 1000 == 0 or idx == total_files:
                elapsed = time.time() - start_time
                fps = idx / max(0.1, elapsed)
                remaining = (total_files - idx) / max(0.1, fps)
                print(f" [{idx:05d}/{total_files:05d}] ({idx/total_files*100:.1f}%) | "
                      f"Packs: {total_written:,} | Targets: {total_targets:,} | "
                      f"Speed: {fps:.1f} files/s | ETA: {remaining/60:.1f}m", flush=True)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(" 🎉 85k PREPROCESSING COMPLETED!")
    print(f" - Successfully Converted: {success_count:,} / {total_files:,} files")
    print(f" - Total PT Packs Saved:   {total_written:,}")
    print(f" - Total Targets Extracted: {total_targets:,}")
    print(f" - Total Time Elapsed:     {total_time/60:.1f} minutes")
    print(f" - Target Directory:       {args.out_dir}")
    print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    main()
