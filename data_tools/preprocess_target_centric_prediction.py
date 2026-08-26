#!/usr/bin/env python3
"""Create target-centric motion-prediction training packs from raw TFRecord.

This script is intentionally self-contained and read-only with respect to raw data.
It writes only new files under the --output-dir. It does not require TensorFlow:
standard tf.train.Example payloads are decoded through the previously prepared
minimal protobuf helper module.

Output design
-------------
One .pt file is produced per raw TFRecord record. A pack stores:
- static roadgraph once;
- three observation windows (each has only past/current information);
- target-only future labels for each window.
This prevents future information from entering model input tensors and avoids
copying map data once per target.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch

PAST_STEPS = 150
FUTURE_STEPS = 150
OBS_STEPS = 11
PRED_STEPS = 80
DEFAULT_ANCHORS = (10, 110, 210)  # global 10 Hz frame indices: 1s, 11s, 21s
SCHEMA_VERSION = "target_centric_prediction_v1"
TYPE_NAMES = {0: "unknown_or_padding", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 5: "other_static_or_misc"}


def iter_tfrecord_records(path: str) -> Iterator[bytes]:
    """Yield payloads of a TFRecord file without TensorFlow."""
    with open(path, "rb") as handle:
        while True:
            header = handle.read(12)  # uint64 length + uint32 crc
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord header: {path}")
            length = struct.unpack("<Q", header[:8])[0]
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"Truncated TFRecord payload: {path}")
            data_crc = handle.read(4)
            if len(data_crc) != 4:
                raise ValueError(f"Truncated TFRecord CRC: {path}")
            yield payload


def to_numpy(features: Dict[str, Tuple[str, list]], key: str, dtype: np.dtype, default: float = 0.0) -> np.ndarray:
    kind_values = features.get(key)
    if kind_values is None:
        return np.asarray([], dtype=dtype)
    _, values = kind_values
    # In protobuf, negative int64 values are serialized as 10-byte unsigned
    # varints. The minimal decoder returns that unsigned representation, so
    # restore signed two's-complement values before requesting np.int64.
    if np.dtype(dtype) == np.dtype(np.int64):
        values = [int(v) - (1 << 64) if isinstance(v, (int, np.integer)) and int(v) >= (1 << 63) else v for v in values]
    return np.asarray(values, dtype=dtype)


def reshape_track_time(values: np.ndarray, n_tracks: int, n_steps: int, key: str) -> np.ndarray:
    expected = n_tracks * n_steps
    if values.size != expected:
        raise ValueError(f"{key}: expected {expected} values, got {values.size}")
    return values.reshape(n_tracks, n_steps)


def concat_track_series(features: Dict[str, Tuple[str, list]], stem: str, n_tracks: int, dtype: np.dtype) -> np.ndarray:
    past = reshape_track_time(to_numpy(features, f"state/past/{stem}", dtype), n_tracks, PAST_STEPS, f"state/past/{stem}")
    current = to_numpy(features, f"state/current/{stem}", dtype)
    if current.size != n_tracks:
        raise ValueError(f"state/current/{stem}: expected {n_tracks}, got {current.size}")
    future = reshape_track_time(to_numpy(features, f"state/future/{stem}", dtype), n_tracks, FUTURE_STEPS, f"state/future/{stem}")
    return np.concatenate([past, current[:, None], future], axis=1)


def concat_signal_series(features: Dict[str, Tuple[str, list]], stem: str, n_signals: int, dtype: np.dtype) -> np.ndarray:
    if n_signals == 0:
        return np.zeros((0, PAST_STEPS + 1 + FUTURE_STEPS), dtype=dtype)
    past = reshape_track_time(to_numpy(features, f"traffic_light_state/past/{stem}", dtype), n_signals, PAST_STEPS, f"traffic_light_state/past/{stem}")
    current = to_numpy(features, f"traffic_light_state/current/{stem}", dtype)
    if current.size != n_signals:
        raise ValueError(f"traffic_light_state/current/{stem}: expected {n_signals}, got {current.size}")
    future = reshape_track_time(to_numpy(features, f"traffic_light_state/future/{stem}", dtype), n_signals, FUTURE_STEPS, f"traffic_light_state/future/{stem}")
    return np.concatenate([past, current[:, None], future], axis=1)


def decode_scenario_id(features: Dict[str, Tuple[str, list]], fallback: str) -> str:
    item = features.get("scenario/id")
    if item is None:
        return fallback
    kind, values = item
    if kind == "bytes" and values:
        return bytes(values[0]).decode("utf-8", errors="replace")
    return fallback


def stable_split(relative_source: str, val_percent: int) -> str:
    digest = hashlib.sha1(relative_source.encode("utf-8")).hexdigest()
    return "val" if int(digest[:8], 16) % 100 < val_percent else "train"


def sanitize_identifier(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)[:80]


def build_pack(features: Dict[str, Tuple[str, list]], source_rel: str, record_index: int, target_types: set[int]) -> Dict[str, Any] | None:
    is_sdc = to_numpy(features, "state/is_sdc", np.int64)
    n_tracks = int(is_sdc.size)
    if n_tracks == 0:
        raise ValueError("state/is_sdc is missing")
    ego_rows = np.flatnonzero(is_sdc == 1)
    if ego_rows.size != 1:
        raise ValueError(f"Expected one SDC row, got {ego_rows.size}")
    ego_row = int(ego_rows[0])

    # Past/current/future are concatenated only in memory. Output windows expose
    # observations and target labels separately so model inputs never carry future.
    x = concat_track_series(features, "x", n_tracks, np.float32)
    y = concat_track_series(features, "y", n_tracks, np.float32)
    yaw = concat_track_series(features, "bbox_yaw", n_tracks, np.float32)
    vx = concat_track_series(features, "velocity_x", n_tracks, np.float32)
    vy = concat_track_series(features, "velocity_y", n_tracks, np.float32)
    valid = concat_track_series(features, "valid", n_tracks, np.int64).astype(np.bool_)

    agent_type_float = to_numpy(features, "state/type", np.float32)
    if agent_type_float.size != n_tracks:
        raise ValueError(f"state/type: expected {n_tracks}, got {agent_type_float.size}")
    agent_type = np.rint(agent_type_float).astype(np.int16)
    agent_ids = to_numpy(features, "state/id", np.float64)
    if agent_ids.size != n_tracks:
        agent_ids = np.arange(n_tracks, dtype=np.int64)
    lengths = concat_track_series(features, "length", n_tracks, np.float32)
    widths = concat_track_series(features, "width", n_tracks, np.float32)
    heights = concat_track_series(features, "height", n_tracks, np.float32)

    # Static roadgraph. Values are stored in global/world coordinates and will be
    # transformed into target-local coordinates by the training dataset.
    road_xyz = to_numpy(features, "roadgraph_samples/xyz", np.float32)
    if road_xyz.size % 3 != 0:
        raise ValueError("roadgraph_samples/xyz is not divisible by 3")
    n_road = road_xyz.size // 3
    road_xyz = road_xyz.reshape(n_road, 3)
    road_dir = to_numpy(features, "roadgraph_samples/dir", np.float32).reshape(n_road, 3)
    road_type = np.rint(to_numpy(features, "roadgraph_samples/type", np.float32)).astype(np.int16)
    road_valid = to_numpy(features, "roadgraph_samples/valid", np.int64).astype(np.bool_)
    if road_type.size != n_road or road_valid.size != n_road:
        raise ValueError("roadgraph type/valid size mismatch")

    # Signal IDs are encoded as unsigned protobuf varints and can exceed signed
    # int64 range. Preserve their bit pattern as uint64; their values are static
    # metadata and are never used for arithmetic in the converter.
    signal_ids = to_numpy(features, "traffic_light_state/current/id", np.uint64)
    n_signals = int(signal_ids.size)
    if n_signals:
        sig_x = concat_signal_series(features, "x", n_signals, np.float32)
        sig_y = concat_signal_series(features, "y", n_signals, np.float32)
        sig_state = concat_signal_series(features, "state", n_signals, np.int64).astype(np.int16)
        sig_valid = concat_signal_series(features, "valid", n_signals, np.int64).astype(np.bool_)
    else:
        sig_x = np.zeros((0, 301), dtype=np.float32)
        sig_y = np.zeros((0, 301), dtype=np.float32)
        sig_state = np.zeros((0, 301), dtype=np.int16)
        sig_valid = np.zeros((0, 301), dtype=np.bool_)

    windows: List[Dict[str, Any]] = []
    for window_index, anchor in enumerate(DEFAULT_ANCHORS):
        obs_start = anchor - (OBS_STEPS - 1)
        pred_start = anchor + 1
        pred_end = pred_start + PRED_STEPS
        if obs_start < 0 or pred_end > x.shape[1]:
            raise ValueError(f"Anchor {anchor} is outside available range")

        ego_xy = np.array([x[ego_row, anchor], y[ego_row, anchor]], dtype=np.float32)
        distance = np.hypot(x[:, anchor] - ego_xy[0], y[:, anchor] - ego_xy[1])
        future_valid = valid[:, pred_start:pred_end]
        target_mask = (
            valid[:, anchor]
            & (np.arange(n_tracks) != ego_row)
            & (distance <= 50.0)
            & np.isin(agent_type, list(target_types))
            & future_valid.any(axis=1)
        )
        target_rows = np.flatnonzero(target_mask).astype(np.int16)

        # Five dynamic input features, all from observation window only.
        agent_history = np.stack([
            x[:, obs_start:anchor + 1],
            y[:, obs_start:anchor + 1],
            yaw[:, obs_start:anchor + 1],
            vx[:, obs_start:anchor + 1],
            vy[:, obs_start:anchor + 1],
        ], axis=-1).astype(np.float32)
        agent_history_valid = valid[:, obs_start:anchor + 1]
        agent_size = np.stack([lengths[:, anchor], widths[:, anchor], heights[:, anchor]], axis=-1).astype(np.float32)

        # Signal history contains only past/current values. future signal state is
        # deliberately not saved as an input feature in this initial baseline.
        signal_history = np.stack([
            sig_x[:, obs_start:anchor + 1],
            sig_y[:, obs_start:anchor + 1],
            sig_state[:, obs_start:anchor + 1].astype(np.float32),
        ], axis=-1).astype(np.float32)
        signal_history_valid = sig_valid[:, obs_start:anchor + 1]

        windows.append({
            "window_index": int(window_index),
            "anchor_global_step": int(anchor),
            "observation_global_steps": torch.arange(obs_start, anchor + 1, dtype=torch.int16),
            "prediction_global_steps": torch.arange(pred_start, pred_end, dtype=torch.int16),
            "ego_row": int(ego_row),
            "inputs": {
                "agent_history_world": torch.from_numpy(agent_history),
                "agent_history_valid": torch.from_numpy(agent_history_valid),
                "agent_size_m": torch.from_numpy(agent_size),
                "signal_history_world_state": torch.from_numpy(signal_history),
                "signal_history_valid": torch.from_numpy(signal_history_valid),
            },
            "targets": {
                "target_rows": torch.from_numpy(target_rows),
                "target_types": torch.from_numpy(agent_type[target_rows].astype(np.int16)),
                "future_xy_world": torch.from_numpy(np.stack([x[target_rows, pred_start:pred_end], y[target_rows, pred_start:pred_end]], axis=-1).astype(np.float32)),
                "future_valid": torch.from_numpy(future_valid[target_rows]),
            },
        })

    if not any(int(w["targets"]["target_rows"].numel()) > 0 for w in windows):
        return None

    fallback = f"{Path(source_rel).stem}_r{record_index:03d}"
    scenario_id = decode_scenario_id(features, fallback)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_tfrecord": source_rel,
        "record_index": int(record_index),
        "scenario_id": scenario_id,
        "time_hz": 10,
        "observation_seconds": 1.0,
        "prediction_seconds": 8.0,
        "type_mapping": TYPE_NAMES,
        "static": {
            "agent_ids_raw": torch.from_numpy(agent_ids),
            "agent_types": torch.from_numpy(agent_type),
            "agent_is_sdc": torch.from_numpy(is_sdc.astype(np.bool_)),
            "roadgraph_xyz_world": torch.from_numpy(road_xyz.astype(np.float32)),
            "roadgraph_dir_world": torch.from_numpy(road_dir.astype(np.float32)),
            "roadgraph_type": torch.from_numpy(road_type),
            "roadgraph_valid": torch.from_numpy(road_valid),
            "signal_ids": torch.from_numpy(signal_ids.astype(np.uint64)),
        },
        "windows": windows,
    }


def parse_target_types(text: str) -> set[int]:
    values = {int(part.strip()) for part in text.split(",") if part.strip()}
    if not values:
        raise ValueError("--target-types must have at least one type code")
    return values


def write_metadata(output_dir: Path, args: argparse.Namespace, total_files: int) -> None:
    path = output_dir / "metadata" / "schema.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "raw_data_dir": os.path.abspath(args.raw_dir),
        "source_file_count_discovered": total_files,
        "anchors_global_steps": list(DEFAULT_ANCHORS),
        "window_definition": {"observation_steps": OBS_STEPS, "prediction_steps": PRED_STEPS, "time_hz": 10},
        "target_selection": {"exclude_sdc": True, "radius_m": 50.0, "types": sorted(parse_target_types(args.target_types)), "requires_any_future_valid": True},
        "storage": "one static context + three windows per raw record; inputs do not contain future target/signal states",
        "split": {"method": "stable SHA-1 hash of source relative path", "val_percent": args.val_percent},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only TFRecord to target-centric motion-prediction pack converter")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--helper-dir", required=True, help="Directory containing raw_tfrecord_type_probe.py")
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files")
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--target-types", default="1,2,3")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.val_percent < 100:
        raise ValueError("--val-percent must be in [0, 100)")

    sys.path.insert(0, args.helper_dir)
    from raw_tfrecord_type_probe import parse_example

    output_dir = Path(args.output_dir)
    for split in ("train", "val"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(glob.glob(os.path.join(args.raw_dir, "**", "*.tfrecord"), recursive=True))
    if args.max_files:
        raw_paths = raw_paths[: args.max_files]
    if not raw_paths:
        raise FileNotFoundError(f"No .tfrecord files under {args.raw_dir}")
    target_types = parse_target_types(args.target_types)
    write_metadata(output_dir, args, len(raw_paths))

    manifest_handles = {
        split: open(output_dir / "manifests" / f"{split}_manifest.jsonl", "a", encoding="utf-8")
        for split in ("train", "val")
    }
    summary = {"discovered_files": len(raw_paths), "processed_records": 0, "written_packs": 0, "skipped_no_targets": 0, "failed_files": 0, "windows": 0, "targets": 0, "errors": []}
    raw_root = os.path.abspath(args.raw_dir)

    try:
        for file_index, raw_path in enumerate(raw_paths, start=1):
            source_rel = os.path.relpath(raw_path, raw_root).replace(os.sep, "/")
            split = stable_split(source_rel, args.val_percent)
            try:
                for record_index, record_bytes in enumerate(iter_tfrecord_records(raw_path)):
                    summary["processed_records"] += 1
                    features = parse_example(record_bytes)
                    pack = build_pack(features, source_rel, record_index, target_types)
                    if pack is None:
                        summary["skipped_no_targets"] += 1
                        continue
                    content_key = hashlib.sha1(f"{source_rel}:{record_index}".encode("utf-8")).hexdigest()[:16]
                    file_name = f"{content_key}_r{record_index:03d}.pt"
                    relative_output = f"{split}/{file_name}"
                    output_path = output_dir / relative_output
                    if output_path.exists() and not args.overwrite:
                        continue
                    torch.save(pack, output_path)
                    counts = [int(window["targets"]["target_rows"].numel()) for window in pack["windows"]]
                    manifest_handles[split].write(json.dumps({
                        "path": relative_output,
                        "scenario_id": pack["scenario_id"],
                        "source_tfrecord": source_rel,
                        "record_index": record_index,
                        "target_counts": counts,
                        "total_targets": sum(counts),
                    }, ensure_ascii=False) + "\n")
                    manifest_handles[split].flush()
                    summary["written_packs"] += 1
                    summary["windows"] += len(pack["windows"])
                    summary["targets"] += sum(counts)
            except Exception as exc:
                summary["failed_files"] += 1
                summary["errors"].append({"file": source_rel, "error": repr(exc)})
            if file_index % 10 == 0 or file_index == len(raw_paths):
                print(json.dumps({"progress_files": file_index, **{k: summary[k] for k in ("written_packs", "skipped_no_targets", "failed_files", "targets")}}, ensure_ascii=False), flush=True)
    finally:
        for handle in manifest_handles.values():
            handle.close()

    summary_path = output_dir / "metadata" / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
