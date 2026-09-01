#!/usr/bin/env python3
"""Add roadgraph_id + resampled map polylines to existing prediction_pt_85k packs.

Does not rebuild agent history/targets. Matches the original 85k train/val hash
and filename scheme, then patches static tensors in place.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_85k_dataset import iter_tfrecord_records, to_numpy
from raw_tfrecord_type_probe import parse_example
from src.map_polylines import build_map_polylines


def pack_path(raw_path: str, raw_root: str, output_dir: str, val_percent: int, record_index: int) -> str:
    source_rel = os.path.relpath(raw_path, raw_root).replace(os.sep, "/")
    digest = hashlib.sha1(source_rel.encode("utf-8")).hexdigest()
    split = "val" if (int(digest[:8], 16) % 100) < val_percent else "train"
    content_key = hashlib.sha1(f"{source_rel}:{record_index}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(output_dir, split, f"{content_key}_r{record_index:03d}.pt")


def process_tfrecord(args_tuple):
    raw_path, raw_root, output_dir, val_percent, force = args_tuple
    updated = skipped = missing = mismatched = 0
    try:
        for record_index, record_bytes in enumerate(iter_tfrecord_records(raw_path)):
            out_file = pack_path(raw_path, raw_root, output_dir, val_percent, record_index)
            if not os.path.exists(out_file):
                missing += 1
                continue
            pack = torch.load(out_file, map_location="cpu", weights_only=False)
            static = pack.get("static", {})
            already = ("roadgraph_id" in static) and ("map_polyline_xy" in static)
            if already and not force:
                skipped += 1
                continue
            features = parse_example(record_bytes)
            rg_xyz = to_numpy(features, "roadgraph_samples/xyz", np.float32).reshape(-1, 3)
            rg_dir = to_numpy(features, "roadgraph_samples/dir", np.float32).reshape(-1, 3)
            rg_type = to_numpy(features, "roadgraph_samples/type", np.int64).reshape(-1)
            rg_valid = to_numpy(features, "roadgraph_samples/valid", np.int64).reshape(-1) > 0
            rg_id = to_numpy(features, "roadgraph_samples/id", np.int64).reshape(-1)
            n_pack = int(static["roadgraph_xyz_world"].shape[0]) if "roadgraph_xyz_world" in static else -1
            if rg_id.size != n_pack:
                mismatched += 1
                continue
            static["roadgraph_id"] = torch.from_numpy(rg_id.astype(np.int32))
            static.update(build_map_polylines(rg_xyz, rg_dir, rg_type, rg_valid, rg_id))
            pack["static"] = static
            torch.save(pack, out_file)
            updated += 1
        return (True, raw_path, updated, skipped, missing, mismatched)
    except Exception as exc:
        return (False, raw_path, str(exc), 0, 0, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=r"E:\motion_data\rideflux_91f_full\rideflux")
    parser.add_argument("--out-dir", default=r"E:\motion_planning\data\processed\prediction_pt_85k")
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_files = sorted(glob.glob(os.path.join(args.raw_dir, "**", "*.tfrecord"), recursive=True))
    if args.max_files > 0:
        raw_files = raw_files[: args.max_files]
    print("=" * 80)
    print(" BACKFILL roadgraph_id + map polylines into existing 85k .pt packs")
    print(f" Raw TFRecords: {args.raw_dir}")
    print(f" Packs:         {args.out_dir}")
    print(f" Files:         {len(raw_files):,}  workers={args.workers}  force={args.force}")
    print("=" * 80, flush=True)

    tasks = [(path, args.raw_dir, args.out_dir, args.val_percent, args.force) for path in raw_files]
    start = time.time()
    tot_u = tot_s = tot_m = tot_x = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_tfrecord, task): task[0] for task in tasks}
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            if not res[0]:
                fail += 1
                print(f" FAIL {res[1]}: {res[2]}", flush=True)
            else:
                _, _, updated, skipped, missing, mismatched = res
                tot_u += updated
                tot_s += skipped
                tot_m += missing
                tot_x += mismatched
            if i % 200 == 0 or i == len(tasks):
                dt = max(1e-6, time.time() - start)
                print(
                    f" [{i}/{len(tasks)}] {i/dt:.1f} files/s | "
                    f"updated={tot_u} skipped={tot_s} missing_pt={tot_m} id_mismatch={tot_x} fail={fail}",
                    flush=True,
                )
    print(
        f"Done in {time.time()-start:.1f}s | updated={tot_u} skipped={tot_s} "
        f"missing_pt={tot_m} id_mismatch={tot_x} fail={fail}",
        flush=True,
    )


if __name__ == "__main__":
    main()
