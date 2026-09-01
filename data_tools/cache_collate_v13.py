#!/usr/bin/env python3
"""Precompute V13 collate tensors (with 16x20x8 polylines) so training is GPU-bound."""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from src.train_motion_prediction_v1 import list_pt_paths
from src.train_motion_prediction_v13_collate import _scene_v13


def _to_compact(packed: dict) -> dict:
    out = {}
    for k, v in packed.items():
        if torch.is_floating_point(v):
            out[k] = v.detach().to(torch.float16).contiguous()
        else:
            out[k] = v.detach().contiguous()
    return out


def process_one(args_tuple):
    src, dst, overwrite = args_tuple
    if os.path.exists(dst) and not overwrite:
        return True, src, "skip"
    try:
        pack = torch.load(src, map_location="cpu", weights_only=False)
        window = pack["windows"][0]
        packed = _scene_v13(pack["static"], window, max_targets=0, train=False)
        if packed is None:
            return True, src, "empty"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        torch.save(_to_compact(packed), dst)
        return True, src, "ok"
    except Exception as e:
        return False, src, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2")
    parser.add_argument("--out-dir", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2_cache_v13")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-packs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print(" COLLATE CACHE V13 — Polylines 16x20x8 + Tokens + All Targets")
    print(f" In:  {args.data_root}")
    print(f" Out: {args.out_dir}")
    print(f" Workers: {args.workers}")
    print("=" * 80, flush=True)

    tasks = []
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)
        try:
            paths = list_pt_paths(args.data_root, split, args.max_packs)
        except FileNotFoundError:
            continue
        for src in paths:
            dst = os.path.join(args.out_dir, split, os.path.basename(src))
            tasks.append((src, dst, args.overwrite))
    print(f"-> Total packs to process: {len(tasks):,}", flush=True)

    t0 = time.time()
    ok = skip = empty = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), start=1):
            success, src, msg = fut.result()
            if not success:
                fail += 1
                print(f" FAIL {os.path.basename(src)}: {msg}", flush=True)
            elif msg == "skip":
                skip += 1
            elif msg == "empty":
                empty += 1
            else:
                ok += 1
            if i % 2000 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                rate = i / max(0.1, elapsed)
                eta = (len(tasks) - i) / max(0.1, rate)
                print(
                    f" [{i:06d}/{len(tasks):06d}] write {ok:,} skip {skip:,} empty {empty:,} fail {fail} "
                    f"| {rate:.1f}/s ETA {eta/60:.1f}m",
                    flush=True,
                )
    print(f"Done! {ok:,} written, {skip:,} skipped, {empty:,} empty, {fail} failed in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
