#!/usr/bin/env python3
"""Evaluate SWA (Stochastic Weight Averaging) on Motion Prediction V17 X-Large.

Averages weights from:
- best_error_score.pth
- best_minade6.pth
- last.pth (Epoch 30)

Then runs full Density NMS evaluation across all validation scenes!
"""
from __future__ import annotations

import argparse, glob, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cached_collate_v13 import CachedWindowCollateV13, try_cached_loader_v13
from src.train_motion_prediction_v17 import MotionPredictorV17
from train_motion_prediction_v2_awta import configure_runtime, make_loader
from evaluate_v17_nms import evaluate_nms_v17


def average_checkpoints_v17(ckpt_paths: list[str]) -> dict:
    print(f"-> Averaging {len(ckpt_paths)} V17 checkpoints:")
    for p in ckpt_paths:
        print(f"   * {p}")

    avg_state = {}
    n = len(ckpt_paths)
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        for k, v in state.items():
            if k not in avg_state:
                avg_state[k] = v.clone().float()
            else:
                avg_state[k] += v.float()

    for k in avg_state:
        avg_state[k] /= n
    return avg_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=r"E:\motion_prediction\checkpoints\v17_xlarge")
    parser.add_argument("--cache-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2_cache_v13")
    parser.add_argument("--batch-scenes", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", default="bf16")
    args = parser.parse_args()

    configure_runtime(args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else torch.float32)

    # Collect available top checkpoints
    cand_files = [
        os.path.join(args.ckpt_dir, "best_error_score.pth"),
        os.path.join(args.ckpt_dir, "best_minade6.pth"),
        os.path.join(args.ckpt_dir, "last.pth"),
    ]
    ckpt_paths = [f for f in cand_files if os.path.exists(f)]
    if not ckpt_paths:
        ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pth")))

    print("=" * 80)
    print(" SWA CHECKPOINT AVERAGING + SOFT DENSITY NMS EVALUATION (V17 X-LARGE)")
    print("=" * 80)

    avg_state = average_checkpoints_v17(ckpt_paths)

    model = MotionPredictorV17(hidden=768, modes=6, nhead=12, dropout=0.1).to(device)
    model.load_state_dict(avg_state)

    # Save SWA model checkpoint
    swa_path = os.path.join(args.ckpt_dir, "swa_model.pth")
    torch.save({"model_state": avg_state, "averaged_ckpts": ckpt_paths}, swa_path)
    print(f"-> Saved SWA Model Checkpoint: {swa_path}")

    val_ds = try_cached_loader_v13(args.cache_root, "val", 0)
    val_collate = CachedWindowCollateV13(args.batch_scenes, False)
    val_loader = make_loader(val_ds, args.batch_scenes, args.workers, 4, False, val_collate, False)

    evaluate_nms_v17(model, val_loader, device, amp_dtype)


if __name__ == "__main__":
    main()
