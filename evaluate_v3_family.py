#!/usr/bin/env python3
"""Compare V3/V6/V7 checkpoints on the same 24-target val protocol as V3."""
from __future__ import annotations
import argparse, json, os, sys, time
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v1 import SceneWindowDataset, list_pt_paths
from src.train_motion_prediction_v3 import MAP_K, MotionPredictorV3, WindowSampleCollateV3, load_compatible_state
from src.train_motion_prediction_v6 import MotionPredictorV6
from src.train_motion_prediction_v7 import MotionPredictorV7, WindowSampleCollateV7
from src.losses.awta_loss import AdaptiveWTALoss
from train_motion_prediction_v2_awta import configure_runtime, make_loader, move_samples
from train_motion_prediction_v3 import evaluate_validation_v3
from train_motion_prediction_v7 import evaluate_validation_v7


def build_model(arch: str, hidden: int):
    if arch == "v3":
        return MotionPredictorV3(hidden=hidden, modes=6)
    if arch == "v6":
        return MotionPredictorV6(hidden=hidden, modes=6)
    if arch == "v7":
        return MotionPredictorV7(hidden=hidden, modes=6)
    raise ValueError(arch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["v3", "v6", "v7"], required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default=r"E:\motion_planning\data\processed\prediction_pt_85k")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--batch-scenes", type=int, default=32)
    parser.add_argument("--max-targets", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()

    configure_runtime(args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    val_paths = list_pt_paths(args.data_root, "val", 0)
    probe = torch.load(val_paths[0], map_location="cpu", weights_only=False)
    n_windows = max(1, len(probe.get("windows", [])))
    val_ds = SceneWindowDataset(args.data_root, "val", paths=val_paths, n_windows=n_windows)
    pin = device.type == "cuda"
    if args.arch == "v7":
        collate = WindowSampleCollateV7(args.max_targets, 16, 4, MAP_K, False)
        batch = 16 if args.max_targets <= 0 else args.batch_scenes
    else:
        collate = WindowSampleCollateV3(args.max_targets, 16, 4, MAP_K, False)
        batch = args.batch_scenes
    val_loader = make_loader(val_ds, batch, args.workers, args.prefetch, False, collate, pin)

    model = build_model(args.arch, args.hidden).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    load_compatible_state(model, ckpt["model_state"])
    model.eval()
    criterion = AdaptiveWTALoss(modes=6).to(device)

    t0 = time.time()
    if args.arch == "v7":
        res = evaluate_validation_v7(model, val_loader, criterion, device, epoch=19, total_epochs=20, amp_dtype=amp_dtype)
    else:
        res = evaluate_validation_v3(model, val_loader, criterion, device, epoch=19, total_epochs=20, amp_dtype=amp_dtype)
    res["eval_sec"] = time.time() - t0
    res["arch"] = args.arch
    res["ckpt"] = args.ckpt
    res["max_targets"] = args.max_targets
    print(json.dumps(res, indent=2), flush=True)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
