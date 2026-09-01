#!/usr/bin/env python3
"""Train V12: ordered token stages + per-type heads, ADE6-first loss (same as V11)."""
from __future__ import annotations
import argparse, json, os, sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v1 import SceneWindowDataset, list_pt_paths
from src.train_motion_prediction_v3 import load_compatible_state
from src.cached_collate import CachedWindowCollate, try_cached_loader
from src.train_motion_prediction_v10 import WindowSampleCollateV10
from src.train_motion_prediction_v12 import MotionPredictorV12
from src.losses.awta_loss import AdaptiveWTALoss
from train_motion_prediction_v2_awta import configure_runtime, make_loader
from train_motion_prediction_v3 import train_one_epoch_v3
from train_motion_prediction_v10 import _list_or_fallback, evaluate_validation_v10, model_forward
import train_motion_prediction_v3 as v3_train

v3_train.model_forward = model_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2")
    parser.add_argument("--cache-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2_cache")
    parser.add_argument("--out-dir", default=r"E:\motion_prediction\checkpoints\v12_ade6")
    parser.add_argument("--resume-ckpt", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-scenes", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--max-targets", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-packs", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--tau-start", type=float, default=1.5)
    parser.add_argument("--tau-end", type=float, default=0.4)
    parser.add_argument("--tau-cls", type=float, default=0.5)
    parser.add_argument("--top-m-start", type=int, default=4)
    parser.add_argument("--top-m-end", type=int, default=2)
    parser.add_argument("--time-weight-end", type=float, default=1.0)
    parser.add_argument("--weight-fde", type=float, default=0.0)
    parser.add_argument("--weight-goal", type=float, default=0.15)
    parser.add_argument("--weight-div", type=float, default=0.08)
    parser.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--no-fused-adam", action="store_true")
    args = parser.parse_args()

    configure_runtime(args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        use_scaler = False
    elif args.amp != "fp32" and device.type == "cuda":
        amp_dtype = torch.float16
        use_scaler = True
    else:
        amp_dtype = torch.float32
        use_scaler = False

    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, "training.log")
    metrics_file = os.path.join(args.out_dir, "metrics.json")

    print("=" * 80)
    print(" MOTION PREDICTION V12 — ORDERED TOKENS + PER-TYPE HEADS")
    print(" Stages: speed → lane → lead → signal → roadside → adjacent")
    print(" Separate vehicle / pedestrian / cyclist trajectory heads")
    print(f" Resume: {args.resume_ckpt or '(none / random init)'}")
    print(f" Epochs {args.epochs}  batch {args.batch_scenes}  lr {args.lr}")
    print("=" * 80, flush=True)

    train_ds = try_cached_loader(args.cache_root, "train", args.max_packs)
    val_ds = try_cached_loader(args.cache_root, "val", args.max_packs)
    pin = device.type == "cuda"
    if train_ds is not None and val_ds is not None:
        print(f"-> using collate cache {args.cache_root}", flush=True)
        train_collate = CachedWindowCollate(args.max_targets, True)
        val_collate = CachedWindowCollate(args.max_targets, False)
    else:
        print("-> cache missing, live collate (slow CPU)", flush=True)
        train_paths = list_pt_paths(args.data_root, "train", args.max_packs)
        val_paths = _list_or_fallback(args.data_root, "val", args.max_packs, fallback=train_paths)
        probe = torch.load(train_paths[0], map_location="cpu", weights_only=False)
        n_windows = max(1, len(probe.get("windows", [])))
        train_ds = SceneWindowDataset(args.data_root, "train", paths=train_paths, n_windows=n_windows)
        val_ds = SceneWindowDataset(args.data_root, "val", paths=val_paths, n_windows=n_windows)
        train_collate = WindowSampleCollateV10(args.max_targets, True)
        val_collate = WindowSampleCollateV10(args.max_targets, False)
    train_loader = make_loader(
        train_ds, args.batch_scenes, args.workers, args.prefetch, True, train_collate, pin,
    )
    val_loader = make_loader(
        val_ds, args.batch_scenes, args.workers, args.prefetch, False, val_collate, pin,
    )
    print(f"-> Train {len(train_ds)}  Val {len(val_ds)}", flush=True)

    model = MotionPredictorV12(hidden=args.hidden, modes=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"-> params {n_params/1e6:.3f}M", flush=True)
    criterion = AdaptiveWTALoss(
        modes=6, tau_start=args.tau_start, tau_end=args.tau_end, tau_cls=args.tau_cls,
        top_m_start=args.top_m_start, top_m_end=args.top_m_end,
        time_weight_end=args.time_weight_end, weight_fde=args.weight_fde,
        weight_goal=args.weight_goal, weight_div=args.weight_div,
    ).to(device)

    best_error_score = float("inf")
    best_minade6 = float("inf")
    start_epoch = 1
    history = []
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        missing, _ = load_compatible_state(model, ckpt["model_state"])
        loaded = sum(1 for k in model.state_dict() if k not in missing)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"-> loaded {loaded}/{len(model.state_dict())}  resume epoch {start_epoch}", flush=True)
        if os.path.isfile(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            if history:
                best_error_score = min(h.get("val_error_score", 1e9) for h in history)
                best_minade6 = min(h.get("val_minade6", 1e9) for h in history)
    else:
        print("-> random init: V12 (no V11 load; fusion+heads changed)", flush=True)

    fused = (device.type == "cuda") and (not args.no_fused_adam)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=fused)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        fused = False
    print(f"-> AdamW lr={args.lr} fused={fused}", flush=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "params": n_params, "fused": fused}, f, ensure_ascii=False, indent=2)

    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()
    for epoch in range(start_epoch, args.epochs + 1):
        train_res = train_one_epoch_v3(
            model, train_loader, optimizer, scaler, criterion, device, args, epoch, amp_dtype, use_scaler
        )
        scheduler.step()
        val_res = evaluate_validation_v10(
            model, val_loader, criterion, device, epoch=epoch - 1, total_epochs=args.epochs,
            amp_dtype=amp_dtype, max_steps=args.max_val_steps or None,
        )
        rec = {"epoch": epoch, **train_res, **val_res, "lr": optimizer.param_groups[0]["lr"]}
        history.append(rec)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        msg = (
            f"Epoch {epoch:02d}/{args.epochs} ({train_res['epoch_sec']:.1f}s) | "
            f"Train ADE6 {train_res['train_minade6']:.3f} ADE1 {train_res['train_minade1']:.3f} | "
            f"Val ADE6 {val_res['val_minade6']:.4f} ADE1 {val_res['val_minade1']:.4f} "
            f"FDE6 {val_res['val_minfde6']:.4f} Error {val_res['val_error_score']:.4f}"
        )
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": rec}, os.path.join(args.out_dir, "last.pth"))
        if val_res["val_minade6"] < best_minade6:
            best_minade6 = val_res["val_minade6"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": rec}, os.path.join(args.out_dir, "best_minade6.pth"))
            print(f"[NEW BEST minADE6] {best_minade6:.4f}", flush=True)
        if val_res["val_error_score"] < best_error_score:
            best_error_score = val_res["val_error_score"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": rec}, os.path.join(args.out_dir, "best_error_score.pth"))
            print(f"[NEW BEST Error] {best_error_score:.4f}", flush=True)
        if args.max_train_steps:
            break
    print(f"V12 done. Best minADE6 {best_minade6:.4f} Error {best_error_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
