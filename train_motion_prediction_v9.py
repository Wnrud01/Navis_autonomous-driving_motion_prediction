#!/usr/bin/env python3
"""Train V9 mixed-interaction + basic driving from scratch."""
from __future__ import annotations
import argparse, json, os, sys, time
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v1 import SceneWindowDataset, list_pt_paths
from src.train_motion_prediction_v3 import MAP_K, load_compatible_state
from src.train_motion_prediction_v8 import POLY_K
from src.train_motion_prediction_v9 import MotionPredictorV9, WindowSampleCollateV9
from src.losses.awta_loss import AdaptiveWTALoss
from train_motion_prediction_v2_awta import configure_runtime, make_loader, query_gpu
from train_motion_prediction_v3 import evaluate_validation_v3, train_one_epoch_v3
import train_motion_prediction_v3 as v3_train


def model_forward(model, samples):
    return model(
        samples["target_hist"],
        samples["neighbors"],
        samples["neighbor_valid"],
        samples["map_feat"],
        samples["map_valid"],
        samples["signals"],
        samples["type_idx"],
        kin_feat=samples.get("kin_feat"),
        lane_sig=samples.get("lane_sig"),
        lane_sig_valid=samples.get("lane_sig_valid"),
        interact=samples.get("interact"),
        interact_valid=samples.get("interact_valid"),
    )


v3_train.model_forward = model_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"E:\motion_planning\data\processed\prediction_pt_85k")
    parser.add_argument("--out-dir", default=r"E:\motion_prediction\checkpoints\v9_interact")
    parser.add_argument("--resume-ckpt", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-scenes", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--max-targets", type=int, default=24)
    parser.add_argument("--neighbor-k", type=int, default=16)
    parser.add_argument("--signal-k", type=int, default=4)
    parser.add_argument("--map-k", type=int, default=MAP_K)
    parser.add_argument("--poly-k", type=int, default=POLY_K)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-packs", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--tau-start", type=float, default=1.5)
    parser.add_argument("--tau-end", type=float, default=0.25)
    parser.add_argument("--tau-cls", type=float, default=0.5)
    parser.add_argument("--top-m-start", type=int, default=3)
    parser.add_argument("--top-m-end", type=int, default=1)
    parser.add_argument("--time-weight-end", type=float, default=3.0)
    parser.add_argument("--weight-fde", type=float, default=0.4)
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
    print(" MOTION PREDICTION V9 — BASIC DRIVING + MIXED INTERACTION (from scratch)")
    print(" Tokens: my-lane+light+lead | signal-11step | adj cut-in | stop/xwalk | ego")
    print(" Height excluded. All layers trained together (decoder + mix). aWTA tau 1.5->0.25")
    print(f" Resume: {args.resume_ckpt or '(none)'}")
    print(f" Epochs {args.epochs}  batch {args.batch_scenes}  lr {args.lr}")
    print("=" * 80, flush=True)

    train_paths = list_pt_paths(args.data_root, "train", args.max_packs)
    val_paths = list_pt_paths(args.data_root, "val", args.max_packs)
    probe = torch.load(train_paths[0], map_location="cpu", weights_only=False)
    n_windows = max(1, len(probe.get("windows", [])))
    train_ds = SceneWindowDataset(args.data_root, "train", paths=train_paths, n_windows=n_windows)
    val_ds = SceneWindowDataset(args.data_root, "val", paths=val_paths, n_windows=n_windows)
    pin = device.type == "cuda"
    train_loader = make_loader(
        train_ds, args.batch_scenes, args.workers, args.prefetch, True,
        WindowSampleCollateV9(args.max_targets, args.neighbor_k, args.signal_k, args.map_k, args.poly_k, True), pin,
    )
    val_loader = make_loader(
        val_ds, args.batch_scenes, args.workers, args.prefetch, False,
        WindowSampleCollateV9(args.max_targets, args.neighbor_k, args.signal_k, args.map_k, args.poly_k, False), pin,
    )
    print(f"-> Train {len(train_ds.paths)}  Val {len(val_ds.paths)}", flush=True)

    model = MotionPredictorV9(hidden=args.hidden, modes=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"-> params {n_params/1e6:.3f}M", flush=True)
    criterion = AdaptiveWTALoss(
        modes=6, tau_start=args.tau_start, tau_end=args.tau_end, tau_cls=args.tau_cls,
        top_m_start=args.top_m_start, top_m_end=args.top_m_end,
        time_weight_end=args.time_weight_end, weight_fde=args.weight_fde,
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
        print("-> random init: decoder + interaction trained together", flush=True)

    fused = (device.type == "cuda") and (not args.no_fused_adam)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=fused)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        fused = False
    print(f"-> AdamW lr={args.lr} fused={fused}  all layers", flush=True)
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
        val_res = evaluate_validation_v3(
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
        if val_res["val_error_score"] < best_error_score:
            best_error_score = val_res["val_error_score"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": rec}, os.path.join(args.out_dir, "best_error_score.pth"))
            print(f"[NEW BEST Error] {best_error_score:.4f}", flush=True)
        if val_res["val_minade6"] < best_minade6:
            best_minade6 = val_res["val_minade6"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": rec}, os.path.join(args.out_dir, "best_minade6.pth"))
            print(f"[NEW BEST minADE6] {best_minade6:.4f}", flush=True)
        if args.max_train_steps:
            break
    print(f"V9 done. Best Error {best_error_score:.4f} ADE6 {best_minade6:.4f}", flush=True)


if __name__ == "__main__":
    main()
