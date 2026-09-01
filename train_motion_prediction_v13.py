#!/usr/bin/env python3
"""Train Motion Prediction V13: Polyline Map Encoder + Mode Query Cross-Attention Decoder + Multi-Horizon aWTA."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v1 import SceneWindowDataset, list_pt_paths
from src.train_motion_prediction_v3 import load_compatible_state
from src.cached_collate_v13 import CachedWindowCollateV13, try_cached_loader_v13
from src.train_motion_prediction_v13_collate import WindowSampleCollateV13
from src.train_motion_prediction_v13 import MotionPredictorV13
from src.losses.awta_loss_v13 import AdaptiveWTALossV13
from train_motion_prediction_v2_awta import configure_runtime, make_loader, move_samples
from train_motion_prediction_v10 import _list_or_fallback, model_forward


def evaluate_validation_v13(model, val_loader, criterion, device, epoch, total_epochs, amp_dtype, max_steps=None):
    model.eval()
    ade6_sum = torch.zeros((), device=device)
    ade1_sum = torch.zeros((), device=device)
    fde6_sum = torch.zeros((), device=device)
    fde1_sum = torch.zeros((), device=device)
    loss_sum = torch.zeros((), device=device)
    n_targets = 0
    n_batches = 0
    with torch.no_grad():
        for step, samples in enumerate(val_loader, start=1):
            if max_steps and step > max_steps:
                break
            if samples is None or samples["target_hist"].shape[0] == 0:
                continue
            samples = move_samples(samples, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                pred, goals, logits = model_forward(model, samples)
                loss, _ = criterion(
                    pred, goals, logits, samples["future"], samples["future_valid"],
                    epoch=epoch, total_epochs=total_epochs,
                )
            future = samples["future"]
            future_valid = samples["future_valid"]
            diff = pred - future[:, None, :, :]
            disp = torch.linalg.vector_norm(diff, dim=-1)
            mask = future_valid[:, None, :].float()
            denom = mask.sum(dim=-1).clamp_min(1.0)
            ade_modes = (disp * mask).sum(dim=-1) / denom

            t = future.shape[1]
            t_ix = torch.arange(t, device=device).view(1, t).expand(future.shape[0], t)
            last_valid_idx = torch.where(future_valid, t_ix, t_ix.new_zeros(())).max(dim=-1).values
            fde_modes = torch.gather(disp, 2, last_valid_idx[:, None, None].expand(-1, 6, 1)).squeeze(2)

            b_idx = torch.arange(pred.shape[0], device=device)
            best_mode_6 = ade_modes.argmin(dim=1)
            top1_mode = logits.argmax(dim=1)
            n = pred.shape[0]
            ade6_sum += ade_modes[b_idx, best_mode_6].sum()
            ade1_sum += ade_modes[b_idx, top1_mode].sum()
            fde6_sum += fde_modes[b_idx, best_mode_6].sum()
            fde1_sum += fde_modes[b_idx, top1_mode].sum()
            loss_sum += loss.detach()
            n_targets += n
            n_batches += 1
    if n_targets == 0:
        return {
            "val_loss": 0.0, "val_minade6": 0.0, "val_minade1": 0.0,
            "val_minfde6": 0.0, "val_minfde1": 0.0, "val_error_score": 0.0,
        }
    mean_ade6 = float(ade6_sum / n_targets)
    mean_ade1 = float(ade1_sum / n_targets)
    return {
        "val_loss": float(loss_sum / max(n_batches, 1)),
        "val_minade6": mean_ade6,
        "val_minade1": mean_ade1,
        "val_minfde6": float(fde6_sum / n_targets),
        "val_minfde1": float(fde1_sum / n_targets),
        "val_error_score": 0.5 * (mean_ade1 + mean_ade6),
    }


def train_one_epoch_v13(model, train_loader, optimizer, scaler, criterion, device, args, epoch, amp_dtype, use_scaler):
    model.train()
    epoch_start = time.time()
    loss_sum = torch.zeros((), device=device)
    ade6_sum = torch.zeros((), device=device)
    ade1_sum = torch.zeros((), device=device)
    n_targets = 0
    n_batches = 0
    cur_tau = 1.0
    cur_top_m = 4

    for step, samples in enumerate(train_loader, start=1):
        if args.max_train_steps and step > args.max_train_steps:
            break
        if samples is None or samples["target_hist"].shape[0] == 0:
            continue
        samples = move_samples(samples, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            pred, goals, logits = model_forward(model, samples)
            loss, metrics = criterion(
                pred, goals, logits, samples["future"], samples["future_valid"],
                epoch=epoch - 1, total_epochs=args.epochs,
            )

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        b = samples["target_hist"].shape[0]
        loss_sum += loss.detach() * b
        ade6_sum += metrics["minade6_batch"] * b
        ade1_sum += metrics["minade1_batch"] * b
        n_targets += b
        n_batches += 1
        cur_tau = metrics["tau"]
        cur_top_m = int(metrics["top_m"])

        if step % args.log_every == 0 or step == len(train_loader):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] Step [{step:05d}/{len(train_loader):05d}] "
                f"loss {loss.item():.4f} ade6 {metrics['minade6_batch']:.4f} ade1 {metrics['minade1_batch']:.4f} "
                f"tau {cur_tau:.3f} top_m {cur_top_m} lr {lr:.2e}",
                flush=True,
            )

    return {
        "train_loss": float(loss_sum / max(n_targets, 1)),
        "train_minade6": float(ade6_sum / max(n_targets, 1)),
        "train_minade1": float(ade1_sum / max(n_targets, 1)),
        "epoch_sec": time.time() - epoch_start,
        "steps": n_batches,
        "targets": n_targets,
        "tau": cur_tau,
        "top_m": cur_top_m,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2")
    parser.add_argument("--cache-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2_cache_v13")
    parser.add_argument("--out-dir", default=r"E:\motion_prediction\checkpoints\v13_ade6")
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
    parser.add_argument("--tau-mid", type=float, default=0.5)
    parser.add_argument("--tau-end", type=float, default=0.1)
    parser.add_argument("--tau-cls", type=float, default=0.5)
    parser.add_argument("--top-m-start", type=int, default=4)
    parser.add_argument("--top-m-mid", type=int, default=2)
    parser.add_argument("--top-m-end", type=int, default=1)
    parser.add_argument("--weight-reg", type=float, default=1.0)
    parser.add_argument("--weight-goal", type=float, default=0.15)
    parser.add_argument("--weight-div", type=float, default=0.08)
    parser.add_argument("--horizon-weight", type=float, default=0.5)
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
    metrics_file = os.path.join(args.out_dir, "metrics.json")

    print("=" * 80)
    print(" MOTION PREDICTION V13 — POLYLINE MAP + MODE QUERY DECODER")
    print(" Architecture: 16x20x8 Polyline PointNet + Mode Query Cross-Attn + 3-Stage aWTA")
    print(f" Resume: {args.resume_ckpt or '(none / random init)'}")
    print(f" Epochs {args.epochs}  batch {args.batch_scenes}  lr {args.lr}  AMP {args.amp}")
    print("=" * 80, flush=True)

    train_ds = try_cached_loader_v13(args.cache_root, "train", args.max_packs)
    val_ds = try_cached_loader_v13(args.cache_root, "val", args.max_packs)
    pin = device.type == "cuda"

    if train_ds is not None and val_ds is not None:
        print(f"-> using precomputed collate cache: {args.cache_root}", flush=True)
        train_collate = CachedWindowCollateV13(args.max_targets, True)
        val_collate = CachedWindowCollateV13(args.max_targets, False)
    else:
        print(f"-> cache missing at {args.cache_root}, using live collate from {args.data_root}", flush=True)
        train_paths = list_pt_paths(args.data_root, "train", args.max_packs)
        val_paths = _list_or_fallback(args.data_root, "val", args.max_packs, fallback=train_paths)
        probe = torch.load(train_paths[0], map_location="cpu", weights_only=False)
        n_windows = max(1, len(probe.get("windows", [])))
        train_ds = SceneWindowDataset(args.data_root, "train", paths=train_paths, n_windows=n_windows)
        val_ds = SceneWindowDataset(args.data_root, "val", paths=val_paths, n_windows=n_windows)
        train_collate = WindowSampleCollateV13(args.max_targets, True)
        val_collate = WindowSampleCollateV13(args.max_targets, False)

    train_loader = make_loader(train_ds, args.batch_scenes, args.workers, args.prefetch, True, train_collate, pin)
    val_loader = make_loader(val_ds, args.batch_scenes, args.workers, args.prefetch, False, val_collate, pin)
    print(f"-> Train samples {len(train_ds)}  Val samples {len(val_ds)}", flush=True)

    model = MotionPredictorV13(hidden=args.hidden, modes=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"-> V13 Model parameters: {n_params/1e6:.3f}M", flush=True)

    criterion = AdaptiveWTALossV13(
        modes=6,
        tau_start=args.tau_start, tau_mid=args.tau_mid, tau_end=args.tau_end,
        tau_cls=args.tau_cls,
        top_m_start=args.top_m_start, top_m_mid=args.top_m_mid, top_m_end=args.top_m_end,
        weight_reg=args.weight_reg, weight_goal=args.weight_goal,
        weight_div=args.weight_div, horizon_weight_scale=args.horizon_weight,
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

    fused = (device.type == "cuda") and (not args.no_fused_adam)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=fused)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        fused = False
    print(f"-> Optimizer AdamW lr={args.lr} fused={fused}", flush=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "params": n_params, "fused": fused}, f, ensure_ascii=False, indent=2)

    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()

    for epoch in range(start_epoch, args.epochs + 1):
        train_res = train_one_epoch_v13(
            model, train_loader, optimizer, scaler, criterion, device, args, epoch, amp_dtype, use_scaler,
        )
        scheduler.step()
        val_res = evaluate_validation_v13(
            model, val_loader, criterion, device, epoch=epoch - 1, total_epochs=args.epochs,
            amp_dtype=amp_dtype, max_steps=args.max_val_steps or None,
        )
        rec = {"epoch": epoch, **train_res, **val_res, "lr": optimizer.param_groups[0]["lr"]}
        history.append(rec)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        is_best_err = val_res["val_error_score"] < best_error_score
        is_best_ade6 = val_res["val_minade6"] < best_minade6
        if is_best_err:
            best_error_score = val_res["val_error_score"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": rec}, os.path.join(args.out_dir, "best_error_score.pth"))
        if is_best_ade6:
            best_minade6 = val_res["val_minade6"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": rec}, os.path.join(args.out_dir, "best_minade6.pth"))
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "metrics": rec}, os.path.join(args.out_dir, "last.pth"))

        flags = []
        if is_best_err:
            flags.append("BEST_ERR")
        if is_best_ade6:
            flags.append("BEST_ADE6")
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({train_res['epoch_sec']:.1f}s) | "
            f"Train ADE6 {train_res['train_minade6']:.3f} ADE1 {train_res['train_minade1']:.3f} | "
            f"Val ADE6 {val_res['val_minade6']:.4f} ADE1 {val_res['val_minade1']:.4f} "
            f"FDE6 {val_res['val_minfde6']:.4f} Error {val_res['val_error_score']:.4f}{flag_str}",
            flush=True,
        )


if __name__ == "__main__":
    main()
