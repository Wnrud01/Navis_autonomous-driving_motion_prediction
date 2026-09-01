#!/usr/bin/env python3
"""Motion Prediction Model V2 Training Pipeline with MF-aWTA (Adaptive Winner-Takes-All Loss).

Optimized for 2026 Autonomous Driving AI Challenge (Error Score minimization).
Throughput-focused: DataLoader workers, worker-side sample collate, bf16 AMP,
fused AdamW, and deferred GPU syncs.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v1 import (
    MotionPredictor,
    SceneWindowDataset,
    WindowSampleCollate,
    list_pt_paths,
    expand_neighbor_encoder_state,
)
from src.losses.awta_loss import AdaptiveWTALoss


def configure_runtime(workers: int) -> None:
    if workers > 0:
        torch.set_num_threads(1)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def make_loader(dataset, batch_size: int, workers: int, prefetch: int, shuffle: bool, collate_fn, pin_memory: bool):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(2, prefetch)
    return DataLoader(**kwargs)


def query_gpu():
    if not torch.cuda.is_available():
        return {"util": -1, "mem_mb": 0, "power_w": -1}
    try:
        import subprocess
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip().split(",")
        return {
            "util": float(raw[0]),
            "mem_mb": float(raw[1]),
            "power_w": float(raw[2]),
        }
    except Exception:
        mem = torch.cuda.memory_allocated() / (1024 ** 2)
        return {"util": -1, "mem_mb": mem, "power_w": -1}


def move_samples(samples, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in samples.items()
    }


@torch.no_grad()
def evaluate_validation(model, val_loader, criterion, device, epoch, total_epochs, amp_dtype, max_steps=None):
    model.eval()
    ade6_sum = torch.zeros((), device=device)
    ade1_sum = torch.zeros((), device=device)
    fde6_sum = torch.zeros((), device=device)
    fde1_sum = torch.zeros((), device=device)
    loss_sum = torch.zeros((), device=device)
    n_targets = 0
    n_batches = 0

    for step, samples in enumerate(val_loader, start=1):
        if max_steps and step > max_steps:
            break
        if samples is None or samples["target_hist"].shape[0] == 0:
            continue
        samples = move_samples(samples, device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            pred, goals, logits = model(
                samples["target_hist"], samples["neighbors"], samples["signals"], samples["type_idx"]
            )
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
        last_valid_idx = (future_valid.sum(dim=-1).long() - 1).clamp(min=0, max=79)
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


def train_one_epoch(
    model, train_loader, optimizer, scaler, criterion, device, args, epoch, amp_dtype, use_scaler
):
    model.train()
    epoch_start = time.time()
    loss_sum = torch.zeros((), device=device)
    ade6_sum = torch.zeros((), device=device)
    ade1_sum = torch.zeros((), device=device)
    n_batches = 0
    n_targets = 0
    log_loss = torch.zeros((), device=device)
    log_ade6 = torch.zeros((), device=device)
    log_ade1 = torch.zeros((), device=device)
    log_n = 0
    tau_cur, top_m_cur = criterion.get_temperature_and_top_m(epoch - 1, args.epochs)
    last_log = epoch_start

    for step, samples in enumerate(train_loader, start=1):
        if args.max_train_steps and step > args.max_train_steps:
            break
        if samples is None or samples["target_hist"].shape[0] == 0:
            continue
        samples = move_samples(samples, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            pred, goals, logits = model(
                samples["target_hist"], samples["neighbors"], samples["signals"], samples["type_idx"]
            )
            loss, loss_dict = criterion(
                pred, goals, logits, samples["future"], samples["future_valid"],
                epoch=epoch - 1, total_epochs=args.epochs,
            )
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        loss_sum += loss.detach()
        ade6_sum += loss_dict["minade6_batch"]
        ade1_sum += loss_dict["minade1_batch"]
        log_loss += loss.detach()
        log_ade6 += loss_dict["minade6_batch"]
        log_ade1 += loss_dict["minade1_batch"]
        n_batches += 1
        log_n += 1
        n_targets += int(samples["target_hist"].shape[0])

        log_every = 50 if step <= 200 else args.log_every
        if step % log_every == 0 and log_n:
            now = time.time()
            dt = max(1e-6, now - last_log)
            steps_per_s = log_every / dt if step > log_every else step / max(1e-6, now - epoch_start)
            gpu = query_gpu()
            print(
                f" [Epoch {epoch:02d}/{args.epochs}] Step [{step:05d}/{len(train_loader)}] | "
                f"Loss: {float(log_loss / log_n):.4f} | minADE6: {float(log_ade6 / log_n):.3f}m | "
                f"minADE1: {float(log_ade1 / log_n):.3f}m | {steps_per_s:.1f} steps/s | "
                f"GPU {gpu['util']:.0f}% {gpu['mem_mb']:.0f}MB {gpu['power_w']:.0f}W | "
                f"tau: {tau_cur:.2f}, Top-{top_m_cur}",
                flush=True,
            )
            log_loss.zero_()
            log_ade6.zero_()
            log_ade1.zero_()
            log_n = 0
            last_log = now

    elapsed = time.time() - epoch_start
    return {
        "train_loss": float(loss_sum / max(n_batches, 1)),
        "train_minade6": float(ade6_sum / max(n_batches, 1)),
        "train_minade1": float(ade1_sum / max(n_batches, 1)),
        "epoch_sec": elapsed,
        "steps": n_batches,
        "targets": n_targets,
        "tau": tau_cur,
        "top_m": top_m_cur,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"C:\Users\andy0\Downloads\behavior_stack_planner\data\processed\prediction_pt")
    parser.add_argument("--out-dir", default=r"E:\motion_prediction\checkpoints\v2_awta")
    parser.add_argument("--resume-ckpt", default=r"E:\motion_prediction\checkpoints\best_minade6.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-scenes", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--max-targets", type=int, default=24)
    parser.add_argument("--neighbor-k", type=int, default=16)
    parser.add_argument("--signal-k", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-packs", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--tau-start", type=float, default=1.5)
    parser.add_argument("--tau-end", type=float, default=0.25)
    parser.add_argument("--tau-cls", type=float, default=0.5)
    parser.add_argument("--top-m-start", type=int, default=3)
    parser.add_argument("--top-m-end", type=int, default=1)
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
    print(" MOTION PREDICTION V2 — ADAPTIVE WINNER-TAKES-ALL (MF-aWTA) TRAINING")
    print(f" Data Root:       {args.data_root}")
    print(f" Output Dir:      {args.out_dir}")
    print(f" Resume Checkpoint: {args.resume_ckpt}")
    print(f" Epochs:          {args.epochs} (Batch: {args.batch_scenes}, LR: {args.lr})")
    print(f" Workers/Prefetch: {args.workers}/{args.prefetch}")
    print(f" AMP:             {args.amp} ({amp_dtype})")
    print(f" aWTA Annealing:  tau: {args.tau_start} -> {args.tau_end}, top-m: {args.top_m_start} -> {args.top_m_end}")
    print(f" Compute Device:  {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80, flush=True)

    train_paths = list_pt_paths(args.data_root, "train", args.max_packs)
    val_paths = list_pt_paths(args.data_root, "val", args.max_packs)
    probe = torch.load(train_paths[0], map_location="cpu", weights_only=False)
    n_windows = max(1, len(probe.get("windows", [])))
    print(f"-> Detected {n_windows} window(s) per pack", flush=True)

    train_ds = SceneWindowDataset(args.data_root, "train", paths=train_paths, n_windows=n_windows)
    val_ds = SceneWindowDataset(args.data_root, "val", paths=val_paths, n_windows=n_windows)
    pin = device.type == "cuda"
    train_loader = make_loader(
        train_ds, args.batch_scenes, args.workers, args.prefetch, True,
        WindowSampleCollate(args.max_targets, args.neighbor_k, args.signal_k, True), pin,
    )
    val_loader = make_loader(
        val_ds, args.batch_scenes, args.workers, args.prefetch, False,
        WindowSampleCollate(args.max_targets, args.neighbor_k, args.signal_k, False), pin,
    )
    print(
        f"-> Train Scenes: {len(train_ds.paths)} | Val Scenes: {len(val_ds.paths)} | "
        f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}",
        flush=True,
    )

    model = MotionPredictor(hidden=args.hidden, modes=6).to(device)
    criterion = AdaptiveWTALoss(
        modes=6,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_cls=args.tau_cls,
        top_m_start=args.top_m_start,
        top_m_end=args.top_m_end,
    ).to(device)

    start_epoch = 1
    best_error_score = float("inf")
    best_minade6 = float("inf")

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"-> Loading pre-trained weights from: {args.resume_ckpt}", flush=True)
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        state = expand_neighbor_encoder_state(ckpt["model_state"])
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Pre-trained model weights loaded successfully!", flush=True)
        if missing or unexpected:
            print(f"-> load_state_dict missing={list(missing)} unexpected={list(unexpected)}", flush=True)

    fused = (device.type == "cuda") and (not args.no_fused_adam)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=fused
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        fused = False
    print(f"-> Optimizer: AdamW fused={fused}", flush=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {**vars(args), "device": str(device), "amp_dtype": str(amp_dtype), "n_windows": n_windows,
             "train_packs": len(train_ds.paths), "val_packs": len(val_ds.paths), "fused_adam": fused},
            f, ensure_ascii=False, indent=2,
        )

    history_metrics = []
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                history_metrics = json.load(f)
        except Exception:
            history_metrics = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_res = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, args, epoch, amp_dtype, use_scaler
        )
        scheduler.step()
        val_res = evaluate_validation(
            model, val_loader, criterion, device, epoch=epoch - 1, total_epochs=args.epochs,
            amp_dtype=amp_dtype, max_steps=args.max_val_steps or None,
        )
        epoch_record = {
            "epoch": epoch,
            **train_res,
            **val_res,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_metrics.append(epoch_record)
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(history_metrics, f, ensure_ascii=False, indent=2)

        log_msg = (
            f"Epoch {epoch:02d}/{args.epochs} ({train_res['epoch_sec']:.1f}s, {train_res['steps']} steps) | "
            f"Train Loss: {train_res['train_loss']:.4f}, minADE6: {train_res['train_minade6']:.3f}m, "
            f"minADE1: {train_res['train_minade1']:.3f}m | "
            f"Val Loss: {val_res['val_loss']:.4f}, val_minADE6: {val_res['val_minade6']:.4f}m, "
            f"val_minADE1: {val_res['val_minade1']:.4f}m, Error Score: {val_res['val_error_score']:.4f}"
        )
        print(log_msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "metrics": epoch_record},
            os.path.join(args.out_dir, "last.pth"),
        )
        if val_res["val_error_score"] < best_error_score:
            best_error_score = val_res["val_error_score"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "metrics": epoch_record},
                os.path.join(args.out_dir, "best_error_score.pth"),
            )
            print(f"[NEW BEST Error Score]: {best_error_score:.4f} saved!", flush=True)
        if val_res["val_minade6"] < best_minade6:
            best_minade6 = val_res["val_minade6"]
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "metrics": epoch_record},
                os.path.join(args.out_dir, "best_minade6.pth"),
            )
            print(f"[NEW BEST minADE6]: {best_minade6:.4f}m saved!", flush=True)

        if args.max_train_steps:
            print("-> max-train-steps reached; stopping after this epoch.", flush=True)
            break

    print(
        f"\nV2 aWTA Training Completed! Best Error Score: {best_error_score:.4f}, "
        f"Best minADE6: {best_minade6:.4f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
