#!/usr/bin/env python3
"""Fine-Tune Motion Prediction V18: High-Speed Velocity Weighted & Kinematic Refinement.

Warm-starts from V17 SWA Checkpoint (0.8035 baseline) and fine-tunes for 15 epochs:
- Equal-weighted 80-step ADE (matching official metric).
- Velocity-proportional sample weighting (1 + v_cur / 10.0) focusing on high-speed cruise regime.
- Type-aware soft acceleration hinge penalty on vehicles.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v1 import SceneWindowDataset, list_pt_paths
from src.train_motion_prediction_v3 import load_compatible_state
from src.cached_collate_v13 import CachedWindowCollateV13, try_cached_loader_v13
from src.train_motion_prediction_v13_collate import WindowSampleCollateV13
from src.train_motion_prediction_v17 import MotionPredictorV17
from src.losses.awta_loss_v18 import AdaptiveWTALossV18
from train_motion_prediction_v2_awta import configure_runtime, make_loader, move_samples
from train_motion_prediction_v10 import _list_or_fallback


def model_forward_v18(model, samples):
    return model(
        samples["target_hist"],
        samples["neighbors"],
        samples["neighbor_valid"],
        samples["map_feat"],
        samples["map_valid"],
        samples["signals"],
        samples["type_idx"],
        agent_tok=samples.get("agent_tok"),
        lane_tok=samples.get("lane_tok"),
        lane_valid=samples.get("lane_valid"),
        map_tok=samples.get("map_tok"),
        inter_tok=samples.get("inter_tok"),
    )


def evaluate_validation_v18(model, val_loader, criterion, device, epoch, total_epochs, amp_dtype, max_steps=None):
    model.eval()
    ade6_s1_sum = torch.zeros((), device=device)
    ade6_s2_sum = torch.zeros((), device=device)
    ade1_s2_sum = torch.zeros((), device=device)
    fde6_s2_sum = torch.zeros((), device=device)
    fde1_s2_sum = torch.zeros((), device=device)
    loss_sum = torch.zeros((), device=device)
    pick_correct_sum = torch.zeros((), device=device)
    n_targets = 0
    n_batches = 0

    with torch.no_grad():
        for step, samples in enumerate(val_loader, start=1):
            if max_steps and step > max_steps:
                break
            if samples is None or samples["target_hist"].shape[0] == 0:
                continue
            samples = move_samples(samples, device)
            cur_speed = torch.linalg.vector_norm(samples["target_hist"][:, -1, 3:5], dim=-1)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                pred_2, goals_2, logits_2, pred_1, goals_1 = model_forward_v18(model, samples)
                loss, metrics = criterion(
                    pred_2, goals_2, logits_2, pred_1, goals_1,
                    samples["future"], samples["future_valid"],
                    samples["type_idx"], cur_speed,
                    epoch=epoch, total_epochs=total_epochs,
                )

            future = samples["future"]
            future_valid = samples["future_valid"]
            mask = future_valid[:, None, :].float()
            denom = mask.sum(dim=-1).clamp_min(1.0)
            b = future.shape[0]
            b_idx = torch.arange(b, device=device)

            # Stage 1 evaluation
            diff_1 = pred_1 - future[:, None, :, :]
            disp_1 = torch.linalg.vector_norm(diff_1, dim=-1)
            ade_modes_1 = (disp_1 * mask).sum(dim=-1) / denom
            best_mode_1 = ade_modes_1.argmin(dim=1)
            ade6_s1_sum += ade_modes_1[b_idx, best_mode_1].sum()

            # Stage 2 evaluation
            diff_2 = pred_2 - future[:, None, :, :]
            disp_2 = torch.linalg.vector_norm(diff_2, dim=-1)
            ade_modes_2 = (disp_2 * mask).sum(dim=-1) / denom

            t = future.shape[1]
            t_ix = torch.arange(t, device=device).view(1, t).expand(b, t)
            last_valid_idx = torch.where(future_valid, t_ix, t_ix.new_zeros(())).max(dim=-1).values
            fde_modes_2 = torch.gather(disp_2, 2, last_valid_idx[:, None, None].expand(-1, 6, 1)).squeeze(2)

            best_mode_2 = ade_modes_2.argmin(dim=1)
            top1_mode = logits_2.argmax(dim=1)

            ade6_s2_sum += ade_modes_2[b_idx, best_mode_2].sum()
            ade1_s2_sum += ade_modes_2[b_idx, top1_mode].sum()
            fde6_s2_sum += fde_modes_2[b_idx, best_mode_2].sum()
            fde1_s2_sum += fde_modes_2[b_idx, top1_mode].sum()
            pick_correct_sum += (top1_mode == best_mode_2).float().sum()
            loss_sum += loss.detach()
            n_targets += b
            n_batches += 1

    if n_targets == 0:
        return {
            "val_loss": 0.0, "val_minade6_s1": 0.0, "val_minade6": 0.0,
            "val_minade1": 0.0, "val_minfde6": 0.0, "val_minfde1": 0.0,
            "val_error_score": 0.0, "val_pick_acc": 0.0,
        }
    mean_ade6_s1 = float(ade6_s1_sum / n_targets)
    mean_ade6_s2 = float(ade6_s2_sum / n_targets)
    mean_ade1_s2 = float(ade1_s2_sum / n_targets)
    return {
        "val_loss": float(loss_sum / max(n_batches, 1)),
        "val_minade6_s1": mean_ade6_s1,
        "val_minade6": mean_ade6_s2,
        "val_minade1": mean_ade1_s2,
        "val_minfde6": float(fde6_s2_sum / n_targets),
        "val_minfde1": float(fde1_s2_sum / n_targets),
        "val_pick_acc": float(pick_correct_sum / n_targets),
        "val_error_score": 0.5 * (mean_ade1_s2 + mean_ade6_s2),
    }


def train_one_epoch_v18(model, train_loader, optimizer, scaler, criterion, device, args, epoch, amp_dtype, use_scaler):
    model.train()
    epoch_start = time.time()
    loss_sum = torch.zeros((), device=device)
    ade6_s1_sum = torch.zeros((), device=device)
    ade6_s2_sum = torch.zeros((), device=device)
    ade1_s2_sum = torch.zeros((), device=device)
    pick_sum = torch.zeros((), device=device)
    n_targets = 0
    n_batches = 0
    cur_tau = 0.5
    cur_top_m = 2

    for step, samples in enumerate(train_loader, start=1):
        if samples is None or samples["target_hist"].shape[0] == 0:
            continue
        samples = move_samples(samples, device)
        cur_speed = torch.linalg.vector_norm(samples["target_hist"][:, -1, 3:5], dim=-1)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            pred_2, goals_2, logits_2, pred_1, goals_1 = model_forward_v18(model, samples)
            loss, metrics = criterion(
                pred_2, goals_2, logits_2, pred_1, goals_1,
                samples["future"], samples["future_valid"],
                samples["type_idx"], cur_speed,
                epoch=epoch, total_epochs=args.epochs,
            )

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        b = samples["future"].shape[0]
        n_targets += b
        n_batches += 1
        loss_sum += loss.detach()
        ade6_s1_sum += metrics["s1_ade6"] * b
        ade6_s2_sum += metrics["s2_ade6"] * b
        ade1_s2_sum += metrics["s2_ade1"] * b
        pick_sum += metrics["pick_acc"] * b
        cur_tau = metrics["tau"]
        cur_top_m = metrics["top_m"]

        if step % args.log_every == 0 or step == len(train_loader):
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] Step [{step:05d}/{len(train_loader):05d}] "
                f"loss {metrics['loss']:.4f} s1_ade6 {metrics['s1_ade6']:.4f} s2_ade6 {metrics['s2_ade6']:.4f} "
                f"s2_ade1 {metrics['s2_ade1']:.4f} pick {metrics['pick_acc']:.3f} "
                f"tau {cur_tau:.3f} top_m {cur_top_m} lr {cur_lr:.2e}",
                flush=True,
            )

    epoch_sec = max(time.time() - epoch_start, 1e-4)
    return {
        "train_loss": float(loss_sum / max(n_batches, 1)),
        "train_minade6_s1": float(ade6_s1_sum / max(n_targets, 1)),
        "train_minade6": float(ade6_s2_sum / max(n_targets, 1)),
        "train_minade1": float(ade1_s2_sum / max(n_targets, 1)),
        "train_pick_acc": float(pick_sum / max(n_targets, 1)),
        "epoch_sec": epoch_sec,
        "steps": n_batches,
        "targets": n_targets,
        "tau": cur_tau,
        "top_m": cur_top_m,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Motion Prediction V18")
    parser.add_argument("--data-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2")
    parser.add_argument("--cache-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2_cache_v13")
    parser.add_argument("--out-dir", default=r"E:\motion_prediction\checkpoints\v18_speed_finetune")
    parser.add_argument("--init-ckpt", default=r"E:\motion_prediction\checkpoints\v17_xlarge\swa_model.pth")
    parser.add_argument("--batch-scenes", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--amp", default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configure_runtime(args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else torch.float32)
    use_scaler = (args.amp == "fp16" and device.type == "cuda")

    print(f"Initializing MotionPredictorV18 with 45.2M Architecture (hidden={args.hidden}, nhead={args.nhead})...")
    model = MotionPredictorV17(hidden=args.hidden, modes=6, nhead=args.nhead, dropout=args.dropout).to(device)

    if os.path.exists(args.init_ckpt):
        print(f"-> Warm-starting weights from: {args.init_ckpt}")
        init_ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        load_compatible_state(model, init_ckpt.get("model_state", init_ckpt))
    else:
        print(f"[WARNING] Init ckpt not found at {args.init_ckpt}, checking fallback...")
        fallback = r"E:\motion_prediction\checkpoints\v17_xlarge\best_error_score.pth"
        if os.path.exists(fallback):
            init_ckpt = torch.load(fallback, map_location=device, weights_only=False)
            load_compatible_state(model, init_ckpt.get("model_state", init_ckpt))

    train_ds = try_cached_loader_v13(args.cache_root, "train", 0)
    val_ds = try_cached_loader_v13(args.cache_root, "val", 0)

    train_collate = CachedWindowCollateV13(args.batch_scenes, True)
    val_collate = CachedWindowCollateV13(args.batch_scenes, False)

    train_loader = make_loader(train_ds, args.batch_scenes, args.workers, args.prefetch, True, train_collate, True)
    val_loader = make_loader(val_ds, args.batch_scenes, args.workers, args.prefetch, False, val_collate, False)

    criterion = AdaptiveWTALossV18().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_error = float("inf")
    best_minade6 = float("inf")
    metrics_history = []

    print("\n" + "=" * 80)
    print(f" STARTING V18 SPEED-WEIGHTED FINE-TUNING ({args.epochs} EPOCHS, LR={args.lr:.2e})")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch_v18(
            model, train_loader, optimizer, scaler, criterion, device, args, epoch, amp_dtype, use_scaler
        )
        val_metrics = evaluate_validation_v18(
            model, val_loader, criterion, device, epoch, args.epochs, amp_dtype
        )

        epoch_record = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        metrics_history.append(epoch_record)
        scheduler.step()

        # Checkpoints
        ckpt_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": epoch_record,
        }
        torch.save(ckpt_payload, os.path.join(args.out_dir, "last.pth"))

        saved_tags = []
        if val_metrics["val_error_score"] < best_error:
            best_error = val_metrics["val_error_score"]
            torch.save(ckpt_payload, os.path.join(args.out_dir, "best_error_score.pth"))
            saved_tags.append("BEST_ERR")

        if val_metrics["val_minade6"] < best_minade6:
            best_minade6 = val_metrics["val_minade6"]
            torch.save(ckpt_payload, os.path.join(args.out_dir, "best_minade6.pth"))
            saved_tags.append("BEST_ADE6")

        tag_str = f" [{', '.join(saved_tags)}]" if saved_tags else ""
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({train_metrics['epoch_sec']:.1f}s) | "
            f"Train S1_ADE6 {train_metrics['train_minade6_s1']:.3f} S2_ADE6 {train_metrics['train_minade6']:.3f} "
            f"S2_ADE1 {train_metrics['train_minade1']:.3f} pick {train_metrics['train_pick_acc']:.3f} | "
            f"Val S1_ADE6 {val_metrics['val_minade6_s1']:.4f} S2_ADE6 {val_metrics['val_minade6']:.4f} "
            f"S2_ADE1 {val_metrics['val_minade1']:.4f} pick {val_metrics['val_pick_acc']:.3f} "
            f"Error {val_metrics['val_error_score']:.4f}{tag_str}",
            flush=True,
        )

        with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_history, f, indent=2)


if __name__ == "__main__":
    main()
