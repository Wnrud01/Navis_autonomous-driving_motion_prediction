#!/usr/bin/env python3
"""Evaluate Model Averaging (Soup / Ensemble) + Density NMS on V16."""
from __future__ import annotations

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cached_collate_v13 import CachedWindowCollateV13, try_cached_loader_v13
from src.train_motion_prediction_v16 import MotionPredictorV16
from src.train_motion_prediction_v3 import load_compatible_state
from train_motion_prediction_v2_awta import configure_runtime, make_loader, move_samples
from train_motion_prediction_v16 import model_forward_v16
from evaluate_v16_nms import apply_soft_density_nms


def average_checkpoints(ckpt_paths: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    """Average model weights across multiple checkpoints (Model Soup / SWA)."""
    avg_state = {}
    valid_paths = [p for p in ckpt_paths if os.path.exists(p)]
    print(f"-> Averaging {len(valid_paths)} checkpoints: {valid_paths}", flush=True)

    for i, path in enumerate(valid_paths):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt["model_state"]
        for k, v in state.items():
            if i == 0:
                avg_state[k] = v.clone().float()
            else:
                avg_state[k] += v.float()

    for k in avg_state:
        avg_state[k] = (avg_state[k] / len(valid_paths)).to(dtype=torch.float32)

    return avg_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=r"E:\motion_prediction\checkpoints\v16_twostage")
    parser.add_argument("--cache-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2_cache_v13")
    parser.add_argument("--batch-scenes", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--max-targets", type=int, default=24)
    parser.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    configure_runtime(args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" and torch.cuda.is_bf16_supported() else torch.float32

    val_ds = try_cached_loader_v13(args.cache_root, "val", 0)
    val_collate = CachedWindowCollateV13(args.max_targets, False)
    val_loader = make_loader(val_ds, args.batch_scenes, args.workers, args.prefetch, False, val_collate, True)

    # 1. Load Averaged Model (Best Error + Best ADE6 + Last)
    ckpt_paths = [
        os.path.join(args.ckpt_dir, "best_error_score.pth"),
        os.path.join(args.ckpt_dir, "best_minade6.pth"),
        os.path.join(args.ckpt_dir, "last.pth"),
    ]
    avg_state = average_checkpoints(ckpt_paths, device)
    model = MotionPredictorV16(hidden=args.hidden, modes=6).to(device)
    load_compatible_state(model, avg_state)
    model.eval()

    sigmas = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5]
    total_targets = 0
    ade6_sum = 0.0
    ade1_sums = {"baseline_avg": 0.0}
    pick_sums = {"baseline_avg": 0.0}

    for s in sigmas:
        ade1_sums[f"density_pick_s{s}"] = 0.0
        pick_sums[f"density_pick_s{s}"] = 0.0
        ade1_sums[f"density_blend_s{s}"] = 0.0
        pick_sums[f"density_blend_s{s}"] = 0.0

    t_start = time.time()

    with torch.no_grad():
        for step, samples in enumerate(val_loader, start=1):
            if samples is None or samples["target_hist"].shape[0] == 0:
                continue
            samples = move_samples(samples, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                pred_2, goals_2, logits_2, pred_1, goals_1 = model_forward_v16(model, samples)

            future = samples["future"]
            future_valid = samples["future_valid"]
            mask = future_valid[:, None, :].float()
            denom = mask.sum(dim=-1).clamp_min(1.0)
            b = future.shape[0]
            b_idx = torch.arange(b, device=device)

            diff_modes = pred_2 - future.unsqueeze(1)
            disp_modes = torch.linalg.vector_norm(diff_modes, dim=-1)
            ade_modes = (disp_modes * mask).sum(dim=-1) / denom

            best_mode_oracle = ade_modes.argmin(dim=1)
            ade6_sum += float(ade_modes[b_idx, best_mode_oracle].sum())

            def eval_top1_traj(top1_tr, chosen_mode=None):
                diff_top1 = top1_tr - future
                disp_top1 = torch.linalg.vector_norm(diff_top1, dim=-1)
                ade_top1 = (disp_top1 * future_valid.float()).sum(dim=-1) / denom.squeeze(1)
                ade_val = float(ade_top1.sum())
                correct_val = float((chosen_mode == best_mode_oracle).float().sum()) if chosen_mode is not None else 0.0
                return ade_val, correct_val

            # Baseline
            base_m = logits_2.argmax(dim=-1)
            a, c = eval_top1_traj(pred_2[b_idx, base_m], base_m)
            ade1_sums["baseline_avg"] += a
            pick_sums["baseline_avg"] += c

            # Soft Density NMS
            for s in sigmas:
                tr_pick, m_pick = apply_soft_density_nms(pred_2, logits_2, sigma=s, do_blend=False)
                a, c = eval_top1_traj(tr_pick, m_pick)
                ade1_sums[f"density_pick_s{s}"] += a
                pick_sums[f"density_pick_s{s}"] += c

                tr_blend, m_blend = apply_soft_density_nms(pred_2, logits_2, sigma=s, do_blend=True)
                a_b, _ = eval_top1_traj(tr_blend, m_blend)
                ade1_sums[f"density_blend_s{s}"] += a_b
                pick_sums[f"density_blend_s{s}"] += c

            total_targets += b

    minade6 = ade6_sum / total_targets
    print("\n" + "=" * 80)
    print(f" SWA MODEL SOUP + DENSITY NMS RESULTS ({total_targets} targets, {time.time()-t_start:.1f}s)")
    print("=" * 80)
    base_ade1 = ade1_sums["baseline_avg"] / total_targets
    base_err = 0.5 * (minade6 + base_ade1)
    base_acc = pick_sums["baseline_avg"] / total_targets
    print(f"{'Averaged Checkpoints (Raw Argmax)':<35} | {base_ade1:.4f}m    | {base_acc*100:.2f}%     | {base_err:.4f}")
    print("-" * 80)

    results = []
    for k in ade1_sums:
        if k == "baseline_avg":
            continue
        ade1 = ade1_sums[k] / total_targets
        acc = pick_sums[k] / total_targets
        err = 0.5 * (minade6 + ade1)
        results.append((k, ade1, acc, err))

    results.sort(key=lambda x: x[3])
    for name, ade1, acc, err in results[:10]:
        diff_str = f"{(err - 0.8614)/0.8614 * 100:+.2f}%"
        print(f"{name:<35} | {ade1:.4f}m    | {acc*100:.2f}%     | {err:.4f}       | vs V16 Best: {diff_str}")
    print("=" * 80)


if __name__ == "__main__":
    main()
