#!/usr/bin/env python3
"""Evaluate Trajectory NMS & Soft Cluster Aggregation on Motion Prediction V16.

Techniques evaluated:
1. Baseline: Raw Argmax (single mode selection).
2. Density-Weighted Mode Selection: Selects mode with highest neighborhood density score.
3. Soft Cluster Trajectory Averaging: Averages trajectories within the winning cluster weighted by probability.
4. Greedy NMS: Merges modes within distance threshold r into cluster centers.
"""
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


def compute_pairwise_traj_dist(traj: torch.Tensor) -> torch.Tensor:
    """Compute pairwise mean ADE between all pairs of 6 modes.
    traj: [B, K=6, T=80, 2]
    Returns: [B, K, K] distance matrix in meters.
    """
    B, K, T, _ = traj.shape
    # [B, K, 1, T, 2] - [B, 1, K, T, 2] -> [B, K, K, T, 2]
    diff = traj.unsqueeze(2) - traj.unsqueeze(1)
    dist_per_step = torch.linalg.vector_norm(diff, dim=-1)  # [B, K, K, T]
    return dist_per_step.mean(dim=-1)  # [B, K, K]


def apply_soft_density_nms(
    traj: torch.Tensor,     # [B, 6, 80, 2]
    logits: torch.Tensor,   # [B, 6]
    sigma: float = 1.2,
    do_blend: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft Density-Weighted NMS & Trajectory Blending.
    sigma: kernel bandwidth in meters.
    do_blend: if True, blends trajectories within the cluster.
    Returns: top1_traj [B, 80, 2], top1_mode_idx [B]
    """
    B, K, T, _ = traj.shape
    device = traj.device
    probs = F.softmax(logits, dim=-1)  # [B, 6]

    # Pairwise trajectory distance [B, 6, 6]
    dist_mat = compute_pairwise_traj_dist(traj)  # [B, 6, 6]

    # Gaussian affinity weights: [B, 6, 6]
    affinity = torch.exp(- (dist_mat ** 2) / (2.0 * (sigma ** 2)))

    # Density score for each mode i: sum_j (probs_j * affinity_ij)
    density_scores = (probs.unsqueeze(1) * affinity).sum(dim=-1)  # [B, 6]
    best_mode = density_scores.argmax(dim=-1)  # [B]

    b_idx = torch.arange(B, device=device)
    if not do_blend:
        return traj[b_idx, best_mode], best_mode

    # Blend trajectories in the cluster of best_mode
    # Cluster weights for each mode j: probs_j * affinity(best_mode, j)
    cluster_weights = probs * affinity[b_idx, best_mode, :]  # [B, 6]
    cluster_weights = cluster_weights / cluster_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [B, 6]

    # Weighted trajectory: sum_j (weight_j * traj_j)
    blended_traj = (cluster_weights[:, :, None, None] * traj).sum(dim=1)  # [B, 80, 2]
    return blended_traj, best_mode


def apply_greedy_nms(
    traj: torch.Tensor,     # [B, 6, 80, 2]
    logits: torch.Tensor,   # [B, 6]
    radius: float = 1.5,
    do_blend: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy Trajectory NMS.
    radius: distance threshold in meters to group trajectories.
    """
    B, K, T, _ = traj.shape
    device = traj.device
    probs = F.softmax(logits, dim=-1)
    dist_mat = compute_pairwise_traj_dist(traj)  # [B, 6, 6]

    # Mask of modes within radius of each mode: [B, 6, 6]
    in_cluster = (dist_mat <= radius)

    # Cluster aggregated probability for each mode:
    cluster_prob = (probs.unsqueeze(1) * in_cluster.float()).sum(dim=-1)  # [B, 6]
    best_mode = cluster_prob.argmax(dim=-1)

    b_idx = torch.arange(B, device=device)
    if not do_blend:
        return traj[b_idx, best_mode], best_mode

    # Blend all modes within the winning cluster
    cluster_mask = in_cluster[b_idx, best_mode, :].float()  # [B, 6]
    cluster_weights = probs * cluster_mask
    cluster_weights = cluster_weights / cluster_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    blended_traj = (cluster_weights[:, :, None, None] * traj).sum(dim=1)  # [B, 80, 2]
    return blended_traj, best_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=r"E:\motion_prediction\checkpoints\v16_twostage\best_error_score.pth")
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

    print("=" * 80)
    print(" EVALUATING TRAJECTORY NMS & CLUSTER AGGREGATION ON V16")
    print(f" Checkpoint: {args.ckpt}")
    print(f" Validation Cache: {args.cache_root}")
    print("=" * 80, flush=True)

    val_ds = try_cached_loader_v13(args.cache_root, "val", 0)
    val_collate = CachedWindowCollateV13(args.max_targets, False)
    val_loader = make_loader(val_ds, args.batch_scenes, args.workers, args.prefetch, False, val_collate, True)
    print(f"-> Val samples: {len(val_ds)} scenes across {len(val_loader)} batches", flush=True)

    model = MotionPredictorV16(hidden=args.hidden, modes=6).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    missing, unexpected = load_compatible_state(model, ckpt["model_state"])
    print(f"-> Loaded checkpoint epoch {ckpt.get('epoch', '?')} (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    model.eval()

    # Configurations to test:
    # 1. Baseline Argmax (no NMS)
    # 2. Soft Density NMS with different sigmas (pick only vs pick+blend)
    # 3. Greedy NMS with different radii (pick only vs pick+blend)
    sigmas = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    radii = [0.8, 1.0, 1.2, 1.5, 2.0]

    # Accumulators
    total_targets = 0
    ade6_sum = 0.0

    # Dictionaries for various strategies
    ade1_sums = {"baseline": 0.0}
    pick_correct_sums = {"baseline": 0.0}

    for s in sigmas:
        ade1_sums[f"density_pick_s{s}"] = 0.0
        ade1_sums[f"density_blend_s{s}"] = 0.0
        pick_correct_sums[f"density_pick_s{s}"] = 0.0
        pick_correct_sums[f"density_blend_s{s}"] = 0.0

    for r in radii:
        ade1_sums[f"greedy_pick_r{r}"] = 0.0
        ade1_sums[f"greedy_blend_r{r}"] = 0.0
        pick_correct_sums[f"greedy_pick_r{r}"] = 0.0
        pick_correct_sums[f"greedy_blend_r{r}"] = 0.0

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

            # Ground truth step distance for all 6 modes
            diff_modes = pred_2 - future.unsqueeze(1)  # [B, 6, 80, 2]
            disp_modes = torch.linalg.vector_norm(diff_modes, dim=-1)  # [B, 6, 80]
            ade_modes = (disp_modes * mask).sum(dim=-1) / denom  # [B, 6]

            # 1. Oracle minADE6
            best_mode_oracle = ade_modes.argmin(dim=1)
            ade6_sum += float(ade_modes[b_idx, best_mode_oracle].sum())

            # Helper function to evaluate a chosen/blended top-1 trajectory
            def eval_top1_traj(top1_tr, chosen_mode=None):
                diff_top1 = top1_tr - future  # [B, 80, 2]
                disp_top1 = torch.linalg.vector_norm(diff_top1, dim=-1)  # [B, 80]
                ade_top1 = (disp_top1 * future_valid.float()).sum(dim=-1) / denom.squeeze(1)  # [B]
                ade_val = float(ade_top1.sum())
                correct_val = float((chosen_mode == best_mode_oracle).float().sum()) if chosen_mode is not None else 0.0
                return ade_val, correct_val

            # 1. Baseline Argmax
            baseline_mode = logits_2.argmax(dim=-1)
            b_ade, b_corr = eval_top1_traj(pred_2[b_idx, baseline_mode], baseline_mode)
            ade1_sums["baseline"] += b_ade
            pick_correct_sums["baseline"] += b_corr

            # 2. Soft Density NMS
            for s in sigmas:
                tr_pick, m_pick = apply_soft_density_nms(pred_2, logits_2, sigma=s, do_blend=False)
                a, c = eval_top1_traj(tr_pick, m_pick)
                ade1_sums[f"density_pick_s{s}"] += a
                pick_correct_sums[f"density_pick_s{s}"] += c

                tr_blend, m_blend = apply_soft_density_nms(pred_2, logits_2, sigma=s, do_blend=True)
                a_b, _ = eval_top1_traj(tr_blend, m_blend)
                ade1_sums[f"density_blend_s{s}"] += a_b
                pick_correct_sums[f"density_blend_s{s}"] += c

            # 3. Greedy NMS
            for r in radii:
                tr_pick, m_pick = apply_greedy_nms(pred_2, logits_2, radius=r, do_blend=False)
                a, c = eval_top1_traj(tr_pick, m_pick)
                ade1_sums[f"greedy_pick_r{r}"] += a
                pick_correct_sums[f"greedy_pick_r{r}"] += c

                tr_blend, m_blend = apply_greedy_nms(pred_2, logits_2, radius=r, do_blend=True)
                a_b, _ = eval_top1_traj(tr_blend, m_blend)
                ade1_sums[f"greedy_blend_r{r}"] += a_b
                pick_correct_sums[f"greedy_blend_r{r}"] += c

            total_targets += b

            if step % 50 == 0 or step == len(val_loader):
                elapsed = time.time() - t_start
                print(f"Processed [{step:04d}/{len(val_loader):04d}] ({total_targets} targets, {elapsed:.1f}s)", flush=True)

    minade6 = ade6_sum / total_targets

    print("\n" + "=" * 80)
    print(f" FULL VALIDATION RESULTS ACROSS {total_targets} TARGETS (minADE6 = {minade6:.4f}m)")
    print("=" * 80)
    print(f"{'Strategy / Configuration':<35} | {'minADE1':<10} | {'Pick Acc':<10} | {'Error Score':<12} | {'vs Baseline':<10}")
    print("-" * 80)

    base_ade1 = ade1_sums["baseline"] / total_targets
    base_acc = pick_correct_sums["baseline"] / total_targets
    base_err = 0.5 * (minade6 + base_ade1)
    print(f"{'1. Baseline (Raw Argmax)':<35} | {base_ade1:.4f}m    | {base_acc*100:.2f}%     | {base_err:.4f}       | {'0.00% (Ref)':<10}")
    print("-" * 80)

    results = []
    for k in ade1_sums:
        if k == "baseline":
            continue
        ade1 = ade1_sums[k] / total_targets
        acc = pick_correct_sums[k] / total_targets
        err = 0.5 * (minade6 + ade1)
        diff_err = (err - base_err) / base_err * 100
        results.append((k, ade1, acc, err, diff_err))

    # Sort results by Error Score ascending
    results.sort(key=lambda x: x[3])

    print("--- TOP 10 NMS / CLUSTER STRATEGIES ---")
    for name, ade1, acc, err, diff_err in results[:10]:
        diff_str = f"{diff_err:+.2f}%"
        print(f"{name:<35} | {ade1:.4f}m    | {acc*100:.2f}%     | {err:.4f}       | {diff_str:<10}")
    print("=" * 80)


if __name__ == "__main__":
    main()
