#!/usr/bin/env python3
"""Evaluate Trajectory NMS & Soft Density Post-Processing on Motion Prediction V17 X-Large.

Evaluates across all 24,097 validation scenes (292,298 targets):
1. Raw Top-1 Selection
2. Density-Weighted Mode Selection (without blending)
3. Soft Density Cluster Blending (sigma=1.2, 2.0, 3.0m)
"""
from __future__ import annotations

import argparse, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cached_collate_v13 import CachedWindowCollateV13, try_cached_loader_v13
from src.train_motion_prediction_v17 import MotionPredictorV17
from src.train_motion_prediction_v3 import load_compatible_state
from train_motion_prediction_v2_awta import configure_runtime, make_loader, move_samples
from train_motion_prediction_v17 import model_forward_v17
from evaluate_v16_nms import compute_pairwise_traj_dist, apply_soft_density_nms


def evaluate_nms_v17(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    sigma_list: list[float] = [1.2, 2.0, 3.0, 4.0],
):
    model.eval()

    total_targets = 0
    total_minade6 = 0.0
    total_minfde6 = 0.0
    total_minade1_raw = 0.0
    total_minfde1_raw = 0.0
    correct_pick_raw = 0

    density_results = {
        f"density_nobyo_s{s:.1f}": {"ade1": 0.0, "pick": 0} for s in sigma_list
    }
    blend_results = {
        f"blend_s{s:.1f}": {"ade1": 0.0, "fde1": 0.0} for s in sigma_list
    }

    t0 = time.time()
    with torch.no_grad():
        for step, samples in enumerate(val_loader, start=1):
            samples = move_samples(samples, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                pred_2, goals_2, logits_2, pred_1, goals_1 = model_forward_v17(model, samples)

            future = samples["future"]  # [B, 80, 2]
            f_valid = samples["future_valid"]  # [B, 80]
            B = future.shape[0]
            total_targets += B

            # ADE & FDE for all 6 modes: [B, 6]
            diff = pred_2 - future.unsqueeze(1)  # [B, 6, 80, 2]
            dist_per_step = torch.linalg.vector_norm(diff, dim=-1)  # [B, 6, 80]
            valid_mask = f_valid.unsqueeze(1).float()  # [B, 1, 80]
            valid_counts = valid_mask.sum(dim=-1).clamp_min(1.0)  # [B, 1]

            ade_modes = (dist_per_step * valid_mask).sum(dim=-1) / valid_counts  # [B, 6]
            fde_modes = dist_per_step[:, :, -1]  # [B, 6]

            # Best ground truth mode (Winner)
            best_gt_mode = ade_modes.argmin(dim=-1)  # [B]
            min_ade6 = ade_modes.min(dim=-1).values.sum().item()
            min_fde6 = fde_modes.gather(1, best_gt_mode.unsqueeze(-1)).squeeze(-1).sum().item()
            total_minade6 += min_ade6
            total_minfde6 += min_fde6

            # 1. Raw Top-1 Selection (Argmax)
            top1_raw = logits_2.argmax(dim=-1)  # [B]
            ade1_raw = ade_modes.gather(1, top1_raw.unsqueeze(-1)).squeeze(-1).sum().item()
            fde1_raw = fde_modes.gather(1, top1_raw.unsqueeze(-1)).squeeze(-1).sum().item()
            total_minade1_raw += ade1_raw
            total_minfde1_raw += fde1_raw
            correct_pick_raw += (top1_raw == best_gt_mode).sum().item()

            # 2. Test Soft Density NMS across various sigmas
            for s in sigma_list:
                # Without blending
                _, mode_dens = apply_soft_density_nms(pred_2, logits_2, sigma=s, do_blend=False)
                ade1_dens = ade_modes.gather(1, mode_dens.unsqueeze(-1)).squeeze(-1).sum().item()
                density_results[f"density_nobyo_s{s:.1f}"]["ade1"] += ade1_dens
                density_results[f"density_nobyo_s{s:.1f}"]["pick"] += (mode_dens == best_gt_mode).sum().item()

                # With trajectory blending
                blend_traj, _ = apply_soft_density_nms(pred_2, logits_2, sigma=s, do_blend=True)
                diff_blend = blend_traj - future  # [B, 80, 2]
                dist_blend = torch.linalg.vector_norm(diff_blend, dim=-1)  # [B, 80]
                ade1_bl = (dist_blend * f_valid.float()).sum(dim=-1) / f_valid.float().sum(dim=-1).clamp_min(1.0)
                fde1_bl = dist_blend[:, -1]
                blend_results[f"blend_s{s:.1f}"]["ade1"] += ade1_bl.sum().item()
                blend_results[f"blend_s{s:.1f}"]["fde1"] += fde1_bl.sum().item()

            if step % 200 == 0 or step == len(val_loader):
                print(f"-> Evaluated [{step}/{len(val_loader)}] batches ({total_targets:,} targets)...", flush=True)

    eval_sec = time.time() - t0
    m_minade6 = total_minade6 / total_targets
    m_minfde6 = total_minfde6 / total_targets
    m_ade1_raw = total_minade1_raw / total_targets
    m_fde1_raw = total_minfde1_raw / total_targets
    raw_pick = correct_pick_raw / total_targets
    err_raw = 0.5 * (m_minade6 + m_ade1_raw)

    print("\n" + "=" * 80)
    print(f" V17 X-LARGE FULL VALIDATION RESULTS ({total_targets:,} targets in {eval_sec:.1f}s)")
    print("=" * 80)
    print(f"-> Base minADE6:  {m_minade6:.4f} m | minFDE6: {m_minfde6:.4f} m")
    print(f"-> Raw minADE1:   {m_ade1_raw:.4f} m | minFDE1: {m_fde1_raw:.4f} m | Pick: {raw_pick*100:.2f}% | Error: {err_raw:.4f}")
    print("-" * 80)
    print(" POST-PROCESSING TECHNIQUES:")

    best_method = "Raw Top-1"
    best_err = err_raw
    best_ade1 = m_ade1_raw

    for s in sigma_list:
        k_d = f"density_nobyo_s{s:.1f}"
        ade_d = density_results[k_d]["ade1"] / total_targets
        pick_d = density_results[k_d]["pick"] / total_targets
        err_d = 0.5 * (m_minade6 + ade_d)
        print(f"  [Density Selection  sigma={s:.1f}m] minADE1: {ade_d:.4f} m | Pick: {pick_d*100:.2f}% | Error: {err_d:.4f} (Delta: {err_d - err_raw:+.4f})")
        if err_d < best_err:
            best_err, best_method, best_ade1 = err_d, f"Density Mode sigma={s:.1f}", ade_d

        k_b = f"blend_s{s:.1f}"
        ade_b = blend_results[k_b]["ade1"] / total_targets
        fde_b = blend_results[k_b]["fde1"] / total_targets
        err_b = 0.5 * (m_minade6 + ade_b)
        print(f"  [Soft Cluster Blend sigma={s:.1f}m] minADE1: {ade_b:.4f} m | minFDE1: {fde_b:.4f} m | Error: {err_b:.4f} (Delta: {err_b - err_raw:+.4f})")
        if err_b < best_err:
            best_err, best_method, best_ade1 = err_b, f"Cluster Blend sigma={s:.1f}", ade_b

    print("=" * 80)
    print(f"🏆 BEST POST-PROCESSED SCORE: {best_err:.4f} ({best_method})")
    print(f"   minADE6: {m_minade6:.4f} m | minADE1: {best_ade1:.4f} m | Error Score: {best_err:.4f}")
    print("=" * 80)
    return {
        "minade6": m_minade6,
        "raw_ade1": m_ade1_raw,
        "raw_error": err_raw,
        "best_ade1": best_ade1,
        "best_error": best_err,
        "best_method": best_method,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=r"E:\motion_prediction\checkpoints\v17_xlarge\best_error_score.pth")
    parser.add_argument("--cache-root", default=r"E:\motion_prediction\data\processed\prediction_pt_85k_v2_cache_v13")
    parser.add_argument("--batch-scenes", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", default="bf16")
    args = parser.parse_args()

    configure_runtime(args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else (torch.float16 if args.amp == "fp16" else torch.float32)
    print(f"Loading V17 X-Large Model from: {args.ckpt}")
    model = MotionPredictorV17(hidden=768, modes=6, nhead=12, dropout=0.1).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    load_compatible_state(model, ckpt["model_state"])

    val_ds = try_cached_loader_v13(args.cache_root, "val", 0)
    val_collate = CachedWindowCollateV13(args.batch_scenes, False)
    val_loader = make_loader(val_ds, args.batch_scenes, args.workers, 4, False, val_collate, False)

    evaluate_nms_v17(model, val_loader, device, amp_dtype)


if __name__ == "__main__":
    main()
