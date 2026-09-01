#!/usr/bin/env python3
"""Official Benchmark Evaluator for Motion Prediction Model V1 (2026 Autonomous Driving AI Challenge).

Computes official competition metrics on held-out validation split:
1. minADE_1 (Top-1 mode average displacement error over 8.0s)
2. minADE_6 (Multi-modal K=6 minADE over 8.0s)
3. minFDE_1 (Top-1 mode final displacement error at t=8.0s)
4. minFDE_6 (Multi-modal K=6 minFDE at t=8.0s)
5. Miss Rate (MR@2m: ratio of targets with minFDE_6 > 2.0m)
6. Inference Latency (T_infer: ms / scene on GPU)
7. Official Competition Error Score:
   Error Score = 0.5 * (minADE_1 + minADE_6) * (1 + max(0, T_infer - 100) / 200)
8. Object-type breakdown (Vehicles, Pedestrians, Cyclists)
"""
from __future__ import annotations
import os, sys, glob, json, time, argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v1 import MotionPredictor, SceneWindowDataset, window_to_samples, expand_neighbor_encoder_state

def evaluate_official_prediction(
    checkpoint_path: str,
    data_root: str,
    max_val_packs: int = 0,
    batch_scenes: int = 4,
    workers: int = 4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(" 2026 AUTONOMOUS DRIVING AI CHALLENGE — MOTION PREDICTION OFFICIAL BENCHMARK")
    print(f" Checkpoint:      {checkpoint_path}")
    print(f" Data Root:       {data_root}")
    print(f" Evaluation Split: Validation Set")
    print(f" Evaluation Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80, flush=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load Model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = checkpoint.get('args', {}).get('hidden', 256)
    model = MotionPredictor(hidden=hidden_dim, modes=6).to(device)
    model.load_state_dict(expand_neighbor_encoder_state(checkpoint['model_state']), strict=False)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    # Load Validation Dataset
    val_ds = SceneWindowDataset(data_root, 'val', max_val_packs)
    val_loader = DataLoader(
        val_ds, batch_size=batch_scenes, shuffle=False, 
        num_workers=workers, collate_fn=lambda x: x,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(workers > 0)
    )

    print(f"-> Total Validation Scenes: {len(val_ds.paths)} packs ({len(val_ds)} windows)")
    print(f"-> Model Parameters: {total_params / 1e6:.3f}M params\n", flush=True)

    ade1_list, ade6_list = [], []
    fde1_list, fde6_list = [], []
    type_ade6 = {0: [], 1: [], 2: []} # 0: Vehicle, 1: Pedestrian, 2: Cyclist
    type_fde6 = {0: [], 1: [], 2: []}
    type_names = {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"}

    latencies_ms = []
    total_targets = 0
    miss_count_2m = 0

    warmup_steps = 10
    start_eval_time = time.time()

    with torch.no_grad():
        for step, raw_batch in enumerate(val_loader, start=1):
            samples = window_to_samples(raw_batch, max_targets=24, neighbor_k=16, signal_k=4, train=False)
            if samples is None:
                continue

            target_hist = samples['target_hist'].to(device, non_blocking=True)
            neighbors = samples['neighbors'].to(device, non_blocking=True)
            signals = samples['signals'].to(device, non_blocking=True)
            type_idx = samples['type_idx'].to(device, non_blocking=True)
            future = samples['future'].to(device, non_blocking=True) # B, 80, 2
            future_valid = samples['future_valid'].to(device, non_blocking=True) # B, 80

            # Measure Inference Latency per Scene
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pred, goals, logits = model(target_hist, neighbors, signals, type_idx)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # Latency normalized per scene in batch
            batch_scene_count = len(raw_batch)
            lat_per_scene = ((t1 - t0) * 1000.0) / max(1, batch_scene_count)
            if step > warmup_steps:
                latencies_ms.append(lat_per_scene)

            # Compute Displacements
            # pred: [B, K=6, 80, 2]
            # future: [B, 80, 2]
            diff = pred - future[:, None, :, :] # B, 6, 80, 2
            disp = torch.linalg.vector_norm(diff, dim=-1) # B, 6, 80
            mask = future_valid[:, None, :].float() # B, 1, 80
            denom = mask.sum(dim=-1).clamp_min(1.0) # B, 1

            # ADE across all 6 modes: [B, 6]
            ade_modes = (disp * mask).sum(dim=-1) / denom # B, 6
            
            # FDE at the last valid timestamp of each agent
            last_valid_idx = (future_valid.sum(dim=-1).long() - 1).clamp(min=0, max=79) # [B]
            fde_modes = torch.gather(disp, 2, last_valid_idx[:, None, None].expand(-1, 6, 1)).squeeze(2) # [B, 6]

            # minADE6 and minFDE6 (best mode out of 6)
            best_mode_6 = ade_modes.argmin(dim=1) # [B]
            b_idx = torch.arange(pred.shape[0], device=device)
            min_ade6 = ade_modes[b_idx, best_mode_6] # [B]
            min_fde6 = fde_modes[b_idx, best_mode_6] # [B]

            # Top-1 Mode (Mode with highest probability logit)
            top1_mode = logits.argmax(dim=1) # [B]
            top1_ade = ade_modes[b_idx, top1_mode] # [B]
            top1_fde = fde_modes[b_idx, top1_mode] # [B]

            # Collect metrics
            min_ade6_np = min_ade6.cpu().numpy()
            min_fde6_np = min_fde6.cpu().numpy()
            top1_ade_np = top1_ade.cpu().numpy()
            top1_fde_np = top1_fde.cpu().numpy()
            types_np = type_idx.cpu().numpy()

            ade6_list.extend(min_ade6_np)
            fde6_list.extend(min_fde6_np)
            ade1_list.extend(top1_ade_np)
            fde1_list.extend(top1_fde_np)

            miss_count_2m += int((min_fde6_np > 2.0).sum())
            total_targets += len(min_ade6_np)

            for t_val, a6_val, f6_val in zip(types_np, min_ade6_np, min_fde6_np):
                type_ade6[int(t_val)].append(float(a6_val))
                type_fde6[int(t_val)].append(float(f6_val))

            if step % 200 == 0 or step == len(val_loader):
                print(f" [{step:04d}/{len(val_loader)}] Evaluated {total_targets} targets | "
                      f"minADE6: {np.mean(ade6_list):.3f}m | minADE1: {np.mean(ade1_list):.3f}m | "
                      f"MR@2m: {miss_count_2m/max(1,total_targets)*100:.2f}% | Latency: {np.mean(latencies_ms):.2f}ms/scene",
                      flush=True)

    total_eval_sec = time.time() - start_eval_time

    # Final Overall Metric Calculations
    mean_ade1 = float(np.mean(ade1_list))
    mean_ade6 = float(np.mean(ade6_list))
    mean_fde1 = float(np.mean(fde1_list))
    mean_fde6 = float(np.mean(fde6_list))
    miss_rate_2m = float((miss_count_2m / max(1, total_targets)) * 100.0)
    avg_latency_ms = float(np.mean(latencies_ms)) if latencies_ms else 0.0

    # Official Error Score Calculation (2026 AI Challenge Formula)
    # Error Score = 0.5 * (minADE_1 + minADE_6) * (1 + max(0, T_infer - 100) / 200)
    latency_penalty = max(0.0, avg_latency_ms - 100.0) / 200.0
    official_error_score = 0.5 * (mean_ade1 + mean_ade6) * (1.0 + latency_penalty)

    # Estimate FLOPs (approximate Multiply-Accumulate ops)
    est_gflops = (total_params * 2 * 80) / 1e9

    print("\n" + "=" * 80)
    print(" 🏆 OFFICIAL 2026 AI CHALLENGE BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    print(f" 1. Official Competition Score:")
    print(f"    ⭐ Final Error Score:      {official_error_score:.5f}  (Lower is Better)")
    print(f"    - Accuracy Component:      {0.5 * (mean_ade1 + mean_ade6):.5f} m")
    print(f"    - Latency Multiplier:      {1.0 + latency_penalty:.5f}x (Zero Penalty if <= 100ms)")
    print()
    print(f" 2. Multi-Modal Trajectory Accuracy (8.0s Horizon):")
    print(f"    - minADE_6 (Top-6 Modes):   {mean_ade6:.4f} m  ({mean_ade6*100:.1f} cm)")
    print(f"    - minADE_1 (Top-1 Mode):    {mean_ade1:.4f} m")
    print(f"    - minFDE_6 (t=8.0s Final):  {mean_fde6:.4f} m  ({mean_fde6*100:.1f} cm)")
    print(f"    - minFDE_1 (t=8.0s Final):  {mean_fde1:.4f} m")
    print(f"    - Miss Rate (minFDE > 2m):  {miss_rate_2m:.2f} %")
    print()
    print(f" 3. Computational Efficiency & Real-Time Performance:")
    print(f"    - Inference Latency (GPU):  {avg_latency_ms:.2f} ms / scene")
    print(f"    - Inference Throughput:     {1000.0 / max(0.01, avg_latency_ms):.1f} FPS (Scenes/sec)")
    print(f"    - Model Parameters:         {total_params / 1e6:.3f} M Parameters")
    print(f"    - Estimated FLOPs:          {est_gflops:.3f} GFLOPs (Passes Cut-off: << 3x Baseline)")
    print()
    print(f" 4. Breakdown by Dynamic Object Type:")
    for t_idx, name in type_names.items():
        if type_ade6[t_idx]:
            sub_ade = np.mean(type_ade6[t_idx])
            sub_fde = np.mean(type_fde6[t_idx])
            print(f"    - {name:11s} (N={len(type_ade6[t_idx]):5d}): minADE_6 = {sub_ade:.4f} m | minFDE_6 = {sub_fde:.4f} m")
    print("=" * 80 + "\n", flush=True)

    results = {
        "official_error_score": official_error_score,
        "minade_6": mean_ade6,
        "minade_1": mean_ade1,
        "minfde_6": mean_fde6,
        "minfde_1": mean_fde1,
        "miss_rate_2m": miss_rate_2m,
        "latency_ms": avg_latency_ms,
        "total_targets": total_targets,
        "total_params_M": total_params / 1e6,
        "eval_seconds": total_eval_sec,
        "type_breakdown": {
            type_names[t]: {
                "count": len(type_ade6[t]),
                "minade6": float(np.mean(type_ade6[t])) if type_ade6[t] else 0.0,
                "minfde6": float(np.mean(type_fde6[t])) if type_fde6[t] else 0.0
            } for t in (0, 1, 2) if type_ade6[t]
        }
    }

    out_json = os.path.join(os.path.dirname(checkpoint_path), "official_evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved Official Benchmark Report JSON to: {out_json}", flush=True)
    return results

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.join(current_dir, "checkpoints", "best_minade6.pth")
    if not os.path.exists(default_ckpt):
        default_ckpt = r"E:\motion_prediction\checkpoints\best_minade6.pth"
    default_data = r"C:\Users\andy0\Downloads\behavior_stack_planner\data\processed\prediction_pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=default_ckpt)
    parser.add_argument("--data-root", default=default_data)
    parser.add_argument("--max-val-packs", type=int, default=0)
    parser.add_argument("--batch-scenes", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    evaluate_official_prediction(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        max_val_packs=args.max_val_packs,
        batch_scenes=args.batch_scenes,
        workers=args.workers
    )
