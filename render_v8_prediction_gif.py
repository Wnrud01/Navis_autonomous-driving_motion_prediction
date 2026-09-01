#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v1 import SceneWindowDataset, list_pt_paths
from src.train_motion_prediction_v3 import MAP_K, WindowSampleCollateV3
from src.train_motion_prediction_v6 import MotionPredictorV6
from train_motion_prediction_v2_awta import make_loader, move_samples
from train_motion_prediction_v3 import model_forward

COLORS = ["#ff4d6d", "#ffd166", "#06d6a0", "#118ab2", "#9b5de5", "#f77f00"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"E:\motion_planning\data\processed\prediction_pt_85k")
    parser.add_argument("--checkpoint", default=r"checkpoints\v8_hardcls\best_error_score.pth")
    parser.add_argument("--output", default=r"checkpoints\v8_hardcls\v8_prediction_rollout.gif")
    parser.add_argument("--png", default=r"checkpoints\v8_hardcls\v8_prediction_snapshot.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MotionPredictorV6(hidden=256, modes=6).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    val_paths = list_pt_paths(args.data_root, "val", 0)
    probe = torch.load(val_paths[0], map_location="cpu", weights_only=False)
    n_windows = max(1, len(probe.get("windows", [])))
    dataset = SceneWindowDataset(args.data_root, "val", paths=val_paths, n_windows=n_windows)
    loader = make_loader(
        dataset, batch_size=1, workers=0, prefetch=2, shuffle=False,
        collate_fn=WindowSampleCollateV3(24, 16, 4, MAP_K, False), pin_memory=(device.type == "cuda"),
    )

    samples = next(s for s in loader if s is not None and s["target_hist"].shape[0] > 0)
    samples = move_samples(samples, device)
    with torch.no_grad():
        pred, _, logits = model_forward(model, samples)

    pred = pred[0].float().cpu().numpy()
    gt = samples["future"][0].float().cpu().numpy()
    gt_valid = samples["future_valid"][0].cpu().numpy().astype(bool)
    hist = samples["target_hist"][0].float().cpu().numpy()
    neigh = samples["neighbors"][0].float().cpu().numpy()
    probs = torch.softmax(logits[0], dim=-1).float().cpu().numpy()

    valid_len = int(gt_valid.sum())
    if valid_len <= 0:
        valid_len = len(gt)
    gt = gt[:valid_len]
    pred = pred[:, :valid_len]
    dists = np.linalg.norm(pred - gt[None, :, :], axis=-1)
    ades = dists.mean(axis=1)
    best = int(np.argmin(ades))
    selected = int(np.argmax(probs))
    radius = max(25.0, float(np.max(np.abs(np.concatenate([gt, pred.reshape(-1, 2)], axis=0)))) * 1.15)

    def make_frame(step: int) -> Image.Image:
        fig, ax = plt.subplots(figsize=(9, 9), facecolor="#0b0e14")
        ax.set_facecolor("#0b0e14")
        ax.scatter([0], [0], s=140, c="#ff9f1c", marker="o", edgecolors="white", linewidths=1.0, label="Target now", zorder=10)
        if hist.shape[0] > 1:
            ax.plot(hist[:, 0], hist[:, 1], color="#ff9f1c", linewidth=2.0, alpha=0.9, label="History")
        current_neighbors = neigh[:, -1, :2]
        current_valid = neigh[:, -1, 5] > 0.5
        if current_valid.any():
            ax.scatter(current_neighbors[current_valid, 0], current_neighbors[current_valid, 1], s=34, c="#64748b", alpha=0.8, label="Neighbors")
        if step > 0:
            ax.plot(gt[:step, 0], gt[:step, 1], "--", color="#00e5ff", linewidth=3.0, label="Ground truth", zorder=7)
            ax.scatter([gt[min(step, len(gt))-1, 0]], [gt[min(step, len(gt))-1, 1]], c="#00e5ff", s=55, zorder=8)
            for mode in range(6):
                alpha = 1.0 if mode in (selected, best) else max(0.18, float(probs[mode]) * 1.4)
                lw = 3.0 if mode == selected else (2.4 if mode == best else 1.1)
                label = f"Mode {mode+1} P={probs[mode]*100:.1f}%" if mode in (selected, best) else None
                ax.plot(pred[mode, :step, 0], pred[mode, :step, 1], color=COLORS[mode], linewidth=lw, alpha=alpha, label=label, zorder=6)
                ax.scatter([pred[mode, step-1, 0]], [pred[mode, step-1, 1]], color=COLORS[mode], marker="*", s=75 if mode in (selected, best) else 30, zorder=9)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect("equal")
        ax.grid(color="#273244", alpha=0.35)
        ax.tick_params(colors="#aab4c3")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.set_title(f"V8 Motion Prediction | t={step*0.1:.1f}s/{valid_len*0.1:.1f}s\nselected mode={selected+1}, ADE-best mode={best+1}, minADE6={ades[best]:.2f}m", color="white", fontsize=13, pad=12)
        leg = ax.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#334155", labelcolor="white")
        fig.tight_layout()
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        plt.close(fig)
        return image

    frames = [make_frame(step) for step in range(0, valid_len + 1, 2)]
    if len(frames) < 2 or frames[-1].size != frames[0].size:
        frames.append(make_frame(valid_len))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.png) or ".", exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=False)
    frames[-1].save(args.png)
    print(f"Saved GIF: {args.output}")
    print(f"Saved PNG: {args.png}")
    print(f"selected_mode={selected+1}, ade_best_mode={best+1}, minADE6={ades[best]:.4f}, selected_ADE={ades[selected]:.4f}")


if __name__ == "__main__":
    main()
