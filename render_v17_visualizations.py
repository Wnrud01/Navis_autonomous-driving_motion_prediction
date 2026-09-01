#!/usr/bin/env python3
"""High-Fidelity Visualizer for Motion Prediction V17 X-Large SWA Model (All-Time SOTA Champion).

Generates:
1. Multi-Target Vector HD Map Snapshot PNGs with road centerlines, crosswalks, stop lines.
2. 8.0-Second 10Hz Animated Rollout Prediction GIFs.
3. 10 Diverse Validation Scenes + 2x5 Showcase Summary Collage.
"""
from __future__ import annotations

import os, sys, glob, json, math, random
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v17 import MotionPredictorV17
from src.train_motion_prediction_v3 import load_compatible_state
from src.train_motion_prediction_v13_collate import _scene_v13
from src.train_motion_prediction_v1 import TYPE_TO_INDEX

# Vector HD Map Styles
RG_STYLE = {
    1:  ("#64748b", 0.9, 0.45, 1, "--"), # LANE_FREEWAY
    2:  ("#64748b", 0.9, 0.45, 1, "--"), # LANE_SURFACE_STREET
    3:  ("#64748b", 0.8, 0.40, 1, "--"), # LANE_BIKE_LANE
    6:  ("#eab308", 1.4, 0.80, 2, "-"),  # ROAD_LINE_BROKEN_YELLOW
    7:  ("#eab308", 1.6, 0.85, 2, "-"),  # ROAD_LINE_SOLID_YELLOW
    8:  ("#eab308", 1.8, 0.90, 2, "-"),  # ROAD_LINE_SOLID_DOUBLE_YELLOW
    9:  ("#e2e8f0", 1.1, 0.65, 2, "--"), # ROAD_LINE_BROKEN_WHITE
    10: ("#e2e8f0", 1.3, 0.75, 2, "-"),  # ROAD_LINE_SOLID_WHITE
    11: ("#f1f5f9", 1.6, 0.90, 2, "-"),  # ROAD_LINE_SOLID_DOUBLE_WHITE
    12: ("#eab308", 1.3, 0.70, 2, "--"), # ROAD_LINE_PASSING_DOUBLE_YELLOW
    15: ("#334155", 2.4, 0.95, 3, "-"),  # ROAD_EDGE_BOUNDARY
    16: ("#475569", 2.0, 0.80, 3, "-"),  # ROAD_EDGE_MEDIAN
    17: ("#e2e8f0", 1.4, 0.70, 2, "-"),  # STOP_SIGN / STOP_LINE
    18: ("#cbd5e1", 1.0, 0.55, 2, "--"), # CROSSWALK
    19: ("#fbbf24", 1.3, 0.65, 2, "-"),  # SPEED_BUMP
    20: ("#f87171", 1.1, 0.55, 2, "-"),  # OTHER
}
DEFAULT_RG_STYLE = ("#475569", 0.9, 0.55, 1, "-")
PALETTE = ["#00FF66", "#FF3366", "#00E5FF", "#FFCC00", "#CC66FF", "#FF9933", "#00FFCC", "#FF5588", "#38BDF8", "#A855F7", "#F43F5E", "#10B981"]


def get_vehicle_corners(cx: float, cy: float, heading: float, length: float = 4.6, width: float = 2.1) -> np.ndarray:
    half_length = max(float(length), 0.5) / 2.0
    half_width = max(float(width), 0.3) / 2.0
    corners = np.array([
        [half_length, half_width],
        [half_length, -half_width],
        [-half_length, -half_width],
        [-half_length, half_width],
    ])
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return corners @ rot.T + np.array([cx, cy])


def draw_vector_roadgraph(ax, static: dict, center_x: float, center_y: float, view_radius: float = 65.0, seg_len: float = 0.8):
    if 'roadgraph_xyz_world' not in static or 'roadgraph_dir_world' not in static:
        return
    rg_xyz = static['roadgraph_xyz_world'].numpy().reshape(-1, 3)
    rg_dir = static['roadgraph_dir_world'].numpy().reshape(-1, 3)
    rg_type = static['roadgraph_type'].numpy().reshape(-1)
    rg_valid = static['roadgraph_valid'].numpy().reshape(-1)

    mask = (rg_valid > 0) & (np.abs(rg_xyz[:, 0] - center_x) < view_radius) & (np.abs(rg_xyz[:, 1] - center_y) < view_radius)
    if not np.any(mask):
        return

    pts = rg_xyz[mask, :2]
    dirs = rg_dir[mask, :2]
    types = rg_type[mask]

    dir_norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dir_norm = np.where(dir_norm < 1e-4, 1.0, dir_norm)
    dirs_u = dirs / dir_norm

    starts = pts - dirs_u * (seg_len / 2.0)
    ends = pts + dirs_u * (seg_len / 2.0)
    segments = np.stack([starts, ends], axis=1)

    type_groups: dict[int, list] = {}
    for seg, tp in zip(segments, types):
        type_groups.setdefault(int(tp), []).append(seg)

    for tp, segs in type_groups.items():
        color, lw, alpha, zorder, ls = RG_STYLE.get(tp, DEFAULT_RG_STYLE)
        lc = LineCollection(segs, colors=color, linewidths=lw, linestyles=ls, alpha=alpha, zorder=zorder)
        ax.add_collection(lc)


def predict_scene_v17(model, static: dict, window: dict, device: torch.device):
    packed = _scene_v13(static, window, max_targets=0, train=False)
    if packed is None:
        return None
    samples = {k: v.to(device) for k, v in packed.items()}
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_2, goals_2, logits_2, pred_1, goals_1 = model(
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

    return {
        "pred_2": pred_2.float().cpu().numpy(),  # [N, 6, 80, 2]
        "goals_2": goals_2.float().cpu().numpy(),
        "logits_2": logits_2.float().cpu().numpy(),  # [N, 6]
        "future": samples["future"].float().cpu().numpy(),  # [N, 80, 2]
        "future_valid": samples["future_valid"].bool().cpu().numpy(),
        "target_hist": samples["target_hist"].float().cpu().numpy(),
    }


def render_scene(pack_path: str, model, device: torch.device, out_dir: str, prefix: str):
    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    static = pack["static"]
    window = pack["windows"][0]
    preds = predict_scene_v17(model, static, window, device)
    if preds is None:
        return None, None

    hist_world = window['inputs']['agent_history_world'].numpy()
    hist_valid = window['inputs']['agent_history_valid'].numpy()
    sizes = window['inputs']['agent_size_m'].numpy()
    targets = window['targets']
    target_rows = targets['target_rows'].numpy()
    is_sdc = static['agent_is_sdc'].numpy()
    types = static['agent_types'].numpy()
    future_val = targets['agent_future_valid'].numpy()

    # Filter target rows exactly as _scene_v13 does
    rows = []
    for r in target_rows:
        r = int(r)
        if r < 0 or r >= hist_world.shape[0]:
            continue
        if is_sdc[r] or hist_valid[r, -1] == 0:
            continue
        if int(types[r]) not in TYPE_TO_INDEX:
            continue
        if int(future_val[r].sum()) < 20:
            continue
        rows.append(r)

    if len(rows) == 0:
        return None, None

    sdc_x = float(hist_world[0, -1, 0])
    sdc_y = float(hist_world[0, -1, 1])

    # Convert predicted trajectories from target-centric frame back to World coordinates
    n_targets = len(rows)
    pred_world = np.zeros((n_targets, 6, 80, 2), dtype=np.float32)
    gt_future_world = np.zeros((n_targets, 80, 2), dtype=np.float32)

    for i, r in enumerate(rows):
        orig = hist_world[r, -1, :2]
        yaw = float(hist_world[r, -1, 2])
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rot_mat = np.array([[cos_y, -sin_y], [sin_y, cos_y]])

        loc_tr = preds["pred_2"][i]
        for m in range(6):
            pred_world[i, m] = (loc_tr[m] @ rot_mat.T) + orig

        loc_gt = preds["future"][i]
        gt_future_world[i] = (loc_gt @ rot_mat.T) + orig

    # Top-1 chosen mode per target
    top1_modes = np.argmax(preds["logits_2"], axis=-1)

    # 1. Render High-Resolution HD Map Snapshot PNG
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150, facecolor="#090d16")
    ax.set_facecolor("#090d16")
    draw_vector_roadgraph(ax, static, sdc_x, sdc_y, view_radius=60.0)

    # Draw all surrounding agents and targets
    for i, r in enumerate(rows):
        col = PALETTE[i % len(PALETTE)]
        # Target current position
        cx, cy, heading = hist_world[r, -1, 0], hist_world[r, -1, 1], hist_world[r, -1, 2]
        corners = get_vehicle_corners(cx, cy, heading, sizes[r, 0], sizes[r, 1])
        poly = patches.Polygon(corners, closed=True, facecolor=col, edgecolor="#ffffff", linewidth=1.5, alpha=0.95, zorder=10)
        ax.add_patch(poly)

        # Past trajectory
        p_mask = hist_valid[r] > 0
        if np.any(p_mask):
            ax.plot(hist_world[r, p_mask, 0], hist_world[r, p_mask, 1], color=col, linewidth=2.0, alpha=0.7, zorder=9)

        # 6 predicted trajectories (thin)
        for m in range(6):
            if m != top1_modes[i]:
                ax.plot(pred_world[i, m, :, 0], pred_world[i, m, :, 1], color=col, linewidth=1.2, alpha=0.35, linestyle=":", zorder=7)

        # Top-1 chosen trajectory (bold bright)
        top_m = top1_modes[i]
        ax.plot(pred_world[i, top_m, :, 0], pred_world[i, top_m, :, 1], color=col, linewidth=3.2, alpha=0.95, zorder=12,
                label=f"Target {i+1} Pred (Top-1)" if i == 0 else "")

        # Ground truth future (dashed white)
        gt_valid = preds["future_valid"][i]
        if np.any(gt_valid):
            ax.plot(gt_future_world[i, gt_valid, 0], gt_future_world[i, gt_valid, 1], color="#ffffff", linewidth=2.0, linestyle="--", alpha=0.85, zorder=11,
                    label="Ground Truth" if i == 0 else "")

    ax.set_xlim(sdc_x - 45.0, sdc_x + 45.0)
    ax.set_ylim(sdc_y - 45.0, sdc_y + 45.0)
    ax.set_aspect("equal")
    ax.axis("off")

    title_txt = f"V17 X-Large Graph Transformer | Multi-Agent Prediction (Val Scene: {prefix})"
    ax.text(0.5, 0.97, title_txt, transform=ax.transAxes, color="#f8fafc", fontsize=13, fontweight="bold", ha="center", va="top")
    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=10)

    png_path = os.path.join(out_dir, f"{prefix}_hdmap.png")
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1, facecolor="#090d16")
    plt.close(fig)

    # 2. Render 8.0-Second Animated Rollout GIF (16 frames, step=5 -> 0.5s interval)
    frames = []
    step_indices = list(range(0, 80, 5))
    for s_idx in step_indices:
        fig_f, ax_f = plt.subplots(figsize=(8, 8), dpi=100, facecolor="#090d16")
        ax_f.set_facecolor("#090d16")
        draw_vector_roadgraph(ax_f, static, sdc_x, sdc_y, view_radius=55.0)

        t_sec = (s_idx + 1) * 0.1
        for i, r in enumerate(rows):
            col = PALETTE[i % len(PALETTE)]
            top_m = top1_modes[i]

            px, py = pred_world[i, top_m, s_idx, 0], pred_world[i, top_m, s_idx, 1]
            if s_idx > 0:
                h_yaw = math.atan2(py - pred_world[i, top_m, s_idx-1, 1], px - pred_world[i, top_m, s_idx-1, 0])
            else:
                h_yaw = float(hist_world[r, -1, 2])

            corners = get_vehicle_corners(px, py, h_yaw, sizes[r, 0], sizes[r, 1])
            poly = patches.Polygon(corners, closed=True, facecolor=col, edgecolor="#ffffff", linewidth=1.2, alpha=0.9, zorder=12)
            ax_f.add_patch(poly)

            ax_f.plot(pred_world[i, top_m, :s_idx+1, 0], pred_world[i, top_m, :s_idx+1, 1], color=col, linewidth=2.8, alpha=0.9, zorder=10)

            gt_valid = preds["future_valid"][i]
            if gt_valid[s_idx]:
                ax_f.plot(gt_future_world[i, :s_idx+1, 0], gt_future_world[i, :s_idx+1, 1], color="#ffffff", linewidth=1.8, linestyle="--", alpha=0.75, zorder=9)

        ax_f.set_xlim(sdc_x - 40.0, sdc_x + 40.0)
        ax_f.set_ylim(sdc_y - 40.0, sdc_y + 40.0)
        ax_f.set_aspect("equal")
        ax_f.axis("off")

        ax_f.text(0.5, 0.96, f"V17 X-Large Rollout | t = +{t_sec:.1f}s / 8.0s", transform=ax_f.transAxes, color="#f8fafc", fontsize=12, fontweight="bold", ha="center")
        fig_f.canvas.draw()
        rgba = np.asarray(fig_f.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("RGB"))
        plt.close(fig_f)

    gif_path = os.path.join(out_dir, f"{prefix}_rollout.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=150, loop=0)
    print(f"-> Rendered {png_path} and {gif_path}", flush=True)
    return png_path, gif_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" RENDERING HIGH-FIDELITY V17 X-LARGE SHOWCASE VISUALIZATIONS & GIFS")
    print("=" * 80)

    model = MotionPredictorV17(hidden=768, modes=6, nhead=12, dropout=0.1).to(device)
    ckpt_path = r"E:\motion_prediction\checkpoints\v17_xlarge\swa_model.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = r"E:\motion_prediction\checkpoints\v17_xlarge\best_error_score.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_compatible_state(model, ckpt.get("model_state", ckpt))
    model.eval()
    print(f"-> Loaded V17 X-Large Model from: {ckpt_path}", flush=True)

    val_paths = sorted(glob.glob(r"E:\motion_prediction\data\processed\prediction_pt_85k_v2\val\*.pt"))
    if not val_paths:
        val_paths = sorted(glob.glob(r"E:\motion_prediction\data\processed\prediction_pt_85k_v2\train\*.pt"))
    print(f"-> Found {len(val_paths)} scene files", flush=True)

    random.seed(42)
    sample_scenes = random.sample(val_paths, min(10, len(val_paths)))

    vis_dir = r"E:\motion_prediction\visualizations"
    rand_dir = os.path.join(vis_dir, "random_10_scenes")
    os.makedirs(rand_dir, exist_ok=True)

    rendered_pngs = []
    for idx, sp in enumerate(sample_scenes, start=1):
        scene_id = os.path.splitext(os.path.basename(sp))[0]
        prefix = f"scene_{idx:02d}_{scene_id[:16]}"
        png_p, gif_p = render_scene(sp, model, device, rand_dir, prefix)
        if png_p is not None:
            rendered_pngs.append(png_p)

        if idx == 1 and png_p is not None:
            im = Image.open(png_p)
            im.save(os.path.join(vis_dir, "prediction_v1_multi_target_hdmap.png"))
            im.save(os.path.join(vis_dir, "prediction_v1_multi_target_50m.png"))
            gif_im = Image.open(gif_p)
            gif_im.save(os.path.join(vis_dir, "prediction_v1_val_rollout.gif"))
            print("-> Updated root showcase GIF and PNG with V17 X-Large predictions!")

    if len(rendered_pngs) >= 10:
        collage = Image.new("RGB", (5 * 600, 2 * 600), "#090d16")
        for i, p in enumerate(rendered_pngs[:10]):
            r, c = i // 5, i % 5
            im = Image.open(p).resize((600, 600), Image.Resampling.LANCZOS)
            collage.paste(im, (c * 600, r * 600))
        collage_path = os.path.join(rand_dir, "summary_10_scenes_grid.png")
        collage.save(collage_path, quality=95)
        print(f"-> Created 2x5 summary grid collage at {collage_path}")

    print("ALL V17 VISUALIZATIONS AND GIFS SUCCESSFULLY GENERATED!")


if __name__ == "__main__":
    main()
