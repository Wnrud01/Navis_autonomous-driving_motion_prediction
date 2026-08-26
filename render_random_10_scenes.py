#!/usr/bin/env python3
"""Batch Visualizer: Renders 10 Random Diverse Validation Scenes on Vector HD Maps.

Generates:
1. 10 Individual High-Resolution Vector HD Map 50m Prediction PNGs.
2. 10 Individual 8.0-Second Animated Rollout GIFs.
3. 1 Comprehensive 2x5 Multi-Scene Grid Summary PNG.
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
from train_motion_prediction_v1 import MotionPredictor, TYPE_TO_INDEX, local_xy, local_vec

RG_STYLE = {
    1:  ("#64748b", 0.8, 0.40, 1, "--"),
    2:  ("#64748b", 0.8, 0.40, 1, "--"),
    3:  ("#64748b", 0.7, 0.35, 1, "--"),
    6:  ("#eab308", 1.2, 0.75, 2, "-"),
    7:  ("#eab308", 1.4, 0.80, 2, "-"),
    8:  ("#eab308", 1.6, 0.85, 2, "-"),
    9:  ("#e2e8f0", 1.0, 0.60, 2, "--"),
    10: ("#e2e8f0", 1.2, 0.70, 2, "-"),
    11: ("#f1f5f9", 1.5, 0.85, 2, "-"),
    12: ("#eab308", 1.2, 0.65, 2, "--"),
    15: ("#334155", 2.2, 0.90, 3, "-"),
    16: ("#475569", 1.8, 0.75, 3, "-"),
    17: ("#e2e8f0", 1.2, 0.65, 2, "-"),
    18: ("#cbd5e1", 0.9, 0.50, 2, "--"),
    19: ("#fbbf24", 1.2, 0.60, 2, "-"),
    20: ("#f87171", 1.0, 0.50, 2, "-"),
}
DEFAULT_RG_STYLE = ("#475569", 0.8, 0.50, 1, "-")

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

def draw_vector_roadgraph(ax, static: dict, sdc_x: float, sdc_y: float, view_radius: float = 80.0, seg_len: float = 0.8):
    if 'roadgraph_xyz_world' not in static or 'roadgraph_dir_world' not in static:
        return
    rg_xyz = static['roadgraph_xyz_world'].numpy().reshape(-1, 3)
    rg_dir = static['roadgraph_dir_world'].numpy().reshape(-1, 3)
    rg_type = static['roadgraph_type'].numpy().reshape(-1)
    rg_valid = static['roadgraph_valid'].numpy().reshape(-1)

    valid_mask = (rg_valid == 1) & (np.abs(rg_xyz[:, 0]) > 1.0) & (np.abs(rg_xyz[:, 1]) > 1.0)
    dist_to_sdc = np.hypot(rg_xyz[:, 0] - sdc_x, rg_xyz[:, 1] - sdc_y)
    valid_mask = valid_mask & (dist_to_sdc < view_radius + 25)

    pts = rg_xyz[valid_mask, :2]
    dirs = rg_dir[valid_mask, :2]
    types = rg_type[valid_mask]

    norms = np.hypot(dirs[:, 0], dirs[:, 1])
    norms[norms == 0] = 1.0
    dirs = dirs / norms[:, None]

    unique_types = np.unique(types)
    for t_val in unique_types:
        t_mask = (types == t_val)
        t_pts = pts[t_mask]
        t_dirs = dirs[t_mask]

        p1 = t_pts - t_dirs * seg_len
        p2 = t_pts + t_dirs * seg_len
        segments = np.stack([p1, p2], axis=1)

        style = RG_STYLE.get(int(t_val), DEFAULT_RG_STYLE)
        color, lw, alpha, zord, ls = style

        lc = LineCollection(segments, colors=color, linewidths=lw, alpha=alpha, zorder=zord, linestyles=ls)
        ax.add_collection(lc)

def local_to_world(xy_local: torch.Tensor, origin: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    x_world = origin[..., 0] + xy_local[..., 0] * c - xy_local[..., 1] * s
    y_world = origin[..., 1] + xy_local[..., 0] * s + xy_local[..., 1] * c
    return torch.stack([x_world, y_world], dim=-1)

def run_inference_on_scene(model, pack, device, radius_m=50.0):
    w = pack['windows'][-1]
    static = pack['static']
    
    hist = w['inputs']['agent_history_world'].float()
    valid = w['inputs']['agent_history_valid'].bool()
    sizes = w['inputs']['agent_size_m'].float()
    types = static['agent_types'].long()
    sig = w['inputs']['signal_history_world_state'].float()
    sig_valid = w['inputs']['signal_history_valid'].bool()
    
    sdc_pos = hist[0, -1, :2]
    sdc_yaw = hist[0, -1, 2]
    dists_to_sdc = torch.linalg.vector_norm(hist[:, -1, :2] - sdc_pos, dim=-1)
    
    target_indices = torch.where(valid[:, -1] & (dists_to_sdc <= radius_m))[0].tolist()
    if len(target_indices) == 0:
        return static, sdc_pos, []

    batch_t_hist, batch_neighbors, batch_signals, batch_type_idx = [], [], [], []
    meta_targets = []
    neighbor_k = 16
    signal_k = 4

    for row in target_indices:
        origin = hist[row, -1, :2]
        yaw = hist[row, -1, 2]
        ttype = int(types[row])
        ttype_idx = TYPE_TO_INDEX.get(ttype, 0)
        
        target_pos = local_xy(hist[row, :, :2], origin, yaw)
        target_vel = local_vec(hist[row, :, 3:5], yaw)
        t_hist = torch.cat([target_pos, hist[row, :, 2:3] - yaw, target_vel], dim=-1)
        
        cur_local = local_xy(hist[:, -1, :2], origin, yaw)
        cur_vel = local_vec(hist[:, -1, 3:5], yaw)
        candidate = valid[:, -1].clone()
        candidate[row] = False
        dist = torch.linalg.vector_norm(cur_local, dim=-1)
        dist[~candidate] = float('inf')
        
        nidx = torch.topk(dist, k=min(neighbor_k, hist.shape[0]-1), largest=False).indices
        nfeat = torch.cat([cur_local[nidx], cur_vel[nidx], sizes[nidx], types[nidx, None].float()], dim=-1)
        if nfeat.shape[0] < neighbor_k:
            nfeat = torch.cat([nfeat, torch.zeros(neighbor_k-nfeat.shape[0], nfeat.shape[1])], dim=0)

        if sig.shape[0]:
            sig_local = local_xy(sig[:, -1, :2], origin, yaw)
            sdist = torch.linalg.vector_norm(sig_local, dim=-1)
            sdist[~sig_valid[:, -1]] = float('inf')
            sidx = torch.topk(sdist, k=min(signal_k, sig.shape[0]), largest=False).indices
            changes = (sig[sidx, 1:, 2] != sig[sidx, :-1, 2]).float().sum(dim=1, keepdim=True)
            sfeat = torch.cat([sig_local[sidx], sig[sidx, -1, 2:3], changes], dim=-1)
            if sfeat.shape[0] < signal_k:
                sfeat = torch.cat([sfeat, torch.zeros(signal_k-sfeat.shape[0], 4)], dim=0)
        else:
            sfeat = torch.zeros(signal_k, 4)

        batch_t_hist.append(t_hist)
        batch_neighbors.append(nfeat)
        batch_signals.append(sfeat)
        batch_type_idx.append(ttype_idx)

        meta_targets.append({
            'row': row,
            'origin': origin,
            'yaw': yaw,
            'type': ttype,
            'is_sdc': (row == 0),
            'length': sizes[row][0].item() if len(sizes[row]) >= 1 else 4.6,
            'width': sizes[row][1].item() if len(sizes[row]) >= 2 else 2.1
        })

    inp_t_hist = torch.stack(batch_t_hist).to(device)
    inp_neighbors = torch.stack(batch_neighbors).to(device)
    inp_signals = torch.stack(batch_signals).to(device)
    inp_type_idx = torch.tensor(batch_type_idx).to(device)

    with torch.no_grad():
        pred_local, goals_local, logits = model(inp_t_hist, inp_neighbors, inp_signals, inp_type_idx)

    probs_all = torch.softmax(logits, dim=-1).cpu()

    all_predictions = []
    for idx, meta in enumerate(meta_targets):
        origin = meta['origin']
        yaw = meta['yaw']
        probs = probs_all[idx].numpy()
        best_m = int(np.argmax(probs))
        
        pred_w = local_to_world(pred_local[idx].cpu(), origin, yaw).numpy()
        
        all_predictions.append({
            'row': meta['row'],
            'origin': origin.numpy(),
            'yaw': yaw.item(),
            'type': meta['type'],
            'is_sdc': meta['is_sdc'],
            'length': meta['length'],
            'width': meta['width'],
            'pred_w': pred_w,
            'probs': probs,
            'best_m': best_m,
            'best_traj': pred_w[best_m]
        })

    return static, sdc_pos, all_predictions

def render_scene(static, sdc_pos, all_predictions, out_png, out_gif, scene_idx, scene_name, radius_m=50.0):
    plt.style.use('dark_background')
    
    # 1. Render HD Map PNG
    fig, ax = plt.subplots(figsize=(11, 10), dpi=130)
    fig.patch.set_facecolor('#0B0E14')
    ax.set_facecolor('#0F172A')

    draw_vector_roadgraph(ax, static, sdc_pos[0].item(), sdc_pos[1].item(), view_radius=radius_m+20)

    circle = patches.Circle((sdc_pos[0].item(), sdc_pos[1].item()), radius_m, 
                            color='#00E5FF', fill=False, linestyle='--', linewidth=1.5, alpha=0.45, label='50m Radius')
    ax.add_patch(circle)

    for idx, target in enumerate(all_predictions):
        row = target['row']
        x, y = target['origin']
        yaw = target['yaw']
        is_sdc = target['is_sdc']
        l, w_size = max(target['length'], 3.8), max(target['width'], 1.8)
        color = '#FFCC00' if is_sdc else PALETTE[idx % len(PALETTE)]
        
        corners = get_vehicle_corners(x, y, yaw, length=l, width=w_size)
        poly = patches.Polygon(corners, facecolor=color, edgecolor='#FFFFFF', alpha=0.85, linewidth=1.2, zorder=6)
        ax.add_patch(poly)

        best_traj = target['best_traj']
        best_p = target['probs'][target['best_m']]
        
        lbl = f"SDC ({best_p*100:.1f}%)" if is_sdc else None
        ax.plot(best_traj[:, 0], best_traj[:, 1], color=color, linewidth=2.6 if is_sdc else 1.8, alpha=0.95, label=lbl, zorder=7)
        ax.scatter(best_traj[-1, 0], best_traj[-1, 1], marker='*', color=color, s=110 if is_sdc else 50, zorder=8)

        ax.text(x, y + 2.2, f"#{row}", color='#FFFFFF', fontsize=8, fontweight='bold', ha='center', zorder=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1E293B', edgecolor='none', alpha=0.7))

    ax.set_xlim(sdc_pos[0].item() - 60, sdc_pos[0].item() + 60)
    ax.set_ylim(sdc_pos[1].item() - 60, sdc_pos[1].item() + 60)
    ax.set_aspect('equal')
    ax.axis('off')

    title_str = f"SCENE #{scene_idx:02d} [{scene_name}] — 50m MULTI-TARGET PREDICTION (N={len(all_predictions)})\nVector HD Map Network + 8.0s Trajectory Forecast"
    ax.set_title(title_str, color='#E6EDF3', fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='upper right', facecolor='#1E293B', edgecolor='#334155', labelcolor='#E6EDF3', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

    # 2. Render 8.0s Step-by-Step Rollout Animation GIF
    frames = []
    for step in range(0, 81, 3): # 27 frames for crisp, smooth, and fast rendering
        fig, ax = plt.subplots(figsize=(9, 8), dpi=100)
        fig.patch.set_facecolor('#0B0E14')
        ax.set_facecolor('#0F172A')

        draw_vector_roadgraph(ax, static, sdc_pos[0].item(), sdc_pos[1].item(), view_radius=radius_m+20)

        circle = patches.Circle((sdc_pos[0].item(), sdc_pos[1].item()), radius_m, 
                                color='#00E5FF', fill=False, linestyle='--', linewidth=1.5, alpha=0.45)
        ax.add_patch(circle)

        for idx, target in enumerate(all_predictions):
            row = target['row']
            x, y = target['origin']
            yaw = target['yaw']
            is_sdc = target['is_sdc']
            l, w_size = max(target['length'], 3.8), max(target['width'], 1.8)
            color = '#FFCC00' if is_sdc else PALETTE[idx % len(PALETTE)]

            corners = get_vehicle_corners(x, y, yaw, length=l, width=w_size)
            poly = patches.Polygon(corners, facecolor=color, edgecolor='#FFFFFF', alpha=0.85, linewidth=1.2, zorder=6)
            ax.add_patch(poly)

            if step > 0:
                best_traj = target['best_traj'][:step]
                ax.plot(best_traj[:, 0], best_traj[:, 1], color=color, linewidth=2.6 if is_sdc else 1.8, alpha=0.95, zorder=7)
                ax.scatter(best_traj[-1, 0], best_traj[-1, 1], marker='*', color=color, s=90 if is_sdc else 40, zorder=8)

            ax.text(x, y + 2.2, f"#{row}", color='#FFFFFF', fontsize=8, fontweight='bold', ha='center', zorder=9)

        ax.set_xlim(sdc_pos[0].item() - 60, sdc_pos[0].item() + 60)
        ax.set_ylim(sdc_pos[1].item() - 60, sdc_pos[1].item() + 60)
        ax.set_aspect('equal')
        ax.axis('off')

        rollout_title = f"SCENE #{scene_idx:02d} [{scene_name}] — ROLLOUT T={step*0.1:.1f}s / 8.0s\nVector HD Map Network + Concurrent 50m Multi-Target Forecasting (N={len(all_predictions)})"
        ax.set_title(rollout_title, color='#E6EDF3', fontsize=11, fontweight='bold', pad=10)

        fig.canvas.draw()
        rgba_array = np.asarray(fig.canvas.buffer_rgba())
        frame_img = Image.fromarray(rgba_array).convert("RGB")
        frames.append(frame_img)
        plt.close(fig)

    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=120, loop=0)

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_minade6.pth")
    data_root = r"C:\Users\andy0\Downloads\behavior_stack_planner\data\processed\prediction_pt"
    
    out_dir = os.path.join(base_dir, "visualizations", "random_10_scenes")
    assets_dir = os.path.join(base_dir, "assets", "random_10_scenes")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    print(f"-> Loading model from: {ckpt_path}", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MotionPredictor(hidden=256, modes=6).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    val_files = sorted(glob.glob(os.path.join(data_root, 'val', '*.pt')))
    print(f"-> Found {len(val_files)} total validation scenes in dataset.", flush=True)

    # Randomly select 10 diverse scenes with active traffic
    random.seed(42)
    sample_indices = random.sample(range(len(val_files)), 80)
    
    chosen_scenes = []
    for s_idx in sample_indices:
        fpath = val_files[s_idx]
        pack = torch.load(fpath, map_location='cpu', weights_only=False)
        w = pack['windows'][-1]
        hist = w['inputs']['agent_history_world'].float()
        valid = w['inputs']['agent_history_valid'].bool()
        sdc_pos = hist[0, -1, :2]
        dists = torch.linalg.vector_norm(hist[:, -1, :2] - sdc_pos, dim=-1)
        active_in_radius = (valid[:, -1] & (dists <= 50.0)).sum().item()
        if active_in_radius >= 6:
            chosen_scenes.append((fpath, active_in_radius))
            if len(chosen_scenes) == 10:
                break

    print(f"-> Selected {len(chosen_scenes)} diverse active traffic scenes for rendering.\n", flush=True)

    rendered_pngs = []
    for i, (fpath, agent_count) in enumerate(chosen_scenes, start=1):
        s_name = Path(fpath).stem
        print(f"[{i:02d}/10] Processing Scene: {s_name} ({agent_count} targets within 50m)...", flush=True)
        
        pack = torch.load(fpath, map_location='cpu', weights_only=False)
        static, sdc_pos, preds = run_inference_on_scene(model, pack, device, radius_m=50.0)
        
        out_png = os.path.join(out_dir, f"scene_{i:02d}_{s_name}.png")
        out_gif = os.path.join(out_dir, f"scene_{i:02d}_{s_name}.gif")
        
        render_scene(static, sdc_pos, preds, out_png, out_gif, scene_idx=i, scene_name=s_name, radius_m=50.0)
        
        # Copy to assets/
        import shutil
        shutil.copy(out_png, os.path.join(assets_dir, f"scene_{i:02d}_{s_name}.png"))
        shutil.copy(out_gif, os.path.join(assets_dir, f"scene_{i:02d}_{s_name}.gif"))
        rendered_pngs.append((out_png, s_name, len(preds)))
        print(f"       ✅ Saved PNG & GIF: scene_{i:02d}_{s_name}", flush=True)

    # 3. Render 2x5 Multi-Scene Collage Grid PNG
    print("\n-> Generating 2x5 Multi-Scene Grid Summary PNG...", flush=True)
    grid_fig, grid_axes = plt.subplots(2, 5, figsize=(25, 10), dpi=120)
    grid_fig.patch.set_facecolor('#0B0E14')
    
    for idx, (png_path, s_name, n_targets) in enumerate(rendered_pngs):
        row, col = divmod(idx, 5)
        ax = grid_axes[row, col]
        img = Image.open(png_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Scene #{idx+1:02d} | {s_name[:12]}... (N={n_targets})", color='#E6EDF3', fontsize=11, fontweight='bold')

    grid_fig.suptitle("MOTION PREDICTION MODEL V1 — 10 DIVERSE VALIDATION SCENES SHOWCASE (VECTOR HD MAP + 8.0s FORECAST)", 
                      color='#E6EDF3', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    grid_png_out = os.path.join(out_dir, "summary_10_scenes_grid.png")
    grid_fig.savefig(grid_png_out, dpi=120, facecolor=grid_fig.get_facecolor(), edgecolor='none')
    plt.close(grid_fig)
    
    shutil.copy(grid_png_out, os.path.join(assets_dir, "summary_10_scenes_grid.png"))
    print(f"✅ Saved 2x5 Multi-Scene Grid PNG to: {grid_png_out}", flush=True)
    print("🎉 All 10 random scene PNGs and animated GIFs rendered successfully!", flush=True)

if __name__ == "__main__":
    main()
