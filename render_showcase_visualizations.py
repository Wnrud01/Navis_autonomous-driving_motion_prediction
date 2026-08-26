#!/usr/bin/env python3
"""High-Fidelity Showcase Renderer for Motion Prediction Model V1.
Renders Vector HD Map multi-target prediction GIFs and multi-modal K=6 analysis PNGs.
"""
from __future__ import annotations
import os, sys, glob, json, math
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

        # Ground truth if available in future
        gt_w = None
        if 'agent_future_world' in w.get('targets', {}):
            gt_w = w['targets']['agent_future_world'][row].numpy()

        meta_targets.append({
            'row': row,
            'origin': origin,
            'yaw': yaw,
            'type': ttype,
            'is_sdc': (row == 0),
            'length': sizes[row][0].item() if len(sizes[row]) >= 1 else 4.6,
            'width': sizes[row][1].item() if len(sizes[row]) >= 2 else 2.1,
            'gt_w': gt_w
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
            'best_traj': pred_w[best_m],
            'gt_w': meta['gt_w']
        })

    return static, sdc_pos, all_predictions

def render_scene_gif_and_png(static, sdc_pos, all_predictions, out_png, out_gif, scene_title="MOTION PREDICTION", radius_m=50.0):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(13, 12), dpi=150)
    fig.patch.set_facecolor('#0B0E14')
    ax.set_facecolor('#0F172A')

    draw_vector_roadgraph(ax, static, sdc_pos[0].item(), sdc_pos[1].item(), view_radius=radius_m+20)

    circle = patches.Circle((sdc_pos[0].item(), sdc_pos[1].item()), radius_m, 
                            color='#00E5FF', fill=False, linestyle='--', linewidth=1.5, alpha=0.45, label='50m Evaluation Radius')
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
        
        lbl = f"SDC Forecast ({best_p*100:.1f}%)" if is_sdc else None
        ax.plot(best_traj[:, 0], best_traj[:, 1], color=color, linewidth=2.8 if is_sdc else 2.0, alpha=0.95, label=lbl, zorder=7)
        ax.scatter(best_traj[-1, 0], best_traj[-1, 1], marker='*', color=color, s=120 if is_sdc else 60, zorder=8)

        ax.text(x, y + 2.5, f"#{row}", color='#FFFFFF', fontsize=9, fontweight='bold', ha='center', zorder=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1E293B', edgecolor='none', alpha=0.7))

    ax.set_xlim(sdc_pos[0].item() - 60, sdc_pos[0].item() + 60)
    ax.set_ylim(sdc_pos[1].item() - 60, sdc_pos[1].item() + 60)
    ax.set_aspect('equal')
    ax.axis('off')

    title_str = f"{scene_title} (N={len(all_predictions)} Targets within 50m)\nVector HD Map Network + Concurrent 8.0s Multimodal Trajectory Forecasting"
    ax.set_title(title_str, color='#E6EDF3', fontsize=13, fontweight='bold', pad=14)
    ax.legend(loc='upper right', facecolor='#1E293B', edgecolor='#334155', labelcolor='#E6EDF3', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"✅ Saved PNG to: {out_png}", flush=True)

    print(f"-> Rendering 8.0s Step-by-Step Animated Rollout GIF for: {out_gif} ...", flush=True)
    frames = []

    for step in range(0, 81, 2):
        fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
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
                ax.plot(best_traj[:, 0], best_traj[:, 1], color=color, linewidth=2.8 if is_sdc else 2.0, alpha=0.95, zorder=7)
                ax.scatter(best_traj[-1, 0], best_traj[-1, 1], marker='*', color=color, s=100 if is_sdc else 50, zorder=8)

            ax.text(x, y + 2.5, f"#{row}", color='#FFFFFF', fontsize=9, fontweight='bold', ha='center', zorder=9)

        ax.set_xlim(sdc_pos[0].item() - 60, sdc_pos[0].item() + 60)
        ax.set_ylim(sdc_pos[1].item() - 60, sdc_pos[1].item() + 60)
        ax.set_aspect('equal')
        ax.axis('off')

        rollout_title = f"{scene_title} — ROLLOUT T={step*0.1:.1f}s / 8.0s\nVector HD Map Network + Concurrent 50m Multi-Target 8-Second Forecasting"
        ax.set_title(rollout_title, color='#E6EDF3', fontsize=12, fontweight='bold', pad=12)

        fig.canvas.draw()
        rgba_array = np.asarray(fig.canvas.buffer_rgba())
        frame_img = Image.fromarray(rgba_array).convert("RGB")
        frames.append(frame_img)
        plt.close(fig)

    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"✅ Saved Animated Rollout GIF to: {out_gif}", flush=True)

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_minade6.pth")
    data_root = r"C:\Users\andy0\Downloads\behavior_stack_planner\data\processed\prediction_pt"
    assets_dir = os.path.join(base_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "visualizations"), exist_ok=True)

    print(f"-> Loading model checkpoint from: {ckpt_path}", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MotionPredictor(hidden=256, modes=6).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    val_files = sorted(glob.glob(os.path.join(data_root, 'val', '*.pt')))
    print(f"-> Found {len(val_files)} validation scene files.", flush=True)

    # 1. Dense Multi-Target Scenario (Scene 0)
    pack0 = torch.load(val_files[0], map_location='cpu', weights_only=False)
    static0, sdc_pos0, preds0 = run_inference_on_scene(model, pack0, device, radius_m=50.0)
    
    out_png1 = os.path.join(assets_dir, "motion_prediction_hdmap_dense_50m.png")
    out_gif1 = os.path.join(assets_dir, "motion_prediction_hdmap_dense_50m.gif")
    render_scene_gif_and_png(static0, sdc_pos0, preds0, out_png1, out_gif1, scene_title="MOTION PREDICTION MODEL V1 — DENSE TRAFFIC CORRIDOR")

    # Also copy to visualizations/
    import shutil
    shutil.copy(out_png1, os.path.join(base_dir, "visualizations", "prediction_v1_multi_target_hdmap.png"))
    shutil.copy(out_gif1, os.path.join(base_dir, "visualizations", "prediction_v1_multi_target_hdmap.gif"))

    # 2. Find an Intersection / Turning Scenario with rich agent interaction
    selected_idx = 1
    for idx, fpath in enumerate(val_files[1:25], start=1):
        pack = torch.load(fpath, map_location='cpu', weights_only=False)
        w = pack['windows'][-1]
        hist = w['inputs']['agent_history_world'].float()
        valid = w['inputs']['agent_history_valid'].bool()
        sdc_pos = hist[0, -1, :2]
        dists = torch.linalg.vector_norm(hist[:, -1, :2] - sdc_pos, dim=-1)
        active_in_radius = (valid[:, -1] & (dists <= 50.0)).sum().item()
        if active_in_radius >= 8:
            selected_idx = idx
            break

    pack1 = torch.load(val_files[selected_idx], map_location='cpu', weights_only=False)
    static1, sdc_pos1, preds1 = run_inference_on_scene(model, pack1, device, radius_m=50.0)
    out_png2 = os.path.join(assets_dir, "motion_prediction_hdmap_intersection_50m.png")
    out_gif2 = os.path.join(assets_dir, "motion_prediction_hdmap_intersection_50m.gif")
    render_scene_gif_and_png(static1, sdc_pos1, preds1, out_png2, out_gif2, scene_title="MOTION PREDICTION MODEL V1 — MULTI-AGENT INTERACTION SCENE")

    print("🎉 All showcase GIFs and PNGs successfully rendered and saved to assets/!", flush=True)

if __name__ == "__main__":
    main()
