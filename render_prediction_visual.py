#!/usr/bin/env python3
"""Render high-resolution visualization for Motion Prediction Model V1 predictions."""
import os, sys, glob, json, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_motion_prediction_v1 import MotionPredictor, TYPE_TO_INDEX, local_xy, local_vec

def local_to_world(xy_local: torch.Tensor, origin: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Transform local target coordinates back to world coordinates."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    x_world = origin[..., 0] + xy_local[..., 0] * c - xy_local[..., 1] * s
    y_world = origin[..., 1] + xy_local[..., 0] * s + xy_local[..., 1] * c
    return torch.stack([x_world, y_world], dim=-1)

def render_scene_prediction(ckpt_path: str, data_root: str, output_png: str, output_gif: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Loading model checkpoint from: {ckpt_path}", flush=True)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = MotionPredictor(hidden=256, modes=6).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    val_files = sorted(glob.glob(os.path.join(data_root, 'val', '*.pt')))
    if not val_files:
        print(f"Error: No val files found in {data_root}/val", flush=True)
        return

    print(f"-> Found {len(val_files)} validation scene files. Rendering samples...", flush=True)

    # We will pick 4 representative target samples to render in a 2x2 grid PNG summary
    sample_targets = []

    for fpath in val_files[:5]:
        pack = torch.load(fpath, map_location='cpu', weights_only=False)
        static = pack['static']
        window = pack['windows'][-1] # use last window
        
        hist = window['inputs']['agent_history_world'].float()
        valid = window['inputs']['agent_history_valid'].bool()
        sizes = window['inputs']['agent_size_m'].float()
        types = static['agent_types'].long()
        sig = window['inputs']['signal_history_world_state'].float()
        sig_valid = window['inputs']['signal_history_valid'].bool()
        target_rows = window['targets']['target_rows'].long()
        target_types = window['targets']['target_types'].long()
        future_world = window['targets']['future_xy_world'].float()
        future_valid = window['targets']['future_valid'].bool()

        roadlines = static.get('roadgraph_polylines', None)

        for ti in range(len(target_rows)):
            ttype = int(target_types[ti])
            if ttype not in TYPE_TO_INDEX: continue
            row = int(target_rows[ti])
            if not valid[row, -1]: continue
            
            origin = hist[row, -1, :2]
            yaw = hist[row, -1, 2]
            
            target_pos = local_xy(hist[row, :, :2], origin, yaw)
            target_vel = local_vec(hist[row, :, 3:5], yaw)
            target_hist = torch.cat([target_pos, hist[row, :, 2:3] - yaw, target_vel], dim=-1).unsqueeze(0)
            
            cur_local = local_xy(hist[:, -1, :2], origin, yaw)
            cur_vel = local_vec(hist[:, -1, 3:5], yaw)
            candidate = valid[:, -1].clone()
            candidate[row] = False
            dist = torch.linalg.vector_norm(cur_local, dim=-1)
            dist[~candidate] = float('inf')
            
            neighbor_k = 16
            nidx = torch.topk(dist, k=min(neighbor_k, hist.shape[0]-1), largest=False).indices
            nfeat = torch.cat([cur_local[nidx], cur_vel[nidx], sizes[nidx], types[nidx, None].float()], dim=-1)
            if nfeat.shape[0] < neighbor_k:
                nfeat = torch.cat([nfeat, torch.zeros(neighbor_k-nfeat.shape[0], nfeat.shape[1])], dim=0)
            nfeat = nfeat.unsqueeze(0)

            signal_k = 4
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
            sfeat = sfeat.unsqueeze(0)
            
            type_idx = torch.tensor([TYPE_TO_INDEX[ttype]])

            with torch.no_grad():
                pred_local, goals_local, logits = model(
                    target_hist.to(device), nfeat.to(device), sfeat.to(device), type_idx.to(device)
                )
            
            probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
            pred_w = local_to_world(pred_local[0].cpu(), origin, yaw).numpy() # K=6, 80, 2
            gt_w = future_world[ti].numpy() # 80, 2
            gt_valid = future_valid[ti].numpy() # 80
            
            # calculate minADE6 and best mode
            if gt_valid.sum() > 0:
                gt_valid_m = gt_valid[:, None]
                dists = np.linalg.norm(pred_w - gt_w[None, :, :], axis=-1)
                ade6 = (dists * gt_valid_m.T).sum(axis=-1) / max(1, gt_valid.sum())
                best_mode = int(np.argmin(ade6))
                min_ade = float(ade6[best_mode])
                fde = float(np.linalg.norm(pred_w[best_mode, -1] - gt_w[-1]))
            else:
                best_mode = int(np.argmax(probs))
                min_ade = 0.0
                fde = 0.0

            sample_targets.append({
                'origin': origin.numpy(),
                'yaw': yaw.item(),
                'hist_w': hist[:, -1, :2].numpy(),
                'valid_h': valid[:, -1].numpy(),
                'sizes': sizes.numpy(),
                'target_row': row,
                'target_type': ttype,
                'roadlines': roadlines,
                'pred_w': pred_w,
                'probs': probs,
                'gt_w': gt_w,
                'gt_valid': gt_valid,
                'best_mode': best_mode,
                'min_ade': min_ade,
                'fde': fde,
                'fname': os.path.basename(fpath)
            })

            if len(sample_targets) >= 4:
                break
        if len(sample_targets) >= 4:
            break

    print(f"-> Collected {len(sample_targets)} sample targets for visual rendering.", flush=True)

    # 1. Generate 2x2 Summary PNG
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor='#0B0E14')
    fig.suptitle("Motion Prediction Model V1 — Multi-Modal K=6 Trajectory Forecast (8.0s)", 
                 fontsize=18, fontweight='bold', color='#E6EDF3', y=0.96)

    colors_modes = ['#00FF66', '#FF3366', '#FFCC00', '#00CCFF', '#CC66FF', '#FF9933']

    for idx, sample in enumerate(sample_targets[:4]):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor('#0B0E14')

        origin = sample['origin']
        roadlines = sample['roadlines']
        
        # Plot HD Map roadlines
        if roadlines is not None:
            for rl in roadlines:
                pts = rl if isinstance(rl, np.ndarray) else np.array(rl)
                if pts.ndim == 2 and pts.shape[0] > 1:
                    ax.plot(pts[:, 0], pts[:, 1], color='#1E2638', linewidth=1.0, alpha=0.7)

        # Plot surrounding vehicles
        hist_w = sample['hist_w']
        valid_h = sample['valid_h']
        sizes = sample['sizes']
        target_row = sample['target_row']

        for ai in range(len(hist_w)):
            if not valid_h[ai]: continue
            x, y = hist_w[ai]
            l, w = sizes[ai][:2] if len(sizes[ai]) >= 2 else (4.5, 2.0)
            if ai == target_row:
                color = '#FF9900'
                zorder = 5
            else:
                color = '#38445D'
                zorder = 3
            rect = patches.Rectangle((x - l/2, y - w/2), l, w, color=color, alpha=0.8, zorder=zorder)
            ax.add_patch(rect)

        # Plot Ground Truth future trajectory
        gt_w = sample['gt_w']
        gt_valid = sample['gt_valid']
        if gt_valid.sum() > 0:
            ax.plot(gt_w[gt_valid, 0], gt_w[gt_valid, 1], '--', color='#00E5FF', linewidth=2.5, label='Ground Truth (8.0s)', zorder=6)
            ax.scatter(gt_w[-1, 0], gt_w[-1, 1], marker='o', color='#00E5FF', s=40, zorder=7)

        # Plot Predicted 6 Modes
        pred_w = sample['pred_w'] # 6, 80, 2
        probs = sample['probs']
        best_mode = sample['best_mode']

        # Sort modes by probability ascending so highest probability is drawn on top
        sorted_modes = np.argsort(probs)
        for m in sorted_modes:
            p = probs[m]
            c = colors_modes[m % len(colors_modes)]
            is_best = (m == best_mode)
            lw = 3.0 if is_best else 1.2
            alpha = 0.95 if is_best else max(0.2, p * 1.5)
            lbl = f"Mode {m+1} (P={p*100:.1f}%)" if is_best else None
            ax.plot(pred_w[m, :, 0], pred_w[m, :, 1], color=c, linewidth=lw, alpha=alpha, label=lbl, zorder=8 if is_best else 7)
            ax.scatter(pred_w[m, -1, 0], pred_w[m, -1, 1], marker='*', color=c, s=100 if is_best else 40, zorder=9 if is_best else 7)

        # Focus zoom on target vehicle
        ax.set_xlim(origin[0] - 45, origin[0] + 45)
        ax.set_ylim(origin[1] - 45, origin[1] + 45)
        ax.set_aspect('equal')
        ax.axis('off')

        # Add Card Overlay info
        t_type_str = "Vehicle" if sample['target_type'] == 1 else "Pedestrian" if sample['target_type'] == 2 else "Cyclist"
        card_text = (
            f"Target Agent #{sample['target_row']} [{t_type_str}]\n"
            f"minADE@8s: {sample['min_ade']:.2f}m | minFDE@8s: {sample['fde']:.2f}m\n"
            f"Best Mode: #{best_mode+1} (Prob: {probs[best_mode]*100:.1f}%)"
        )
        ax.text(0.03, 0.93, card_text, transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#161B22', edgecolor='#30363D', alpha=0.9),
                color='#E6EDF3', fontsize=11, fontweight='medium', verticalalignment='top')
        ax.legend(loc='lower right', facecolor='#161B22', edgecolor='#30363D', labelcolor='#E6EDF3', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_png, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"✅ Saved Multi-Scene Summary PNG to: {output_png}", flush=True)

    # 2. Generate Rollout GIF for Target 0 over 80 timesteps (8 seconds)
    print("-> Rendering 8.0s Step-by-Step Rollout Animation GIF...", flush=True)
    target0 = sample_targets[0]
    frames = []

    for step in range(0, 81, 2): # every 0.2s (2 steps)
        fig, ax = plt.subplots(figsize=(9, 8), facecolor='#0B0E14')
        ax.set_facecolor('#0B0E14')

        origin = target0['origin']
        roadlines = target0['roadlines']
        
        if roadlines is not None:
            for rl in roadlines:
                pts = rl if isinstance(rl, np.ndarray) else np.array(rl)
                if pts.ndim == 2 and pts.shape[0] > 1:
                    ax.plot(pts[:, 0], pts[:, 1], color='#1E2638', linewidth=1.0, alpha=0.7)

        hist_w = target0['hist_w']
        valid_h = target0['valid_h']
        sizes = target0['sizes']
        target_row = target0['target_row']

        for ai in range(len(hist_w)):
            if not valid_h[ai]: continue
            x, y = hist_w[ai]
            l, w = sizes[ai][:2] if len(sizes[ai]) >= 2 else (4.5, 2.0)
            color = '#FF9900' if ai == target_row else '#38445D'
            rect = patches.Rectangle((x - l/2, y - w/2), l, w, color=color, alpha=0.8, zorder=5)
            ax.add_patch(rect)

        # Plot Ground Truth up to step
        gt_w = target0['gt_w']
        gt_valid = target0['gt_valid']
        if step > 0 and gt_valid[:step].sum() > 0:
            ax.plot(gt_w[:step, 0], gt_w[:step, 1], '--', color='#00E5FF', linewidth=2.5, label='Ground Truth Path', zorder=6)
            ax.scatter(gt_w[step-1, 0], gt_w[step-1, 1], color='#00E5FF', s=50, zorder=7)

        # Plot Predicted Trajectories up to step
        pred_w = target0['pred_w'] # 6, 80, 2
        probs = target0['probs']
        best_mode = target0['best_mode']

        for m in range(6):
            if step == 0: continue
            p = probs[m]
            c = colors_modes[m % len(colors_modes)]
            is_best = (m == best_mode)
            lw = 3.2 if is_best else 1.2
            alpha = 0.95 if is_best else max(0.2, p * 1.5)
            ax.plot(pred_w[m, :step, 0], pred_w[m, :step, 1], color=c, linewidth=lw, alpha=alpha, zorder=8 if is_best else 7)
            ax.scatter(pred_w[m, step-1, 0], pred_w[m, step-1, 1], marker='*', color=c, s=80 if is_best else 30, zorder=9 if is_best else 7)

        ax.set_xlim(origin[0] - 40, origin[0] + 40)
        ax.set_ylim(origin[1] - 40, origin[1] + 40)
        ax.set_aspect('equal')
        ax.axis('off')

        title_text = f"MOTION PREDICTION MODEL V1 — ROLLOUT T={step*0.1:.1f}s / 8.0s\nTarget Agent #{target_row} | minADE@8s: {target0['min_ade']:.2f}m | Best Mode #{best_mode+1} ({probs[best_mode]*100:.1f}%)"
        ax.set_title(title_text, color='#E6EDF3', fontsize=12, fontweight='bold', pad=12)

        fig.canvas.draw()
        rgba_array = np.asarray(fig.canvas.buffer_rgba())
        frame_img = Image.fromarray(rgba_array).convert("RGB")
        frames.append(frame_img)
        plt.close(fig)

    frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"✅ Saved Animated Rollout GIF to: {output_gif}", flush=True)

if __name__ == '__main__':
    ckpt_path = "/mnt/c/Users/andy0/Downloads/behavior_stack_planner/motion_prediction_model_v1/runs/full_30_epochs/checkpoints/best_minade6.pth"
    data_root = "/mnt/c/Users/andy0/Downloads/behavior_stack_planner/data/processed/prediction_pt"
    out_png = "/mnt/c/Users/andy0/Downloads/behavior_stack_planner/outputs/visualizations/prediction_v1_val_sample.png"
    out_gif = "/mnt/c/Users/andy0/Downloads/behavior_stack_planner/outputs/visualizations/prediction_v1_val_rollout.gif"
    os.makedirs("/mnt/c/Users/andy0/Downloads/behavior_stack_planner/outputs/visualizations", exist_ok=True)
    render_scene_prediction(ckpt_path, data_root, out_png, out_gif)
