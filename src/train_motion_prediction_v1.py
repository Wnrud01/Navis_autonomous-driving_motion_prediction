#!/usr/bin/env python3
"""Target-centric K=6, 8-second motion-prediction baseline.

Reads only data/processed/prediction_pt and writes all artifacts under the
specified run directory. It does not import or alter the ego planning model.
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, random, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

TYPE_TO_INDEX = {1: 0, 2: 1, 3: 2}


def local_xy(xy: torch.Tensor, origin: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    d = xy - origin
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], dim=-1)


def local_vec(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], dim=-1)


class SceneWindowDataset(Dataset):
    def __init__(self, root: str, split: str, max_packs: int = 0):
        self.paths = sorted(glob.glob(os.path.join(root, split, '*.pt')))
        if max_packs:
            self.paths = self.paths[:max_packs]
        if not self.paths:
            raise FileNotFoundError(f'No {split} packs under {root}')
        self.items = [(path, wi) for path in self.paths for wi in range(3)]

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        path, window_index = self.items[index]
        pack = torch.load(path, map_location='cpu', weights_only=False)
        return pack['static'], pack['windows'][window_index]


def window_to_samples(batch: list[tuple[dict, dict]], max_targets: int, neighbor_k: int, signal_k: int, train: bool) -> dict[str, torch.Tensor]:
    outputs = {k: [] for k in ('target_hist','neighbors','signals','type_idx','future','future_valid')}
    for static, window in batch:
        hist = window['inputs']['agent_history_world'].float()  # N,11,5
        valid = window['inputs']['agent_history_valid'].bool()
        sizes = window['inputs']['agent_size_m'].float()
        types = static['agent_types'].long()
        sig = window['inputs']['signal_history_world_state'].float()
        sig_valid = window['inputs']['signal_history_valid'].bool()
        target_rows = window['targets']['target_rows'].long()
        target_types = window['targets']['target_types'].long()
        future_world = window['targets']['future_xy_world'].float()
        future_valid = window['targets']['future_valid'].bool()
        order = list(range(target_rows.numel()))
        if train and len(order) > max_targets:
            order = random.sample(order, max_targets)
        else:
            order = order[:max_targets]
        for ti in order:
            ttype = int(target_types[ti])
            if ttype not in TYPE_TO_INDEX: continue
            row = int(target_rows[ti])
            origin = hist[row, -1, :2]
            yaw = hist[row, -1, 2]
            target_pos = local_xy(hist[row, :, :2], origin, yaw)
            target_vel = local_vec(hist[row, :, 3:5], yaw)
            target_hist = torch.cat([target_pos, hist[row, :, 2:3] - yaw, target_vel], dim=-1)
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
                # x,y,current state, state-change count in the observation window
                changes = (sig[sidx, 1:, 2] != sig[sidx, :-1, 2]).float().sum(dim=1, keepdim=True)
                sfeat = torch.cat([sig_local[sidx], sig[sidx, -1, 2:3], changes], dim=-1)
                if sfeat.shape[0] < signal_k:
                    sfeat = torch.cat([sfeat, torch.zeros(signal_k-sfeat.shape[0], 4)], dim=0)
            else:
                sfeat = torch.zeros(signal_k, 4)
            future = local_xy(future_world[ti], origin, yaw)
            outputs['target_hist'].append(target_hist)
            outputs['neighbors'].append(nfeat)
            outputs['signals'].append(sfeat)
            outputs['type_idx'].append(torch.tensor(TYPE_TO_INDEX[ttype]))
            outputs['future'].append(future)
            outputs['future_valid'].append(future_valid[ti])
    if not outputs['future']:
        return None
    return {key: torch.stack(value) for key, value in outputs.items()}


class MotionPredictor(nn.Module):
    def __init__(self, hidden: int = 256, modes: int = 6):
        super().__init__()
        self.modes = modes
        self.type_emb = nn.Embedding(3, 16)
        self.target_encoder = nn.Sequential(nn.Flatten(), nn.Linear(55, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.neighbor_encoder = nn.Sequential(nn.Linear(8, hidden//2), nn.GELU(), nn.Linear(hidden//2, hidden))
        self.signal_encoder = nn.Sequential(nn.Linear(4, hidden//2), nn.GELU(), nn.Linear(hidden//2, hidden))
        self.fusion = nn.Sequential(nn.Linear(hidden*3+16, hidden*2), nn.GELU(), nn.LayerNorm(hidden*2), nn.Linear(hidden*2, hidden), nn.GELU())
        self.goal_head = nn.Linear(hidden, modes*2)
        self.delta_head = nn.Linear(hidden, modes*80*2)
        self.mode_head = nn.Linear(hidden, modes)

    def forward(self, target_hist, neighbors, signals, type_idx):
        b = target_hist.shape[0]
        target = self.target_encoder(target_hist)
        neigh = self.neighbor_encoder(neighbors).mean(dim=1)
        signal = self.signal_encoder(signals).mean(dim=1)
        h = self.fusion(torch.cat([target, neigh, signal, self.type_emb(type_idx)], dim=-1))
        goals = self.goal_head(h).view(b, self.modes, 2)
        deltas = self.delta_head(h).view(b, self.modes, 80, 2)
        raw = torch.cumsum(deltas, dim=2)
        t = torch.linspace(1/80, 1, 80, device=h.device, dtype=h.dtype).view(1,1,80,1)
        trajectories = raw + t * (goals.unsqueeze(2) - raw[:, :, -1:, :])
        return trajectories, goals, self.mode_head(h)


def loss_and_metrics(pred, goals, logits, future, valid):
    dist = torch.linalg.vector_norm(pred - future[:, None], dim=-1)
    mask = valid[:, None].float()
    denom = mask.sum(dim=-1).clamp_min(1.0)
    ade_modes = (dist * mask).sum(dim=-1) / denom
    best = ade_modes.argmin(dim=1)
    gather = best[:, None]
    ade1 = ade_modes[:, 0].mean()
    ade6 = ade_modes.min(dim=1).values.mean()
    traj_loss = ade_modes.gather(1, gather).mean()
    batch_index = torch.arange(future.shape[0], device=future.device)
    last_idx = valid.long().sum(dim=1).clamp_min(1) - 1
    gt_goal = future[batch_index, last_idx]
    chosen_goal = goals[batch_index, best]
    goal_loss = torch.linalg.vector_norm(chosen_goal - gt_goal, dim=-1).mean()
    mode_loss = nn.functional.cross_entropy(logits, best)
    chosen_traj = pred[batch_index, best]
    smooth = (chosen_traj[:, 2:] - 2*chosen_traj[:, 1:-1] + chosen_traj[:, :-2]).abs().mean()
    total = traj_loss + 0.5*goal_loss + 0.2*mode_loss + 0.05*smooth
    return total, {'loss': total.detach(), 'minade1': ade1.detach(), 'minade6': ade6.detach(), 'goal_error': goal_loss.detach()}


def run_epoch(model, loader, optimizer, device, args, train: bool, max_steps: int):
    model.train(train)
    totals, count = {}, 0
    start = time.time()
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, raw_batch in enumerate(loader, start=1):
            if max_steps and step > max_steps: break
            samples = window_to_samples(raw_batch, args.max_targets, args.neighbor_k, args.signal_k, train)
            if samples is None: continue
            samples = {k: v.to(device, non_blocking=True) for k,v in samples.items()}
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                pred, goals, logits = model(samples['target_hist'], samples['neighbors'], samples['signals'], samples['type_idx'])
                loss, metrics = loss_and_metrics(pred, goals, logits, samples['future'], samples['future_valid'])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            for key, value in metrics.items(): totals[key] = totals.get(key, 0.0) + float(value)
            count += 1
            if train and step % args.log_every == 0:
                print(json.dumps({'step': step, **{k: round(v/count,4) for k,v in totals.items()}}, ensure_ascii=False), flush=True)
    return {k: v/max(count,1) for k,v in totals.items()} | {'steps': count, 'seconds': time.time()-start}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-scenes', type=int, default=4)
    p.add_argument('--max-targets', type=int, default=24)
    p.add_argument('--neighbor-k', type=int, default=16)
    p.add_argument('--signal-k', type=int, default=4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--max-train-packs', type=int, default=0)
    p.add_argument('--max-val-packs', type=int, default=0)
    p.add_argument('--max-train-steps', type=int, default=0)
    p.add_argument('--max-val-steps', type=int, default=0)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    run_dir = Path(args.run_dir); (run_dir/'checkpoints').mkdir(parents=True, exist_ok=True)
    with open(run_dir/'config.json','w',encoding='utf-8') as f: json.dump(vars(args) | {'device':str(device),'torch':torch.__version__},f,ensure_ascii=False,indent=2)
    train_ds = SceneWindowDataset(args.data_root, 'train', args.max_train_packs)
    val_ds = SceneWindowDataset(args.data_root, 'val', args.max_val_packs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_scenes, shuffle=True, num_workers=args.workers, collate_fn=lambda x:x, pin_memory=device.type=='cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_scenes, shuffle=False, num_workers=args.workers, collate_fn=lambda x:x, pin_memory=device.type=='cuda')
    model = MotionPredictor(args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = float('inf'); rows=[]
    for epoch in range(1, args.epochs+1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, True, args.max_train_steps)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, False, args.max_val_steps)
        row = {'epoch':epoch} | {f'train_{k}':v for k,v in train_metrics.items()} | {f'val_{k}':v for k,v in val_metrics.items()}
        rows.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
        state = {'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'metrics':row,'args':vars(args)}
        torch.save(state, run_dir/'checkpoints'/'last.pth')
        if val_metrics['minade6'] < best:
            best = val_metrics['minade6']; torch.save(state,run_dir/'checkpoints'/'best_minade6.pth')
    with open(run_dir/'metrics.json','w',encoding='utf-8') as f: json.dump(rows,f,ensure_ascii=False,indent=2)
    with open(run_dir/'metrics.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)

if __name__ == '__main__': main()
