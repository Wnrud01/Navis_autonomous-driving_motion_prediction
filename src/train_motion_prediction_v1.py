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
NEIGHBOR_FEAT_DIM = 10  # xy, vel, size, type, is_same_lane, is_same_dir
NEIGHBOR_POOL_K = 32
LANE_LAT_M = 2.2
LANE_HEADING_DEG = 35.0
DIR_HEADING_DEG = 45.0
NEIGHBOR_MAX_RANGE_M = 80.0


def list_pt_paths(root: str, split: str, max_packs: int = 0) -> list[str]:
    folder = os.path.join(root, split)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'No {split} dir under {root}')
    paths = [os.path.join(folder, name) for name in os.listdir(folder) if name.endswith('.pt')]
    paths.sort()
    if max_packs:
        paths = paths[:max_packs]
    if not paths:
        raise FileNotFoundError(f'No {split} packs under {root}')
    return paths


def local_xy(xy: torch.Tensor, origin: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    d = xy - origin
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], dim=-1)


def local_vec(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], dim=-1)


def wrap_angle(rad: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(rad), torch.cos(rad))


def select_neighbor_indices(
    cur_local: torch.Tensor,
    agent_yaw: torch.Tensor,
    target_yaw: torch.Tensor,
    candidate: torch.Tensor,
    neighbor_k: int = 16,
    pool_k: int = NEIGHBOR_POOL_K,
    lane_lat_m: float = LANE_LAT_M,
    lane_heading_deg: float = LANE_HEADING_DEG,
    dir_heading_deg: float = DIR_HEADING_DEG,
    max_range_m: float = NEIGHBOR_MAX_RANGE_M,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick neighbors: same-lane, then same-heading, then nearest others.

    cur_local: [N, 2] or [T, N, 2] in the target frame (+x forward).
    Returns nidx, is_lane, is_dir, valid_pick with shape [K] or [T, K].
    """
    squeeze = cur_local.ndim == 2
    if squeeze:
        cur_local = cur_local.unsqueeze(0)
        candidate = candidate.unsqueeze(0)
        target_yaw = target_yaw.reshape(1)
    t_count, n_agents, _ = cur_local.shape
    device = cur_local.device
    dist = torch.linalg.vector_norm(cur_local, dim=-1)
    dist = dist.masked_fill(~candidate, float("inf"))
    k_keep = min(neighbor_k, max(n_agents - 1, 1), n_agents)
    k_pool = min(pool_k, max(n_agents - 1, 1), n_agents)

    local_yaw = wrap_angle(agent_yaw.unsqueeze(0) - target_yaw.unsqueeze(1))
    heading = local_yaw.abs()
    lat = cur_local[..., 1].abs()
    in_range = dist <= max_range_m
    lane_lim = math.radians(lane_heading_deg)
    dir_lim = math.radians(dir_heading_deg)
    same_lane = candidate & in_range & (lat <= lane_lat_m) & (heading <= lane_lim)
    same_dir = candidate & in_range & (heading <= dir_lim) & ~same_lane

    pool_idx = torch.topk(dist, k=k_pool, largest=False).indices
    in_pool = torch.zeros(t_count, n_agents, dtype=torch.bool, device=device)
    in_pool.scatter_(1, pool_idx, True)
    in_pool &= candidate
    # Keep same-lane agents even if they fall outside the Euclidean 32.
    eligible = same_lane | in_pool
    score = same_lane.float() * 2.0e5 + same_dir.float() * 1.0e5 - dist
    score = score.masked_fill(~eligible, -1.0e9)
    nidx = torch.topk(score, k=k_keep, largest=True).indices
    gather_t = torch.arange(t_count, device=device).unsqueeze(1)
    is_lane = same_lane[gather_t, nidx]
    is_dir = same_dir[gather_t, nidx]
    valid_pick = eligible[gather_t, nidx]
    if squeeze:
        return nidx[0], is_lane[0], is_dir[0], valid_pick[0]
    return nidx, is_lane, is_dir, valid_pick


def pack_neighbor_features(
    cur_local: torch.Tensor,
    cur_vel: torch.Tensor,
    sizes: torch.Tensor,
    types: torch.Tensor,
    nidx: torch.Tensor,
    is_lane: torch.Tensor,
    is_dir: torch.Tensor,
    valid_pick: torch.Tensor,
    neighbor_k: int,
) -> torch.Tensor:
    """Assemble [K, 10] or [T, K, 10] neighbor features, padded to neighbor_k."""
    squeeze = nidx.ndim == 1
    if squeeze:
        cur_local = cur_local.unsqueeze(0)
        cur_vel = cur_vel.unsqueeze(0)
        nidx = nidx.unsqueeze(0)
        is_lane = is_lane.unsqueeze(0)
        is_dir = is_dir.unsqueeze(0)
        valid_pick = valid_pick.unsqueeze(0)
    t_count = nidx.shape[0]
    gather_t = torch.arange(t_count, device=nidx.device).unsqueeze(1)
    feat = torch.cat([
        cur_local[gather_t, nidx],
        cur_vel[gather_t, nidx],
        sizes[nidx],
        types[nidx].to(dtype=cur_local.dtype).unsqueeze(-1),
        is_lane.to(dtype=cur_local.dtype).unsqueeze(-1),
        is_dir.to(dtype=cur_local.dtype).unsqueeze(-1),
    ], dim=-1)
    feat = feat * valid_pick.unsqueeze(-1).to(dtype=feat.dtype)
    pad = neighbor_k - feat.shape[1]
    if pad > 0:
        feat = torch.cat([feat, feat.new_zeros(t_count, pad, feat.shape[-1])], dim=1)
    return feat[0] if squeeze else feat


def expand_neighbor_encoder_state(state: dict, in_features: int = NEIGHBOR_FEAT_DIM) -> dict:
    """Pad a checkpoint neighbor encoder input dim (8 -> 10) so V1 weights still load."""
    key = "neighbor_encoder.0.weight"
    weight = state.get(key)
    if not torch.is_tensor(weight) or weight.shape[1] == in_features:
        return state
    expanded = weight.new_zeros(weight.shape[0], in_features)
    copied = min(weight.shape[1], in_features)
    expanded[:, :copied] = weight[:, :copied]
    state[key] = expanded
    return state


class SceneWindowDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        max_packs: int = 0,
        paths: list[str] | None = None,
        n_windows: int | None = None,
        cache_size: int = 48,
    ):
        self.paths = list(paths) if paths is not None else list_pt_paths(root, split, max_packs)
        if max_packs and paths is not None:
            self.paths = self.paths[:max_packs]
        if not self.paths:
            raise FileNotFoundError(f'No {split} packs under {root}')
        if n_windows is None:
            sample = torch.load(self.paths[0], map_location='cpu', weights_only=False)
            n_windows = max(1, len(sample.get('windows', [])))
        self.n_windows = int(n_windows)
        self.items = [(path, wi) for path in self.paths for wi in range(self.n_windows)]
        self._cache: dict[str, Any] = {}
        self._cache_max = max(0, int(cache_size))

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache'] = {}
        return state

    def __len__(self):
        return len(self.items)

    def _load(self, path: str):
        pack = self._cache.get(path)
        if pack is not None:
            return pack
        pack = torch.load(path, map_location='cpu', weights_only=False)
        if self._cache_max:
            if len(self._cache) >= self._cache_max:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = pack
        return pack

    def __getitem__(self, index):
        path, window_index = self.items[index]
        pack = self._load(path)
        windows = pack['windows']
        if window_index >= len(windows):
            window_index %= len(windows)
        return pack['static'], windows[window_index]


def _scene_targets(static: dict, window: dict, max_targets: int, neighbor_k: int, signal_k: int, train: bool):
    hist = window['inputs']['agent_history_world'].float()
    valid = window['inputs']['agent_history_valid'].bool()
    sizes = window['inputs']['agent_size_m'].float()
    types = static['agent_types'].long()
    sig = window['inputs']['signal_history_world_state'].float()
    sig_valid = window['inputs']['signal_history_valid'].bool()
    targets = window['targets']
    target_rows = targets['target_rows'].long()
    n_all = int(target_rows.numel())
    if n_all == 0:
        return None
    if 'target_types' in targets:
        target_types = targets['target_types'].long()
    else:
        target_types = types[target_rows]
    if 'future_xy_world' in targets:
        future_world = targets['future_xy_world'].float()
        future_valid = targets['future_valid'].bool()
    else:
        future_world = targets['agent_future_world'].float()[target_rows, :, :2]
        future_valid = targets['agent_future_valid'].bool()[target_rows]

    keep_mask = torch.zeros(n_all, dtype=torch.bool)
    for src in TYPE_TO_INDEX:
        keep_mask |= target_types == src
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    if keep_idx.numel() == 0:
        return None
    if train and int(keep_idx.numel()) > max_targets:
        keep_idx = keep_idx[torch.randperm(keep_idx.numel())[:max_targets]]
    else:
        keep_idx = keep_idx[:max_targets]

    rows = target_rows[keep_idx]
    t = int(rows.numel())
    n = hist.shape[0]
    origins = hist[rows, -1, :2]
    yaws = hist[rows, -1, 2]
    yaw_th = yaws[:, None]
    th = hist[rows]
    target_hist = torch.cat([
        local_xy(th[:, :, :2], origins[:, None, :], yaw_th),
        (th[:, :, 2] - yaw_th).unsqueeze(-1),
        local_vec(th[:, :, 3:5], yaw_th),
    ], dim=-1)

    if n <= 1:
        nfeat = target_hist.new_zeros(t, neighbor_k, NEIGHBOR_FEAT_DIM)
    else:
        cur_xy = hist[:, -1, :2]
        cur_v = hist[:, -1, 3:5]
        yaw_n = yaws[:, None]
        cur_local = local_xy(cur_xy.unsqueeze(0), origins[:, None, :], yaw_n)
        cur_vel = local_vec(cur_v.unsqueeze(0), yaw_n)
        candidate = valid[:, -1].unsqueeze(0).expand(t, n).clone()
        candidate[torch.arange(t), rows] = False
        nidx, is_lane, is_dir, valid_pick = select_neighbor_indices(
            cur_local, hist[:, -1, 2], yaws, candidate, neighbor_k=neighbor_k,
        )
        nfeat = pack_neighbor_features(
            cur_local, cur_vel, sizes, types, nidx, is_lane, is_dir, valid_pick, neighbor_k,
        )

    n_sig = sig.shape[0]
    if n_sig == 0:
        sfeat = target_hist.new_zeros(t, signal_k, 4)
    else:
        yaw_s = yaws[:, None]
        sig_local = local_xy(sig[:, -1, :2].unsqueeze(0), origins[:, None, :], yaw_s)
        sdist = torch.linalg.vector_norm(sig_local, dim=-1)
        if sig_valid.numel():
            sdist = sdist.masked_fill(~sig_valid[:, -1].unsqueeze(0), float('inf'))
        sidx = torch.topk(sdist, k=min(signal_k, n_sig), largest=False).indices
        gather_t = torch.arange(t).unsqueeze(1)
        if sig.shape[1] > 1:
            changes = (sig[:, 1:, 2] != sig[:, :-1, 2]).float().sum(dim=1)
        else:
            changes = sig.new_zeros(n_sig)
        sfeat = torch.cat([
            sig_local[gather_t, sidx],
            sig[:, -1, 2][sidx].unsqueeze(-1),
            changes[sidx].unsqueeze(-1),
        ], dim=-1)
        pad = signal_k - sfeat.shape[1]
        if pad > 0:
            sfeat = torch.cat([sfeat, sfeat.new_zeros(t, pad, 4)], dim=1)

    mapped = torch.empty(t, dtype=torch.long)
    selected_types = target_types[keep_idx]
    for src, dst in TYPE_TO_INDEX.items():
        mapped[selected_types == src] = dst
    future = local_xy(future_world[keep_idx], origins[:, None, :], yaws[:, None])
    return target_hist, nfeat, sfeat, mapped, future, future_valid[keep_idx]


def window_to_samples(
    batch: list[tuple[dict, dict]],
    max_targets: int = 24,
    neighbor_k: int = 16,
    signal_k: int = 4,
    train: bool = False,
) -> dict[str, torch.Tensor] | None:
    buckets = {k: [] for k in ('target_hist', 'neighbors', 'signals', 'type_idx', 'future', 'future_valid')}
    for static, window in batch:
        packed = _scene_targets(static, window, max_targets, neighbor_k, signal_k, train)
        if packed is None:
            continue
        target_hist, nfeat, sfeat, mapped, future, future_valid = packed
        buckets['target_hist'].append(target_hist)
        buckets['neighbors'].append(nfeat)
        buckets['signals'].append(sfeat)
        buckets['type_idx'].append(mapped)
        buckets['future'].append(future)
        buckets['future_valid'].append(future_valid)
    if not buckets['future']:
        return None
    return {key: torch.cat(value, dim=0) for key, value in buckets.items()}


def collate_raw_windows(batch):
    return batch


class WindowSampleCollate:
    def __init__(self, max_targets: int = 24, neighbor_k: int = 16, signal_k: int = 4, train: bool = False):
        self.max_targets = max_targets
        self.neighbor_k = neighbor_k
        self.signal_k = signal_k
        self.train = train

    def __call__(self, batch):
        samples = window_to_samples(batch, self.max_targets, self.neighbor_k, self.signal_k, self.train)
        if samples is not None:
            return samples
        return {
            "target_hist": torch.zeros(0, 11, 5),
            "neighbors": torch.zeros(0, self.neighbor_k, NEIGHBOR_FEAT_DIM),
            "signals": torch.zeros(0, self.signal_k, 4),
            "type_idx": torch.zeros(0, dtype=torch.long),
            "future": torch.zeros(0, 80, 2),
            "future_valid": torch.zeros(0, 80, dtype=torch.bool),
        }


class MotionPredictor(nn.Module):
    def __init__(self, hidden: int = 256, modes: int = 6):
        super().__init__()
        self.modes = modes
        self.type_emb = nn.Embedding(3, 16)
        self.target_encoder = nn.Sequential(nn.Flatten(), nn.Linear(55, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.neighbor_encoder = nn.Sequential(nn.Linear(NEIGHBOR_FEAT_DIM, hidden//2), nn.GELU(), nn.Linear(hidden//2, hidden))
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
