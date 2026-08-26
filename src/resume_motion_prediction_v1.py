#!/usr/bin/env python3
"""Resume the separate V1 motion-prediction baseline on full train/validation splits."""
from __future__ import annotations
import argparse, csv, json, os, random, sys
from pathlib import Path
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--module-dir', required=True, help='Directory containing train_motion_prediction_v1.py')
parser.add_argument('--data-root', required=True)
parser.add_argument('--run-dir', required=True)
parser.add_argument('--resume-checkpoint', required=True)
parser.add_argument('--epochs', type=int, default=1, help='Additional full epochs after checkpoint')
parser.add_argument('--batch-scenes', type=int, default=4)
parser.add_argument('--max-targets', type=int, default=24)
parser.add_argument('--neighbor-k', type=int, default=16)
parser.add_argument('--signal-k', type=int, default=4)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--max-train-packs', type=int, default=0)
parser.add_argument('--max-val-packs', type=int, default=0)
parser.add_argument('--max-train-steps', type=int, default=0)
parser.add_argument('--max-val-steps', type=int, default=0)
parser.add_argument('--log-every', type=int, default=500)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

sys.path.insert(0, args.module_dir)
from train_motion_prediction_v1 import MotionPredictor, SceneWindowDataset, run_epoch
from torch.utils.data import DataLoader

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')
run_dir = Path(args.run_dir); (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
model = MotionPredictor(args.hidden).to(device)
model.load_state_dict(checkpoint['model_state'])
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
if 'optimizer_state' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer_state'])

train_ds = SceneWindowDataset(args.data_root, 'train', args.max_train_packs)
val_ds = SceneWindowDataset(args.data_root, 'val', args.max_val_packs)
train_loader = DataLoader(train_ds, batch_size=args.batch_scenes, shuffle=True, num_workers=args.workers, collate_fn=lambda x:x, pin_memory=device.type=='cuda', persistent_workers=args.workers > 0)
val_loader = DataLoader(val_ds, batch_size=args.batch_scenes, shuffle=False, num_workers=args.workers, collate_fn=lambda x:x, pin_memory=device.type=='cuda', persistent_workers=args.workers > 0)
start_epoch = int(checkpoint.get('epoch', 0))
previous_best = float(checkpoint.get('metrics', {}).get('val_minade6', float('inf')))
config = vars(args) | {'device': str(device), 'torch': torch.__version__, 'start_epoch': start_epoch, 'train_packs': len(train_ds.paths), 'val_packs': len(val_ds.paths)}
with open(run_dir / 'config.json', 'w', encoding='utf-8') as f: json.dump(config, f, ensure_ascii=False, indent=2)

metrics_json_path = run_dir / 'metrics.json'
if metrics_json_path.exists():
    try:
        with open(metrics_json_path, 'r', encoding='utf-8') as f:
            rows = json.load(f)
    except Exception:
        rows = []
else:
    rows = []

for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
    train_metrics = run_epoch(model, train_loader, optimizer, device, args, True, args.max_train_steps)
    val_metrics = run_epoch(model, val_loader, optimizer, device, args, False, args.max_val_steps)
    row = {'epoch': epoch} | {f'train_{k}':v for k,v in train_metrics.items()} | {f'val_{k}':v for k,v in val_metrics.items()}
    rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'metrics': row, 'args': config, 'resumed_from': args.resume_checkpoint}
    torch.save(state, run_dir / 'checkpoints' / 'last.pth')
    if val_metrics['minade6'] < previous_best:
        previous_best = val_metrics['minade6']
        torch.save(state, run_dir / 'checkpoints' / 'best_minade6.pth')
with open(run_dir / 'metrics.json', 'w', encoding='utf-8') as f: json.dump(rows, f, ensure_ascii=False, indent=2)
with open(run_dir / 'metrics.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
print(json.dumps({'status':'completed','epochs':args.epochs,'best_minade6':previous_best}, ensure_ascii=False))
