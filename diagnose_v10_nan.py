#!/usr/bin/env python3
"""Find which tensor goes NaN/Inf on a V10 train batch. No training."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.losses.awta_loss import AdaptiveWTALoss
from src.train_motion_prediction_v1 import SceneWindowDataset, list_pt_paths
from src.train_motion_prediction_v10 import MotionPredictorV10, WindowSampleCollateV10
from train_motion_prediction_v2_awta import make_loader, move_samples


def stats(name, t):
    if not torch.is_tensor(t):
        return None
    if not t.is_floating_point():
        return f"{name}: {tuple(t.shape)} {t.dtype} non-float"
    x = t.detach().float()
    n_nan = int(torch.isnan(x).sum())
    n_inf = int(torch.isinf(x).sum())
    mx = float(x.abs().max()) if x.numel() else 0.0
    return f"{name}: shape={tuple(t.shape)} nan={n_nan} inf={n_inf} maxabs={mx:.6g}"


def main():
    data_root = r"E:\motion_prediction\data\processed\prediction_pt_85k_v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    print(f"device={device} amp={amp}", flush=True)

    paths = list_pt_paths(data_root, "train", 8192)
    ds = SceneWindowDataset(data_root, "train", paths=paths, n_windows=1)
    loader = make_loader(ds, 32, 4, 2, True, WindowSampleCollateV10(24, True), device.type == "cuda")
    # stop at first BAD to print neighbor channels
    model = MotionPredictorV10(hidden=256, modes=6).to(device)
    model.eval()
    crit = AdaptiveWTALoss(modes=6, time_weight_end=3.0, weight_fde=0.4).to(device)

    found = 0
    for step, samples in enumerate(loader, start=1):
        if samples is None or samples["target_hist"].shape[0] == 0:
            continue
        samples = move_samples(samples, device)
        n = int(samples["target_hist"].shape[0])
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp, enabled=(device.type == "cuda")):
                pred, goals, logits = model(
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
                loss, parts = crit(
                    pred, goals, logits, samples["future"], samples["future_valid"], 0, 30
                )
        bad = (not torch.isfinite(loss)) or (not torch.isfinite(pred).all()) or (not torch.isfinite(goals).all())
        if not bad:
            if step % 20 == 0:
                print(f" ok step {step} n={n} loss={float(loss):.4f}", flush=True)
            if step >= 200:
                break
            continue
        found += 1
        print("=" * 72, flush=True)
        print(f"BAD step={step} n={n} loss={loss}", flush=True)
        keys = [
            "target_hist", "neighbors", "map_feat", "signals", "future",
            "agent_tok", "lane_tok", "map_tok", "inter_tok",
        ]
        for k in keys:
            if k in samples:
                print(" ", stats(k, samples[k]), flush=True)
        nch = samples["neighbors"].float().abs().amax(dim=(0, 1, 2))
        print("  neighbor_ch_maxabs", [round(float(x), 4) for x in nch], flush=True)
        print(" ", stats("pred", pred), flush=True)
        print(" ", stats("goals", goals), flush=True)
        print(" ", stats("logits", logits), flush=True)
        for k, v in parts.items():
            if torch.is_tensor(v):
                print(" ", stats(f"loss.{k}", v), flush=True)
        # which targets exploded
        pred_f = pred.float()
        row_bad = ~torch.isfinite(pred_f).all(dim=(1, 2, 3))
        print(f"  exploded_targets={int(row_bad.sum())}/{n}", flush=True)
        if int(row_bad.sum()):
            hist = samples["target_hist"][row_bad]
            fut = samples["future"][row_bad]
            types = samples["type_idx"][row_bad]
            print(" ", stats("bad.target_hist", hist), flush=True)
            print(" ", stats("bad.future", fut), flush=True)
            print("  bad.type_idx", types.tolist()[:32], flush=True)
            print("  bad.hist_speed_now", torch.linalg.vector_norm(hist[:, -1, 3:5], dim=-1).tolist()[:16], flush=True)
            print("  bad.future_maxabs", fut.abs().amax(dim=(1, 2)).tolist()[:16], flush=True)
            print("  bad.pred_maxabs", pred_f[row_bad].abs().amax(dim=(1, 2, 3)).tolist()[:16], flush=True)
        if found >= 1:
            break
        if step >= 120:
            break
    print(f"done found={found}", flush=True)


if __name__ == "__main__":
    main()
