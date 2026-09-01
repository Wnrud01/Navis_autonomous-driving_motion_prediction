#!/usr/bin/env python3
"""Where V9 ADE/FDE come from: type, horizon, percentiles, valid-length, GT travel."""
from __future__ import annotations
import json, os, sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v1 import SceneWindowDataset, list_pt_paths
from src.train_motion_prediction_v3 import MAP_K, load_compatible_state
from src.train_motion_prediction_v8 import POLY_K
from src.train_motion_prediction_v9 import MotionPredictorV9, WindowSampleCollateV9
from train_motion_prediction_v2_awta import configure_runtime, make_loader, move_samples
import train_motion_prediction_v3 as v3_train
from train_motion_prediction_v9 import model_forward

v3_train.model_forward = model_forward
TYPE_NAMES = {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"}
SLICES = (("0-1s", 0, 10), ("1-2s", 10, 20), ("2-4s", 20, 40), ("4-6s", 40, 60), ("6-8s", 60, 80))


def pct(x: torch.Tensor, qs=(50, 75, 90, 95, 99)):
    x = x.float().cpu()
    out = {}
    for q in qs:
        out[f"p{q}"] = float(torch.quantile(x, q / 100.0))
    out["mean"] = float(x.mean())
    out["max"] = float(x.max())
    return out


def main():
    ckpt_path = r"E:\motion_prediction\checkpoints\v9_scratch\best_error_score.pth"
    data_root = r"E:\motion_planning\data\processed\prediction_pt_85k"
    configure_runtime(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    val_paths = list_pt_paths(data_root, "val", 0)
    probe = torch.load(val_paths[0], map_location="cpu", weights_only=False)
    n_windows = max(1, len(probe.get("windows", [])))
    ds = SceneWindowDataset(data_root, "val", paths=val_paths, n_windows=n_windows)
    loader = make_loader(
        ds, 32, 8, 4, False,
        WindowSampleCollateV9(24, 16, 4, MAP_K, POLY_K, False),
        device.type == "cuda",
    )
    model = MotionPredictorV9(hidden=256, modes=6).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_compatible_state(model, ckpt["model_state"])
    model.eval()

    fde6_all, fde1_all, ade6_all, ade1_all = [], [], [], []
    types_all, nvalid_all, travel_all = [], [], []
    slice_ade = {k: [] for k, _, _ in SLICES}
    type_fde = {n: [] for n in TYPE_NAMES.values()}
    type_ade6 = {n: [] for n in TYPE_NAMES.values()}
    type_travel = {n: [] for n in TYPE_NAMES.values()}

    with torch.no_grad():
        for samples in loader:
            if samples is None or samples["target_hist"].shape[0] == 0:
                continue
            samples = move_samples(samples, device)
            with torch.amp.autocast("cuda", dtype=amp, enabled=(device.type == "cuda")):
                pred, _, logits = model_forward(model, samples)
            future = samples["future"]
            valid = samples["future_valid"]
            types = samples["type_idx"]
            diff = pred - future[:, None]
            disp = torch.linalg.vector_norm(diff, dim=-1)
            mask = valid[:, None, :].float()
            denom = mask.sum(dim=-1).clamp_min(1.0)
            ade_m = (disp * mask).sum(-1) / denom
            last = (valid.sum(-1).long() - 1).clamp(0, 79)
            fde_m = torch.gather(disp, 2, last[:, None, None].expand(-1, 6, 1)).squeeze(2)
            b = torch.arange(pred.shape[0], device=device)
            best = ade_m.argmin(1)
            top1 = logits.argmax(1)
            ade6 = ade_m[b, best]
            ade1 = ade_m[b, top1]
            fde6 = fde_m[b, best]
            fde1 = fde_m[b, top1]
            best_disp = disp[b, best]
            nval = valid.sum(-1).float()
            first_xy = future[:, 0]
            last_xy = torch.gather(future, 1, last[:, None, None].expand(-1, 1, 2)).squeeze(1)
            travel = torch.linalg.vector_norm(last_xy - first_xy, dim=-1)

            fde6_all.append(fde6.float().cpu())
            fde1_all.append(fde1.float().cpu())
            ade6_all.append(ade6.float().cpu())
            ade1_all.append(ade1.float().cpu())
            types_all.append(types.cpu())
            nvalid_all.append(nval.cpu())
            travel_all.append(travel.float().cpu())
            for name, lo, hi in SLICES:
                v = valid[:, lo:hi].float()
                sl = (best_disp[:, lo:hi] * v).sum(-1) / v.sum(-1).clamp_min(1.0)
                sl = sl.masked_fill(v.sum(-1) < 1, float("nan"))
                slice_ade[name].append(sl.cpu())

    fde6 = torch.cat(fde6_all)
    fde1 = torch.cat(fde1_all)
    ade6 = torch.cat(ade6_all)
    ade1 = torch.cat(ade1_all)
    types = torch.cat(types_all)
    nvalid = torch.cat(nvalid_all)
    travel = torch.cat(travel_all)

    def slice_mean(name):
        x = torch.cat(slice_ade[name])
        x = x[torch.isfinite(x)]
        return float(x.mean()) if x.numel() else None

    # valid-length bins
    bins = [(0, 20, "<2s"), (20, 40, "2-4s"), (40, 60, "4-6s"), (60, 81, "6-8s")]
    valid_bins = {}
    for lo, hi, lab in bins:
        m = (nvalid >= lo) & (nvalid < hi)
        if int(m.sum()) == 0:
            continue
        valid_bins[lab] = {
            "n": int(m.sum()),
            "ade6": float(ade6[m].mean()),
            "fde6": float(fde6[m].mean()),
            "ade1": float(ade1[m].mean()),
            "travel": float(travel[m].mean()),
        }

    # travel bins
    tbin = {}
    for lo, hi, lab in ((0, 5, "0-5m"), (5, 20, "5-20m"), (20, 50, "20-50m"), (50, 1e9, "50m+")):
        m = (travel >= lo) & (travel < hi)
        if int(m.sum()) == 0:
            continue
        tbin[lab] = {
            "n": int(m.sum()),
            "ade6": float(ade6[m].mean()),
            "fde6": float(fde6[m].mean()),
            "fde6_p50": float(torch.quantile(fde6[m], 0.5)),
            "fde6_p90": float(torch.quantile(fde6[m], 0.9)),
        }

    by_type = {}
    for tid, name in TYPE_NAMES.items():
        m = types == tid
        if int(m.sum()) == 0:
            continue
        by_type[name] = {
            "n": int(m.sum()),
            "ade6": float(ade6[m].mean()),
            "ade1": float(ade1[m].mean()),
            "fde6": pct(fde6[m]),
            "miss2": float((fde6[m] > 2).float().mean() * 100),
            "travel": float(travel[m].mean()),
            "nvalid": float(nvalid[m].mean()),
        }

    report = {
        "n": int(ade6.numel()),
        "error": 0.5 * (float(ade1.mean()) + float(ade6.mean())),
        "ade6": float(ade6.mean()),
        "ade1": float(ade1.mean()),
        "ade_gap": float((ade1 - ade6).mean()),
        "fde6": pct(fde6),
        "fde1": pct(fde1),
        "miss2": float((fde6 > 2).float().mean() * 100),
        "horizon_ade6": {k: slice_mean(k) for k, _, _ in SLICES},
        "by_valid_len": valid_bins,
        "by_gt_travel": tbin,
        "by_type": by_type,
        "corr_fde_travel": float(torch.corrcoef(torch.stack([fde6, travel]))[0, 1]),
        "corr_fde_nvalid": float(torch.corrcoef(torch.stack([fde6, nvalid]))[0, 1]),
    }
    print(json.dumps(report, indent=2), flush=True)
    out = r"E:\motion_prediction\checkpoints\v9_scratch\fde_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
