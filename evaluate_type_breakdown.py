#!/usr/bin/env python3
"""Per-type and time-horizon breakdown for V3/V6 on the 24-target val protocol."""
from __future__ import annotations
import argparse, json, os, sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.train_motion_prediction_v1 import SceneWindowDataset, list_pt_paths
from src.train_motion_prediction_v3 import MAP_K, MotionPredictorV3, WindowSampleCollateV3, load_compatible_state
from src.train_motion_prediction_v6 import MotionPredictorV6
from train_motion_prediction_v2_awta import configure_runtime, make_loader, move_samples
from train_motion_prediction_v3 import model_forward

TYPE_NAMES = {0: "Vehicle", 1: "Pedestrian", 2: "Cyclist"}
HORIZONS = (("0-2s", 0, 20), ("2-4s", 20, 40), ("4-8s", 40, 80))


def empty_bucket():
    return {
        "n": 0,
        "ade6": 0.0, "ade1": 0.0, "fde6": 0.0, "fde1": 0.0,
        "miss2": 0,
        "h_ade6": [0.0, 0.0, 0.0],
        "h_n": [0, 0, 0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["v3", "v6"], required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", default=r"E:\motion_planning\data\processed\prediction_pt_85k")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--batch-scenes", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    configure_runtime(args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    val_paths = list_pt_paths(args.data_root, "val", 0)
    probe = torch.load(val_paths[0], map_location="cpu", weights_only=False)
    n_windows = max(1, len(probe.get("windows", [])))
    val_ds = SceneWindowDataset(args.data_root, "val", paths=val_paths, n_windows=n_windows)
    loader = make_loader(
        val_ds, args.batch_scenes, args.workers, 4, False,
        WindowSampleCollateV3(24, 16, 4, MAP_K, False), device.type == "cuda",
    )
    model = (MotionPredictorV6 if args.arch == "v6" else MotionPredictorV3)(hidden=256, modes=6).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    load_compatible_state(model, ckpt["model_state"])
    model.eval()

    buckets = {name: empty_bucket() for name in TYPE_NAMES.values()}
    overall = empty_bucket()

    with torch.no_grad():
        for samples in loader:
            if samples is None or samples["target_hist"].shape[0] == 0:
                continue
            samples = move_samples(samples, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                pred, _, logits = model_forward(model, samples)
            future = samples["future"]
            valid = samples["future_valid"]
            types = samples["type_idx"]
            diff = pred - future[:, None, :, :]
            disp = torch.linalg.vector_norm(diff, dim=-1)
            mask = valid[:, None, :].float()
            denom = mask.sum(dim=-1).clamp_min(1.0)
            ade_modes = (disp * mask).sum(dim=-1) / denom
            last_idx = (valid.sum(dim=-1).long() - 1).clamp(min=0, max=79)
            fde_modes = torch.gather(disp, 2, last_idx[:, None, None].expand(-1, 6, 1)).squeeze(2)
            b_idx = torch.arange(pred.shape[0], device=device)
            best = ade_modes.argmin(dim=1)
            top1 = logits.argmax(dim=1)
            ade6 = ade_modes[b_idx, best]
            ade1 = ade_modes[b_idx, top1]
            fde6 = fde_modes[b_idx, best]
            fde1 = fde_modes[b_idx, top1]
            miss = (fde6 > 2.0).to(dtype=torch.int64)
            best_disp = disp[b_idx, best]

            types_cpu = types.cpu()
            ade6_cpu = ade6.float().cpu()
            ade1_cpu = ade1.float().cpu()
            fde6_cpu = fde6.float().cpu()
            fde1_cpu = fde1.float().cpu()
            miss_cpu = miss.cpu()
            best_disp_cpu = best_disp.float().cpu()
            valid_cpu = valid.cpu()

            for i in range(pred.shape[0]):
                name = TYPE_NAMES.get(int(types_cpu[i].item()), "Vehicle")
                for bucket in (buckets[name], overall):
                    bucket["n"] += 1
                    bucket["ade6"] += float(ade6_cpu[i])
                    bucket["ade1"] += float(ade1_cpu[i])
                    bucket["fde6"] += float(fde6_cpu[i])
                    bucket["fde1"] += float(fde1_cpu[i])
                    bucket["miss2"] += int(miss_cpu[i])
                    for h, (label, lo, hi) in enumerate(HORIZONS):
                        v = valid_cpu[i, lo:hi].float()
                        if float(v.sum()) < 1:
                            continue
                        bucket["h_ade6"][h] += float((best_disp_cpu[i, lo:hi] * v).sum() / v.sum().clamp_min(1.0))
                        bucket["h_n"][h] += 1

    def finish(bucket):
        n = max(1, bucket["n"])
        out = {
            "count": bucket["n"],
            "minade6": bucket["ade6"] / n,
            "minade1": bucket["ade1"] / n,
            "minfde6": bucket["fde6"] / n,
            "minfde1": bucket["fde1"] / n,
            "miss_rate_2m": 100.0 * bucket["miss2"] / n,
        }
        for h, (label, _, _) in enumerate(HORIZONS):
            hn = max(1, bucket["h_n"][h])
            out[f"minade6_{label}"] = bucket["h_ade6"][h] / hn
        return out

    report = {
        "arch": args.arch,
        "ckpt": args.ckpt,
        "overall": finish(overall),
        "by_type": {k: finish(v) for k, v in buckets.items()},
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
