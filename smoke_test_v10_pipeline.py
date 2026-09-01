#!/usr/bin/env python3
"""Smoke-test v2 preprocess + V10 collate / forward / aWTA / 1-step train."""
from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0
ERRORS: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        msg = f"  FAIL  {name}" + (f"  | {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg.strip())


def test_lane_index():
    print("\n[1] lane_index merge / adjacent / drop")
    from src.lane_index import lane_index_from_polylines

    def poly_at(lats, yaw=0.0):
        n = len(lats)
        xy = np.zeros((n, 20, 2), np.float32)
        direc = np.zeros((n, 20, 2), np.float32)
        valid = np.ones((n, 20), bool)
        ptype = np.ones((n,), np.int32)
        for i, lat in enumerate(lats):
            xy[i, :, 0] = np.linspace(-10, 30, 20)
            xy[i, :, 1] = lat
            direc[i, :, 0] = 1.0
        return xy, direc, valid, ptype

    xy, d, v, t = poly_at([0.0, 0.6])
    info = lane_index_from_polylines(xy, d, v, t, 0.0, 0.0, 0.0)
    check("same-lane |dlat|<1.4 merge n=1", info["valid"] and info["n_lanes"] == 1, str(info))

    xy, d, v, t = poly_at([0.0, 3.5])
    info = lane_index_from_polylines(xy, d, v, t, 0.0, 0.0, 0.0)
    check(
        "adjacent 2.5-6.5 n=2 idx=1",
        info["valid"] and info["n_lanes"] == 2 and info["lane_idx"] == 1 and info["has_right"],
        str(info),
    )

    xy, d, v, t = poly_at([0.0, 8.0])
    info = lane_index_from_polylines(xy, d, v, t, 0.0, 0.0, 0.0)
    check("gap>6.5 drop extra n=1", info["valid"] and info["n_lanes"] == 1, str(info))

    xy, d, v, t = poly_at([0.0])
    t[0] = 3
    info = lane_index_from_polylines(xy, d, v, t, 0.0, 0.0, 0.0)
    check("type 3 bike not a driving lane", not info["valid"], str(info))

    xy, d, v, t = poly_at([0.0, 2.0, 5.5])
    info = lane_index_from_polylines(xy, d, v, t, 0.0, 0.0, 0.0)
    check(
        "dead-zone 2.0m skip then 5.5 adjacent n=2 idx=1",
        info["valid"]
        and info["n_lanes"] == 2
        and info["lane_idx"] == 1
        and abs(info["lat"]) < 0.5
        and info["has_right"]
        and not info["has_left"],
        str(info),
    )

    xy, d, v, t = poly_at([0.0, 2.0, 8.0])
    info = lane_index_from_polylines(xy, d, v, t, 0.0, 0.0, 0.0)
    check(
        "gap vs last accepted: 0/2.0/8.0 skip 2.0 then 8-0>6.5 n=1",
        info["valid"] and info["n_lanes"] == 1 and abs(info["lat"]) < 0.5,
        str(info),
    )

    xy, d, v, t = poly_at([-20.0, -4.0, 0.0, 4.0])
    info = lane_index_from_polylines(xy, d, v, t, 0.0, 0.0, 0.0)
    check(
        "parallel road -20/-4/0/4 -> n=3 idx=2 lat~0",
        info["valid"]
        and info["n_lanes"] == 3
        and info["lane_idx"] == 2
        and abs(info["lat"]) < 0.5
        and info["has_left"]
        and info["has_right"],
        str(info),
    )


def test_polyline_order():
    print("\n[1b] polyline dir/PCA sort then resample")
    from src.map_polylines import build_map_polylines, order_polyline_points

    rng = np.random.default_rng(0)
    n = 40
    xy = np.stack([np.linspace(0.0, 100.0, n), np.zeros(n)], axis=1).astype(np.float32)
    direc = np.tile(np.array([1.0, 0.0], dtype=np.float32), (n, 1))
    perm = rng.permutation(n)
    ox, _ = order_polyline_points(xy[perm], direc[perm])
    check("shuffled line x monotonic after dir sort", bool(np.all(np.diff(ox[:, 0]) > -1e-4)), str(ox[:5, 0]))

    ox_pca, _ = order_polyline_points(xy[perm], np.zeros_like(direc))
    dx = np.diff(ox_pca[:, 0])
    check("PCA fallback monotonic", bool(np.all(dx > -1e-4) or np.all(dx < 1e-4)))

    xyz = np.concatenate([xy[perm], np.zeros((n, 1), dtype=np.float32)], axis=1)
    direction = np.concatenate([direc[perm], np.zeros((n, 1), dtype=np.float32)], axis=1)
    types = np.full((n,), 2, dtype=np.int64)
    valid = np.ones((n,), dtype=bool)
    ids = np.full((n,), 7, dtype=np.int64)
    packed = build_map_polylines(xyz, direction, types, valid, ids)
    pxy = packed["map_polyline_xy"][0].numpy()
    check("resampled x monotonic", bool(np.all(np.diff(pxy[:, 0]) > 0)), str(pxy[:, 0]))
    gaps = np.linalg.norm(np.diff(pxy, axis=0), axis=1)
    check("100m lane resampled gaps < 8m", float(gaps.max()) < 8.0, f"max={float(gaps.max()):.2f}")


def test_last_valid_and_loss():
    print("\n[2] last-True-valid FDE + aWTA shapes")
    from src.losses.awta_loss import AdaptiveWTALoss
    from src.train_motion_prediction_v10 import last_true_index

    valid = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    idx = last_true_index(valid)
    check("last_true_index [2,2,0]", idx.tolist() == [2, 2, 0], str(idx.tolist()))

    B, K, T = 4, 6, 80
    pred = torch.randn(B, K, T, 2, requires_grad=True)
    goals = torch.randn(B, K, 2, requires_grad=True)
    logits = torch.zeros(B, K, requires_grad=True)
    gt = torch.randn(B, T, 2)
    fv = torch.zeros(B, T, dtype=torch.bool)
    fv[:, :40] = True
    fv[0, 50] = True
    fv[0, 40:50] = False
    loss_fn = AdaptiveWTALoss(modes=K, weight_fde=0.4)
    loss, metrics = loss_fn(pred, goals, logits, gt, fv, epoch=0, total_epochs=30)
    check("aWTA finite", torch.isfinite(loss), str(float(loss)))
    check("aWTA has minade6", "minade6_batch" in metrics)
    t_ix = torch.arange(T).view(1, T).expand(B, T)
    last = torch.where(fv, t_ix, t_ix.new_zeros(())).max(-1).values
    check("non-prefix last idx sample0=50", int(last[0]) == 50, str(int(last[0])))
    loss.backward()
    check("aWTA backward", True)


def test_split_hist():
    print("\n[3] hist 11x6 split")
    from src.train_motion_prediction_v10 import split_hist

    hist = torch.zeros(3, 11, 6)
    hist[:, :, :5] = 1.0
    hist[:, -1, 5] = 1.0
    xy5, valid = split_hist(hist, None)
    check("split xy5 [3,11,5]", tuple(xy5.shape) == (3, 11, 5), str(tuple(xy5.shape)))
    check("split valid current True", bool(valid[:, -1].all()) and not bool(valid[:, 0].any()))


def test_preprocess_one_file(tmp_out: str) -> list[str]:
    print("\n[4] preprocess 1 TFRecord -> v2 pack")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_tools"))
    from preprocess_85k_v2 import SCHEMA, process_single_file
    from src.map_polylines import KEEP_TYPES

    raw_root = r"E:\motion_data\rideflux_91f_full\rideflux"
    files = sorted(glob.glob(os.path.join(raw_root, "**", "*.tfrecord"), recursive=True))
    check("raw tfrecords exist", len(files) > 0, raw_root)
    if not files:
        return []
    os.makedirs(os.path.join(tmp_out, "train"), exist_ok=True)
    os.makedirs(os.path.join(tmp_out, "val"), exist_ok=True)
    ok, rel, w_cnt, t_cnt, tl_cnt = process_single_file((files[0], raw_root, tmp_out, 10, True))
    check("process_single_file ok", bool(ok), str(rel if ok else w_cnt))
    packs = glob.glob(os.path.join(tmp_out, "**", "*.pt"), recursive=True)
    check("wrote >=1 pack", len(packs) >= 1, str(len(packs)))
    if not packs:
        return []
    pack = torch.load(packs[0], map_location="cpu", weights_only=False)
    win = pack["windows"][0]
    hist = win["inputs"]["agent_history_world"]
    check("schema v2", pack.get("schema_version") == SCHEMA, str(pack.get("schema_version")))
    check("hist [N,11,6]", hist.ndim == 3 and hist.shape[1:] == (11, 6), str(tuple(hist.shape)))
    check("tl_xy key", "tl_xy" in win["inputs"] and win["inputs"]["tl_xy"].shape[-1] == 2)
    check("tl no xyz", "signal_history_world_state" not in win["inputs"])
    check("tl steps 11", win["inputs"]["tl_xy"].shape[1] == 11 if win["inputs"]["tl_xy"].ndim == 3 else True)
    rows = win["targets"]["target_rows"]
    hv = win["inputs"]["agent_history_valid"]
    check("target_rows currently valid", bool(hv[rows, -1].all()) if rows.numel() else True)
    sizes = win["inputs"]["agent_size_m"]
    types_a = pack["static"]["agent_types"].long()
    cur_ok = hv[:, -1]
    check(
        "agent_size_m not all zero",
        bool((sizes[cur_ok, 0] > 0).any()) if bool(cur_ok.any()) else True,
        f"max_len={float(sizes[:, 0].max()) if sizes.numel() else 0}",
    )
    veh = cur_ok & ((types_a == 1) | (types_a == 2))
    if bool(veh.any()):
        lens = sizes[veh, 0]
        check(
            "valid vehicle length 1-20m (not packed zeros)",
            bool(((lens > 1.0) & (lens < 20.0)).sum() >= max(1, int(0.5 * int(veh.sum())))),
            f"n={int(veh.sum())} min={float(lens.min()):.2f} max={float(lens.max()):.2f} mean={float(lens.mean()):.2f}",
        )
    else:
        check("valid vehicle length 1-20m (not packed zeros)", True, "no type 1/2 currently valid")
    pxy = pack["static"]["map_polyline_xy"]
    pvalid = pack["static"]["map_polyline_valid"]
    ptype = pack["static"]["map_polyline_type"]
    ratios = []
    if pxy.numel() and ptype.numel():
        for i in range(pxy.shape[0]):
            if int(ptype[i]) not in (1, 2):
                continue
            pts = pxy[i][pvalid[i]].numpy()
            if pts.shape[0] < 2:
                continue
            chord = float(np.linalg.norm(pts[-1] - pts[0]))
            arclen = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            ratios.append(arclen / max(chord, 1e-3))
    if ratios:
        frac_ok = float(np.mean(np.array(ratios) < 2.5))
        check(
            "lane polylines not zigzag (arc/chord < 2.5)",
            frac_ok >= 0.7,
            f"frac_ok={frac_ok:.2f} n={len(ratios)} median={float(np.median(ratios)):.2f} max={float(np.max(ratios)):.2f}",
        )
    else:
        check("lane polylines not zigzag (arc/chord < 2.5)", True, "no lane polylines")
    if ptype.numel():
        extra = sorted({int(x) for x in ptype.tolist() if int(x) not in KEEP_TYPES})
        check("polyline types in KEEP {1,2,15,16,17,18}", extra == [], str(extra))
    else:
        check("polyline types in KEEP {1,2,15,16,17,18}", True, "empty polylines")
    return packs


def test_collate_forward_train(packs: list[str], tmp_root: str):
    print("\n[5] collate + V10 forward/backward + 1-step train")
    if not packs:
        check("collate skipped (no packs)", False, "preprocess produced no packs")
        return
    from src.losses.awta_loss import AdaptiveWTALoss
    from src.train_motion_prediction_v10 import (
        AGENT_DIM, INTER_DIM, LANE_DIM, MAP_DIM, MotionPredictorV10, WindowSampleCollateV10,
    )

    pack = torch.load(packs[0], map_location="cpu", weights_only=False)
    collate = WindowSampleCollateV10(max_targets=16, train=False)
    samples = collate([(pack["static"], pack["windows"][0])])
    n = int(samples["target_hist"].shape[0])
    check("collate targets > 0", n > 0, f"n={n}")
    if n == 0:
        return
    check("target_hist [T,11,5]", tuple(samples["target_hist"].shape[1:]) == (11, 5), str(tuple(samples["target_hist"].shape)))
    check("agent_tok dim", samples["agent_tok"].shape[-1] == AGENT_DIM)
    check("lane_tok dim", samples["lane_tok"].shape[-1] == LANE_DIM)
    check("map_tok dim", samples["map_tok"].shape[-1] == MAP_DIM)
    check("inter_tok dim", samples["inter_tok"].shape[-1] == INTER_DIM)
    check("future [T,80,2]", tuple(samples["future"].shape[1:]) == (80, 2))
    check("type_idx in 0..2", int(samples["type_idx"].min()) >= 0 and int(samples["type_idx"].max()) <= 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionPredictorV10(hidden=128, modes=6).to(device)
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in samples.items()}
    model.train()
    pred, goals, logits = model(
        batch["target_hist"], batch["neighbors"], batch["neighbor_valid"],
        batch["map_feat"], batch["map_valid"], batch["signals"], batch["type_idx"],
        agent_tok=batch["agent_tok"], lane_tok=batch["lane_tok"], lane_valid=batch["lane_valid"],
        map_tok=batch["map_tok"], inter_tok=batch["inter_tok"],
    )
    check("pred [B,6,80,2]", tuple(pred.shape) == (n, 6, 80, 2), str(tuple(pred.shape)))
    check("goals [B,6,2]", tuple(goals.shape) == (n, 6, 2), str(tuple(goals.shape)))
    check("logits [B,6]", tuple(logits.shape) == (n, 6), str(tuple(logits.shape)))
    check("pred finite", bool(torch.isfinite(pred).all()))
    loss_fn = AdaptiveWTALoss(modes=6, weight_fde=0.4).to(device)
    loss, metrics = loss_fn(pred, goals, logits, batch["future"], batch["future_valid"], 0, 30)
    check("train loss finite", torch.isfinite(loss), str(float(loss)))
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    check("grads finite", grad_ok)

    print("\n[6] trainer CLI 1-step")
    train_dir = os.path.join(tmp_root, "train")
    val_dir = os.path.join(tmp_root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    if not glob.glob(os.path.join(val_dir, "*.pt")):
        shutil.copy2(packs[0], os.path.join(val_dir, os.path.basename(packs[0])))
    ckpt_dir = os.path.join(tmp_root, "ckpt")
    import subprocess
    proc = subprocess.run(
        [
            sys.executable, "train_motion_prediction_v10.py",
            "--data-root", tmp_root, "--out-dir", ckpt_dir,
            "--epochs", "1", "--batch-scenes", "1", "--workers", "0", "--max-packs", "2",
            "--max-train-steps", "1", "--max-val-steps", "1", "--hidden", "128",
            "--max-targets", "8", "--amp", "fp32",
        ],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    check("trainer exit 0", proc.returncode == 0, out[-1500:])
    check("trainer wrote last.pth", os.path.isfile(os.path.join(ckpt_dir, "last.pth")))
    if proc.returncode != 0:
        print(out[-2000:])


def main():
    print("=" * 72)
    print(" V10 PIPELINE SMOKE")
    print("=" * 72)
    tmp = tempfile.mkdtemp(prefix="v10_smoke_")
    try:
        test_lane_index()
        test_polyline_order()
        test_last_valid_and_loss()
        test_split_hist()
        packs = test_preprocess_one_file(tmp)
        test_collate_forward_train(packs, tmp)
    except Exception:
        global FAIL
        FAIL += 1
        ERRORS.append(traceback.format_exc())
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 72)
    print(f" RESULT  pass={PASS}  fail={FAIL}")
    if ERRORS:
        print(" Errors:")
        for e in ERRORS:
            print("  -", e[:500])
    print("=" * 72)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
