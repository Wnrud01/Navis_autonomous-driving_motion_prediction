#!/usr/bin/env python3
"""Wait for V5, then train V6 and V7 and evaluate against V3."""
from __future__ import annotations
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
V5_LOG = os.path.join(ROOT, "checkpoints", "v5_scene", "training.log")
V5_METRICS = os.path.join(ROOT, "checkpoints", "v5_scene", "metrics.json")
V3_CKPT = os.path.join(ROOT, "checkpoints", "v3_map_attn", "best_error_score.pth")
V6_DIR = os.path.join(ROOT, "checkpoints", "v6_temporal")
V7_DIR = os.path.join(ROOT, "checkpoints", "v7_scene_v3")
PY = sys.executable


def log(msg: str) -> None:
    print(msg, flush=True)


def v5_done() -> bool:
    if os.path.isfile(V5_LOG):
        try:
            with open(V5_LOG, "r", encoding="utf-8", errors="replace") as f:
                if "V5 Training Completed" in f.read():
                    return True
        except OSError:
            pass
    if os.path.isfile(V5_METRICS):
        try:
            with open(V5_METRICS, "r", encoding="utf-8") as f:
                hist = json.load(f)
            if hist and int(hist[-1].get("epoch", 0)) >= 40:
                return True
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return False


def run(cmd: list[str], cwd: str = ROOT) -> int:
    log("\n>>> " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd)
    return int(proc.returncode)


def main() -> int:
    log("=" * 80)
    log(" Waiting for V5 to finish, then train V6 -> V7 and evaluate vs V3")
    log("=" * 80)
    t0 = time.time()
    while not v5_done():
        elapsed = time.time() - t0
        log(f"... still waiting for V5 ({elapsed/60:.1f} min)")
        time.sleep(30)
    log(f"V5 finished after { (time.time()-t0)/60:.1f} min of waiting")

    if os.path.isfile(V5_METRICS):
        with open(V5_METRICS, "r", encoding="utf-8") as f:
            hist = json.load(f)
        best = min(hist, key=lambda r: r.get("val_error_score", 1e9))
        log(
            f"V5 best Error={best['val_error_score']:.4f} minADE6={best['val_minade6']:.4f} "
            f"minADE1={best['val_minade1']:.4f} epoch={best['epoch']}"
        )

    rc = run([
        PY, "-u", "train_motion_prediction_v6.py",
        "--resume-ckpt", V3_CKPT,
        "--out-dir", V6_DIR,
        "--epochs", "20",
        "--batch-scenes", "32",
        "--lr", "1.5e-4",
        "--workers", "8",
        "--prefetch", "4",
    ])
    if rc != 0:
        log(f"V6 training failed with code {rc}")
        return rc

    rc = run([
        PY, "-u", "train_motion_prediction_v7.py",
        "--resume-ckpt", V3_CKPT,
        "--out-dir", V7_DIR,
        "--epochs", "20",
        "--batch-scenes", "16",
        "--lr", "1.5e-4",
        "--workers", "8",
        "--prefetch", "4",
        "--max-targets", "0",
    ])
    if rc != 0:
        log(f"V7 training failed with code {rc}")
        return rc

    evals = [
        ("v3", V3_CKPT, 24, os.path.join(ROOT, "checkpoints", "v3_map_attn", "eval_v3_family.json")),
        ("v6", os.path.join(V6_DIR, "best_minade6.pth"), 24, os.path.join(V6_DIR, "eval_24.json")),
        ("v6", os.path.join(V6_DIR, "best_error_score.pth"), 24, os.path.join(V6_DIR, "eval_error_24.json")),
        ("v7", os.path.join(V7_DIR, "best_minade6.pth"), 24, os.path.join(V7_DIR, "eval_24.json")),
        ("v7", os.path.join(V7_DIR, "best_error_score.pth"), 24, os.path.join(V7_DIR, "eval_error_24.json")),
        ("v7", os.path.join(V7_DIR, "best_error_score.pth"), 0, os.path.join(V7_DIR, "eval_all_targets.json")),
    ]
    for arch, ckpt, max_targets, out_json in evals:
        if not os.path.isfile(ckpt):
            log(f"skip missing ckpt {ckpt}")
            continue
        rc = run([
            PY, "-u", "evaluate_v3_family.py",
            "--arch", arch,
            "--ckpt", ckpt,
            "--max-targets", str(max_targets),
            "--out-json", out_json,
            "--workers", "8",
        ])
        if rc != 0:
            log(f"eval failed for {arch} {ckpt} code {rc}")

    log("\nPipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
