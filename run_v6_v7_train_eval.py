#!/usr/bin/env python3
"""Train V6 then V7 from V3, then evaluate on the V3 24-target protocol."""
from __future__ import annotations
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
V3_CKPT = os.path.join(ROOT, "checkpoints", "v3_map_attn", "best_error_score.pth")
V6_DIR = os.path.join(ROOT, "checkpoints", "v6_temporal")
V7_DIR = os.path.join(ROOT, "checkpoints", "v7_scene_v3")
PY = sys.executable


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str]) -> int:
    log("\n>>> " + " ".join(cmd))
    return int(subprocess.run(cmd, cwd=ROOT).returncode)


def main() -> int:
    log("=" * 80)
    log(" Train V6 -> V7 from V3, then evaluate vs V3")
    log("=" * 80)

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
