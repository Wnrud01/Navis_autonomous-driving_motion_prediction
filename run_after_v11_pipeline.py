#!/usr/bin/env python3
"""After V11 finishes: build collate cache → V11 ranker → V12 (cache) → V12 ranker if ranker worked."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
V11_LOG = os.path.join(ROOT, "checkpoints", "v11_ade6", "training.log")
V11_CKPT = os.path.join(ROOT, "checkpoints", "v11_ade6", "best_minade6.pth")
V11_LAST = os.path.join(ROOT, "checkpoints", "v11_ade6", "last.pth")
CACHE = os.path.join(ROOT, "data", "processed", "prediction_pt_85k_v2_cache")
RANKER11 = os.path.join(ROOT, "checkpoints", "v11_ranker")
V12_DIR = os.path.join(ROOT, "checkpoints", "v12_ade6")
RANKER12 = os.path.join(ROOT, "checkpoints", "v12_ranker")


def log(msg: str) -> None:
    print(msg, flush=True)


def run(args: list[str]) -> int:
    log(">> " + " ".join(args))
    proc = subprocess.run(args, cwd=ROOT)
    log(f"<< exit {proc.returncode}")
    return proc.returncode


def v11_finished() -> bool:
    if not os.path.isfile(V11_LOG):
        return False
    with open(V11_LOG, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return "V11 done" in text or "\nEpoch 30/30 (" in text or text.startswith("Epoch 30/30 (")


def v11_train_running() -> bool:
    try:
        raw = subprocess.check_output(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Select-Object -ExpandProperty CommandLine",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except Exception:
        return False
    return "train_motion_prediction_v11.py" in (raw or "")


def wait_v11() -> None:
    log("Waiting for V11 to finish (epoch 30)...")
    while True:
        if v11_finished() and not v11_train_running():
            log("V11 finished.")
            return
        if v11_finished() and v11_train_running():
            log("V11 log has epoch 30, waiting process exit...")
        time.sleep(30)


def cache_ready() -> bool:
    train = os.path.join(CACHE, "train")
    if not os.path.isdir(train):
        return False
    n = sum(1 for name in os.listdir(train) if name.endswith(".pt"))
    log(f"cache train packs={n}")
    return n >= 200000


def ranker_success(path: str) -> bool:
    p = os.path.join(path, "success.json")
    if not os.path.isfile(p):
        return False
    with open(p, "r", encoding="utf-8") as f:
        rec = json.load(f)
    log(f"ranker success.json: {rec}")
    return bool(rec.get("success"))


def main() -> int:
    os.chdir(ROOT)
    wait_v11()

    ckpt = V11_CKPT if os.path.isfile(V11_CKPT) else V11_LAST
    if not os.path.isfile(ckpt):
        log(f"ERROR no V11 ckpt at {ckpt}")
        return 1
    log(f"V11 ckpt: {ckpt}")

    if not cache_ready():
        rc = run([PY, "-u", os.path.join("data_tools", "cache_collate_v10.py"), "--workers", "8"])
        if rc != 0:
            log("cache failed")
            return rc
    else:
        log("cache already large enough, skip rebuild")

    rc = run([
        PY, "-u", "train_ranker.py",
        "--arch", "v11",
        "--ckpt", ckpt,
        "--out-dir", RANKER11,
        "--cache-root", CACHE,
    ])
    if rc != 0:
        log("V11 ranker failed")
        return rc

    ok = ranker_success(RANKER11)
    log(f"V11 ranker success={ok} → starting V12 with cache")

    rc = run([PY, "-u", "train_motion_prediction_v12.py", "--cache-root", CACHE])
    if rc != 0:
        log("V12 train failed")
        return rc

    v12_ckpt = os.path.join(V12_DIR, "best_minade6.pth")
    if not os.path.isfile(v12_ckpt):
        v12_ckpt = os.path.join(V12_DIR, "last.pth")
    if ok and os.path.isfile(v12_ckpt):
        log("V11 ranker worked → apply ranker to V12")
        rc = run([
            PY, "-u", "train_ranker.py",
            "--arch", "v12",
            "--ckpt", v12_ckpt,
            "--out-dir", RANKER12,
            "--cache-root", CACHE,
        ])
        return rc
    log("skip V12 ranker (V11 ranker not successful or no V12 ckpt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
