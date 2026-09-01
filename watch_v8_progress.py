#!/usr/bin/env python3
"""Live monitor for v8 training progress. Stop with Ctrl+C."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

STEP_RE = re.compile(r"\[Epoch\s+(\d+)\s*/\s*\d+\].*?Step\s*\[(\d+)\s*/\s*([\d,]+)\]", re.S)
TRAIN_METRIC_RE = re.compile(
    r"Loss:\s*([\d.]+).*?minADE6:\s*([\d.]+)m\s*\|\s*minADE1:\s*([\d.]+)m",
    re.S,
)
VAL_METRIC_RE = re.compile(
    r"val_minADE6:\s*([\d.]+)m,\s*val_minADE1:\s*([\d.]+)m,\s*"
    r"val_minFDE6:\s*([\d.]+)m,\s*Error Score:\s*([\d.]+)",
    re.S,
)

def get_progress(log_path: Path, total_epochs: int) -> tuple[float, float, str, str, str]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return 0.0, 100.0, "로그 대기 중", "", ""
    matches = STEP_RE.findall(text)
    if not matches:
        return 0.0, 100.0, "학습 시작 대기 중", "", ""
    epoch, step, total_steps = matches[-1]
    epoch, step = int(epoch), int(step)
    total_steps = int(total_steps.replace(",", ""))
    total = total_epochs * total_steps
    completed = min(total, max(0, (epoch - 1) * total_steps + step))
    done = completed / total * 100.0
    detail = f"Epoch {epoch}/{total_epochs}, Step {step:,}/{total_steps:,}"
    train_matches = TRAIN_METRIC_RE.findall(text)
    train = ""
    if train_matches:
        loss, ade6, ade1 = train_matches[-1]
        train = f"Train | Loss {loss} | minADE6 {ade6}m | minADE1 {ade1}m"
    val_matches = VAL_METRIC_RE.findall(text)
    val = ""
    if val_matches:
        v_ade6, v_ade1, v_fde6, score = val_matches[-1]
        val = f"Val   | minADE6 {v_ade6}m | minADE1 {v_ade1}m | minFDE6 {v_fde6}m | Score {score}"
    return done, 100.0 - done, detail, train, val

def main() -> None:
    parser = argparse.ArgumentParser(description="v8 실시간 진행률 모니터")
    parser.add_argument("--log", default=r"checkpoints\v8_hardcls\console.log")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    log_path = Path(args.log)
    try:
        while True:
            done, remain, detail, train, val = get_progress(log_path, args.epochs)
            print("\033[2J\033[H", end="")
            print("V8 Motion Prediction 실시간 모니터")
            print("=" * 58)
            print(f"완료: {done:6.2f}%   남음: {remain:6.2f}%")
            print(detail)
            if train:
                print(train)
            if val:
                print(val)
            print(f"\n다음 갱신: {args.interval:g}초 후 | 종료: Ctrl+C")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n모니터를 종료했습니다.")

if __name__ == "__main__":
    main()
