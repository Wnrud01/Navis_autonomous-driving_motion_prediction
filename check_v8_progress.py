#!/usr/bin/env python3
"""Print only v8 training completion and remaining percentages."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--log",
        default=r"checkpoints\v8_hardcls\console.log",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=7177)
    args = parser.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"\[Epoch\s+(\d+)\s*/\s*\d+\].*?Step\s*\[(\d+)\s*/\s*([\d,]+)\]", text, re.S)

    if not matches:
        print("완료: 0.00%")
        print("남음: 100.00%")
        return

    epoch, step, total_steps = matches[-1]
    epoch = int(epoch)
    step = int(step)
    steps_per_epoch = int(total_steps.replace(",", ""))
    total_steps_all = args.epochs * steps_per_epoch
    completed = max(0, min(total_steps_all, (epoch - 1) * steps_per_epoch + step))
    done_pct = completed / total_steps_all * 100.0
    remain_pct = 100.0 - done_pct

    print(f"완료: {done_pct:.2f}%")
    print(f"남음: {remain_pct:.2f}%")


if __name__ == "__main__":
    main()
