#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from onestate.config import load_config  # noqa: E402
from onestate.training import run_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the OneState synthetic MVP")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/unified.yaml")
    parser.add_argument("--steps", type=int, default=None, help="Optional early stop for smoke runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(load_config(args.config), max_steps=args.steps)
    print(
        json.dumps(
            {
                "status": "complete",
                "steps": result["steps"],
                "device": result["device"],
                "checkpoint": result["checkpoint"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

