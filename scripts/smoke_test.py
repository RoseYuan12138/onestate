#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from onestate.config import load_config  # noqa: E402
from onestate.training import run_training  # noqa: E402


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs/smoke.yaml")
    result = run_training(config, max_steps=30, verbose=False)
    if not result["history"]:
        raise RuntimeError("Smoke run produced no logged training summaries")
    first = result["history"][0]["total"]
    last = result["history"][-1]["total"]
    if not last < first:
        raise RuntimeError(f"Expected training loss to decrease, got first={first:.6f}, last={last:.6f}")
    print(
        json.dumps(
            {
                "status": "passed",
                "device": result["device"],
                "steps": result["steps"],
                "first_total": first,
                "last_total": last,
                "checkpoint": result["checkpoint"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

