#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from onestate.data.synthetic import MovingSlotsDataset  # noqa: E402
from onestate.metrics import batch_metrics  # noqa: E402
from onestate.training import build_model, choose_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a OneState MVP checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = choose_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data_config = config["data"]
    dataset = MovingSlotsDataset(
        num_samples=args.samples,
        history_frames=data_config["history_frames"],
        future_frames=data_config["future_frames"],
        num_slots=data_config["num_slots"],
        slot_dim=data_config["slot_dim"],
        seed=int(config.get("seed", 42)) + 100_000,
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"])
    totals = {"accuracy": 0.0, "slot_rmse": 0.0, "rgb_mae": 0.0}
    batches = 0
    for raw_batch in loader:
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        outputs = model(batch["history_slots"], batch["question_tokens"])
        target = model.decoder(batch["future_slots"])
        metrics = batch_metrics(
            outputs, batch["future_slots"], batch["answer"], target["rgb"]
        )
        for name, value in metrics.items():
            totals[name] += value
        batches += 1
    print(json.dumps({name: value / batches for name, value in totals.items()}, sort_keys=True))


if __name__ == "__main__":
    main()

