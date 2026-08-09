from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from onestate.data.synthetic import MovingSlotsDataset, QUESTION_VOCAB
from onestate.losses import compute_losses
from onestate.metrics import batch_metrics
from onestate.models.unified_model import OneStateModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: dict[str, Any]) -> OneStateModel:
    data = config["data"]
    model = config["model"]
    return OneStateModel(
        slot_dim=data["slot_dim"],
        num_slots=data["num_slots"],
        history_frames=data["history_frames"],
        future_frames=data["future_frames"],
        vocab_size=len(QUESTION_VOCAB),
        **model,
    )


def make_history_mask(
    batch_size: int,
    history_frames: int,
    num_slots: int,
    masked_slots: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(batch_size, history_frames, num_slots, dtype=torch.bool, device=device)
    if masked_slots <= 0 or history_frames <= 1:
        return mask
    masked_slots = min(masked_slots, num_slots)
    indices = torch.rand(batch_size, num_slots, device=device).argsort(dim=-1)[:, :masked_slots]
    for batch_index in range(batch_size):
        mask[batch_index, 1:, indices[batch_index]] = True
    return mask


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_training(
    config: dict[str, Any],
    max_steps: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = choose_device(config.get("device", "auto"))
    data_config = config["data"]
    train_config = config["training"]

    dataset = MovingSlotsDataset(
        num_samples=data_config["num_samples"],
        history_frames=data_config["history_frames"],
        future_frames=data_config["future_frames"],
        num_slots=data_config["num_slots"],
        slot_dim=data_config["slot_dim"],
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )

    step = 0
    history: list[dict[str, float]] = []
    running: defaultdict[str, float] = defaultdict(float)
    log_every = int(train_config.get("log_every", 25))
    epochs = int(train_config["epochs"])

    model.train()
    for epoch in range(epochs):
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            history_mask = make_history_mask(
                batch_size=batch["history_slots"].shape[0],
                history_frames=data_config["history_frames"],
                num_slots=data_config["num_slots"],
                masked_slots=int(data_config.get("masked_slots", 0)),
                device=device,
            )
            with torch.no_grad():
                target_render = model.decoder(batch["future_slots"])
            outputs = model(
                batch["history_slots"], batch["question_tokens"], history_mask=history_mask
            )
            losses = compute_losses(
                outputs, batch, target_render, history_mask, config["loss_weights"]
            )

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("grad_clip", 1.0)))
            optimizer.step()

            metrics = batch_metrics(
                outputs, batch["future_slots"], batch["answer"], target_render["rgb"]
            )
            values = {**{name: value.item() for name, value in losses.items()}, **metrics}
            for name, value in values.items():
                running[name] += value
            step += 1

            if step % log_every == 0:
                summary = {name: value / log_every for name, value in running.items()}
                summary.update({"step": step, "epoch": epoch + 1})
                history.append(summary)
                if verbose:
                    print(json.dumps(summary, sort_keys=True))
                running.clear()
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    output_dir = Path(config.get("output_dir", "outputs/mvp"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "steps": step,
            "history": history,
        },
        checkpoint_path,
    )
    return {
        "model": model,
        "history": history,
        "steps": step,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
    }

