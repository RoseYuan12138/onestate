from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F


def compute_losses(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    target_render: Mapping[str, torch.Tensor],
    history_mask: torch.Tensor,
    weights: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    """Compute all losses while weights determine gradient routing ablations."""
    slot = F.mse_loss(outputs["future_slots"], batch["future_slots"])
    qa = F.cross_entropy(outputs["qa_logits"], batch["answer"])
    rgb = F.l1_loss(outputs["predicted_rgb"], target_render["rgb"])
    masks = F.l1_loss(outputs["predicted_masks"], target_render["masks"])

    if history_mask.any():
        masked_history = F.mse_loss(
            outputs["history_slots"][history_mask], batch["history_slots"][history_mask]
        )
    else:
        masked_history = slot.new_zeros(())

    components = {
        "slot": slot,
        "masked_history": masked_history,
        "qa": qa,
        "rgb": rgb,
        "mask": masks,
    }
    total = sum(weights.get(name, 0.0) * value for name, value in components.items())
    return {"total": total, **components}

