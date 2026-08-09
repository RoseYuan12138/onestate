from __future__ import annotations

import torch


@torch.no_grad()
def batch_metrics(
    outputs: dict[str, torch.Tensor],
    future_slots: torch.Tensor,
    answers: torch.Tensor,
    target_rgb: torch.Tensor,
) -> dict[str, float]:
    return {
        "accuracy": outputs["qa_logits"].argmax(dim=-1).eq(answers).float().mean().item(),
        "slot_rmse": outputs["future_slots"].sub(future_slots).square().mean().sqrt().item(),
        "rgb_mae": outputs["predicted_rgb"].sub(target_rgb).abs().mean().item(),
    }

