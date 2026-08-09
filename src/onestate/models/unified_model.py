from __future__ import annotations

import torch
from torch import nn

from onestate.models.predictor import SlotDynamicsPredictor
from onestate.models.qa_head import ObjectQuestionHead
from onestate.models.renderer import GaussianSlotRenderer


class OneStateModel(nn.Module):
    """One question-independent predicted state with QA and visual readouts."""

    def __init__(
        self,
        slot_dim: int,
        num_slots: int,
        history_frames: int,
        future_frames: int,
        vocab_size: int,
        model_dim: int = 128,
        depth: int = 3,
        heads: int = 4,
        ffn_dim: int = 256,
        qa_hidden_dim: int = 128,
        image_size: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.predictor = SlotDynamicsPredictor(
            slot_dim=slot_dim,
            num_slots=num_slots,
            history_frames=history_frames,
            future_frames=future_frames,
            model_dim=model_dim,
            depth=depth,
            heads=heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.qa_head = ObjectQuestionHead(
            slot_dim=slot_dim,
            vocab_size=vocab_size,
            hidden_dim=qa_hidden_dim,
        )
        self.decoder = GaussianSlotRenderer(image_size=image_size)

    def predict_state(
        self, history_slots: torch.Tensor, history_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Question-independent state transition used by every downstream head."""
        return self.predictor(history_slots, history_mask=history_mask)

    def forward(
        self,
        history_slots: torch.Tensor,
        question_tokens: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = self.predict_state(history_slots, history_mask=history_mask)
        rendered = self.decoder(state["future_slots"])
        return {
            **state,
            "qa_logits": self.qa_head(state["future_slots"], question_tokens),
            "predicted_rgb": rendered["rgb"],
            "predicted_masks": rendered["masks"],
        }

