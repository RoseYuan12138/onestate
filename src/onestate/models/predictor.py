from __future__ import annotations

import torch
from torch import nn


class SlotDynamicsPredictor(nn.Module):
    """C-JEPA-style non-causal transformer over object slots.

    The predictor only consumes observed slots. Future positions are learned
    queries anchored by each object's first-frame slot; questions never enter
    this module. The forward pass is deliberately differentiable, unlike the
    ``@torch.no_grad`` inference helper in the reference C-JEPA repository.
    """

    def __init__(
        self,
        slot_dim: int,
        num_slots: int,
        history_frames: int,
        future_frames: int,
        model_dim: int = 128,
        depth: int = 3,
        heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_dim % heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.total_frames = history_frames + future_frames

        self.input_projection = nn.Linear(slot_dim, model_dim)
        self.anchor_projection = nn.Linear(slot_dim, model_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, model_dim))
        self.time_embedding = nn.Parameter(torch.empty(1, self.total_frames, 1, model_dim))
        self.slot_embedding = nn.Parameter(torch.empty(1, 1, num_slots, model_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(model_dim, slot_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)

    def forward(
        self,
        history_slots: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict future slots and reconstruct optionally masked history.

        Args:
            history_slots: ``(batch, history, slots, slot_dim)``.
            history_mask: Boolean tensor ``(batch, history, slots)`` where true
                entries are replaced by identity-anchored query tokens.
        """
        batch, history_frames, num_slots, slot_dim = history_slots.shape
        expected = (self.history_frames, self.num_slots, self.slot_dim)
        if (history_frames, num_slots, slot_dim) != expected:
            raise ValueError(
                "history_slots has shape "
                f"{tuple(history_slots.shape)}; expected (batch, {expected[0]}, {expected[1]}, {expected[2]})"
            )
        if history_mask is None:
            history_mask = torch.zeros(
                batch, history_frames, num_slots, dtype=torch.bool, device=history_slots.device
            )
        if history_mask.shape != history_slots.shape[:3]:
            raise ValueError("history_mask must match the first three history_slots dimensions")
        if history_mask[:, 0].any():
            raise ValueError("The first frame is the identity anchor and cannot be masked")

        time = self.time_embedding.expand(batch, -1, num_slots, -1)
        slot = self.slot_embedding.expand(batch, self.total_frames, -1, -1)
        anchors = self.anchor_projection(history_slots[:, 0]).unsqueeze(1)

        visible_history = self.input_projection(history_slots) + time[:, :history_frames] + slot[:, :history_frames]
        history_queries = self.mask_token + anchors + time[:, :history_frames] + slot[:, :history_frames]
        history_tokens = torch.where(history_mask.unsqueeze(-1), history_queries, visible_history)

        future_tokens = (
            self.mask_token
            + anchors
            + time[:, history_frames : self.total_frames]
            + slot[:, history_frames : self.total_frames]
        )
        tokens = torch.cat([history_tokens, future_tokens], dim=1)
        encoded = self.transformer(tokens.reshape(batch, self.total_frames * num_slots, -1))
        decoded = self.output_projection(encoded).reshape(
            batch, self.total_frames, num_slots, self.slot_dim
        )
        return {
            "history_slots": decoded[:, :history_frames],
            "future_slots": decoded[:, history_frames:],
        }
