from __future__ import annotations

import torch
from torch import nn


class ObjectQuestionHead(nn.Module):
    """Question-conditioned object readout over a precomputed future state."""

    def __init__(
        self,
        slot_dim: int,
        vocab_size: int,
        hidden_dim: int = 128,
        padding_index: int = 0,
    ) -> None:
        super().__init__()
        self.padding_index = padding_index
        self.question_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_index)
        self.question_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.slot_projection = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.object_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, future_slots: torch.Tensor, question_tokens: torch.Tensor) -> torch.Tensor:
        token_features = self.question_embedding(question_tokens)
        token_mask = question_tokens.ne(self.padding_index).unsqueeze(-1)
        question = (token_features * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        question = self.question_projection(question)

        # Keep object identity and pool only across the predicted time axis.
        objects = self.slot_projection(future_slots).mean(dim=1)
        expanded_question = question.unsqueeze(1).expand(-1, objects.shape[1], -1)
        interaction = objects * expanded_question
        features = torch.cat([objects, expanded_question, interaction], dim=-1)
        return self.object_scorer(features).squeeze(-1)

