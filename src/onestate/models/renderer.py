from __future__ import annotations

import torch
from torch import nn


class GaussianSlotRenderer(nn.Module):
    """Small parameter-free differentiable renderer for synthetic slot states.

    It stands in for the frozen SAVi decoder during the local MVP. Because it
    does not detach its input, RGB and mask losses backpropagate into predicted
    future slots even though the rendering branch itself has no trainable weights.
    """

    def __init__(self, image_size: int = 32) -> None:
        super().__init__()
        coordinates = torch.linspace(-1.0, 1.0, image_size)
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("grid_x", grid_x)
        self.register_buffer("grid_y", grid_y)
        self.image_size = image_size

    def forward(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        if slots.shape[-1] < 6:
            raise ValueError("Renderer requires slot_dim >= 6")
        position = torch.tanh(slots[..., 0:2])
        color = torch.sigmoid(slots[..., 2:5])
        sigma = 0.10 + 0.18 * torch.sigmoid(slots[..., 5:6])

        x = position[..., 0].unsqueeze(-1).unsqueeze(-1)
        y = position[..., 1].unsqueeze(-1).unsqueeze(-1)
        squared_distance = (self.grid_x - x).square() + (self.grid_y - y).square()
        logits = -squared_distance / (2.0 * sigma.unsqueeze(-1).square())
        masks = torch.softmax(logits, dim=2)
        rgb = (masks.unsqueeze(3) * color.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        return {"rgb": rgb, "masks": masks}

