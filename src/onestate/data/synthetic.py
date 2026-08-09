from __future__ import annotations

import torch
from torch.utils.data import Dataset


QUESTION_VOCAB = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "fastest": 3,
    "rightmost": 4,
    "topmost": 5,
}


class MovingSlotsDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic moving-object data for the end-to-end MVP.

    Each slot keeps a stable object identity. Dimensions 0:2 are position,
    2:5 are color logits, 5 is scale, and 6:8 are velocity. Questions ask
    which object is fastest, rightmost, or topmost in the predicted future.
    """

    def __init__(
        self,
        num_samples: int = 512,
        history_frames: int = 3,
        future_frames: int = 2,
        num_slots: int = 4,
        slot_dim: int = 16,
        seed: int = 7,
    ) -> None:
        if slot_dim < 8:
            raise ValueError("slot_dim must be at least 8 for position/color/velocity state")
        self.num_samples = num_samples
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        total_frames = self.history_frames + self.future_frames

        initial_xy = torch.empty(self.num_slots, 2).uniform_(-0.55, 0.55, generator=generator)
        velocity = torch.empty(self.num_slots, 2).uniform_(-0.10, 0.10, generator=generator)
        colors = torch.empty(self.num_slots, 3).uniform_(-2.0, 2.0, generator=generator)
        scale = torch.empty(self.num_slots, 1).uniform_(-0.8, 0.8, generator=generator)
        identity = torch.empty(self.num_slots, self.slot_dim - 8).normal_(
            mean=0.0, std=0.25, generator=generator
        )

        states = []
        for time_index in range(total_frames):
            state = torch.zeros(self.num_slots, self.slot_dim)
            state[:, 0:2] = initial_xy + time_index * velocity
            state[:, 2:5] = colors
            state[:, 5:6] = scale
            state[:, 6:8] = velocity
            state[:, 8:] = identity
            states.append(state)
        states_tensor = torch.stack(states, dim=0)

        history = states_tensor[: self.history_frames]
        future = states_tensor[self.history_frames :]
        question_type = index % 3
        question_token = QUESTION_VOCAB[("fastest", "rightmost", "topmost")[question_type]]
        question_tokens = torch.tensor(
            [QUESTION_VOCAB["<bos>"], question_token, QUESTION_VOCAB["<eos>"]],
            dtype=torch.long,
        )

        if question_type == 0:
            answer = velocity.square().sum(dim=-1).argmax()
        elif question_type == 1:
            answer = future[-1, :, 0].argmax()
        else:
            answer = future[-1, :, 1].argmax()

        return {
            "history_slots": history,
            "future_slots": future,
            "question_tokens": question_tokens,
            "answer": answer.long(),
        }

