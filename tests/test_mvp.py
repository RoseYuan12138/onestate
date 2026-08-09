from __future__ import annotations

import torch
from torch.nn import functional as F

from onestate.data.synthetic import MovingSlotsDataset, QUESTION_VOCAB
from onestate.models.unified_model import OneStateModel


def make_model() -> OneStateModel:
    return OneStateModel(
        slot_dim=16,
        num_slots=4,
        history_frames=3,
        future_frames=2,
        vocab_size=len(QUESTION_VOCAB),
        model_dim=32,
        depth=1,
        heads=4,
        ffn_dim=64,
        qa_hidden_dim=32,
        image_size=12,
    )


def make_batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    dataset = MovingSlotsDataset(num_samples=batch_size)
    items = [dataset[index] for index in range(batch_size)]
    return {key: torch.stack([item[key] for item in items]) for key in items[0]}


def predictor_gradient_norm(model: OneStateModel) -> float:
    return sum(
        parameter.grad.abs().sum().item()
        for parameter in model.predictor.parameters()
        if parameter.grad is not None
    )


def test_output_shapes_and_question_independent_rollout() -> None:
    torch.manual_seed(1)
    model = make_model().eval()
    batch = make_batch()
    first = model(batch["history_slots"], batch["question_tokens"])
    changed_questions = batch["question_tokens"].roll(shifts=1, dims=0)
    second = model(batch["history_slots"], changed_questions)

    assert first["future_slots"].shape == (3, 2, 4, 16)
    assert first["qa_logits"].shape == (3, 4)
    assert first["predicted_rgb"].shape == (3, 2, 3, 12, 12)
    assert first["predicted_masks"].shape == (3, 2, 4, 12, 12)
    assert torch.equal(first["future_slots"], second["future_slots"])


def test_qa_loss_updates_shared_predictor() -> None:
    torch.manual_seed(2)
    model = make_model()
    batch = make_batch()
    outputs = model(batch["history_slots"], batch["question_tokens"])
    F.cross_entropy(outputs["qa_logits"], batch["answer"]).backward()
    assert predictor_gradient_norm(model) > 0.0


def test_generation_loss_updates_shared_predictor() -> None:
    torch.manual_seed(3)
    model = make_model()
    batch = make_batch()
    with torch.no_grad():
        target = model.decoder(batch["future_slots"])
    outputs = model(batch["history_slots"], batch["question_tokens"])
    F.l1_loss(outputs["predicted_rgb"], target["rgb"]).backward()
    assert predictor_gradient_norm(model) > 0.0
    assert list(model.decoder.parameters()) == []


def test_identity_anchor_cannot_be_masked() -> None:
    model = make_model()
    batch = make_batch(batch_size=1)
    invalid_mask = torch.zeros(1, 3, 4, dtype=torch.bool)
    invalid_mask[:, 0, 0] = True
    try:
        model.predict_state(batch["history_slots"], history_mask=invalid_mask)
    except ValueError as error:
        assert "identity anchor" in str(error)
    else:
        raise AssertionError("Expected first-frame masking to be rejected")

