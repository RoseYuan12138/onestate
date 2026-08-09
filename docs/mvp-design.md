# OneState MVP design

## Claim tested by this version

The MVP tests one narrow claim: a single question-independent predicted object
state can be read by both a QA head and a visual decoder, while both objective
functions update the same dynamics predictor.

This is an architectural and gradient-flow MVP. It is not yet a CLEVRER result.

## Invariants

1. `SlotDynamicsPredictor` accepts history slots and an optional masking pattern;
   it never accepts question tokens.
2. The predictor performs one differentiable rollout and returns one
   `future_slots` tensor.
3. The QA head and renderer consume the exact same tensor.
4. Target future slots are detached supervision. Predicted future slots are not.
5. A frozen decoder may freeze its parameters, but its forward pass must not run
   under `torch.no_grad()` when decoding predictions.
6. Object identity is fixed across the synthetic horizon. Real SAVi slots will
   require explicit matching checks before using slot-wise losses.

## Current synthetic task

Each scene contains four persistent object slots with position, color, scale,
velocity, and identity dimensions. The objects move linearly. Questions select
the fastest, final rightmost, or final topmost object. A parameter-free Gaussian
renderer converts slots into RGB frames and masks.

The renderer is intentionally simple: it proves that observable-space loss can
flow through a frozen visual readout into the shared predictor. It will later be
replaced by the pretrained `StoSAVi.decode` adapter.

## Experiments

- `dyn_qa`: dynamics + masked-history + QA supervision.
- `dyn_gen`: dynamics + masked-history + RGB/mask supervision.
- `unified`: all of the above on one predictor.

The first real-data milestone will preserve this interface and replace only the
dataset and decoder:

```text
CLEVRER video -> frozen SAVi encoder -> history slots
history slots -> shared predictor -> future slots
future slots + question -> ALOE-style QA readout
future slots -> frozen SAVi decoder -> RGB/masks
```

## Exit criteria before CLEVRER

- Unit tests prove question-independent rollout.
- QA-only loss creates non-zero gradients on predictor parameters.
- generation-only loss creates non-zero gradients on predictor parameters.
- smoke training decreases joint loss and saves a reloadable checkpoint.
- all three ablation configurations run through the same training entry point.

