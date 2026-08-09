# OneState

One question-independent object-centric future state, shared by video question
answering and future-frame generation.

The current repository contains a runnable synthetic MVP for the proposal. It
is designed to validate architecture and gradient flow before spending GPU time
on CLEVRER.

## What is implemented

- A C-JEPA-style non-causal slot transformer with identity-anchored future queries.
- A strict question boundary: questions enter only the QA readout.
- A shared predicted `future_slots` tensor consumed by QA and generation.
- A differentiable, parameter-free renderer standing in for a frozen SAVi decoder.
- `dyn_qa`, `dyn_gen`, and `unified` loss-routing configurations.
- Tests showing that each downstream loss updates the shared predictor.

See [the MVP design](docs/mvp-design.md) and the
[C-JEPA study guide](docs/cjepa-study-guide.md).

## Environment

PyTorch does not yet support the system Python 3.14 used on this machine. Create
a Python 3.12 environment with `uv`:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]'
```

## Verify the MVP

```bash
pytest
python scripts/smoke_test.py
```

Run one experiment or all three ablations:

```bash
python scripts/train.py --config configs/unified.yaml
python scripts/train.py --config configs/dyn_qa.yaml
python scripts/train.py --config configs/dyn_gen.yaml
```

Evaluate a checkpoint on held-out synthetic scenes:

```bash
python scripts/evaluate.py outputs/unified/last.pt
```

For a very short check, add `--steps 10` to the training command.

## Repository policy

The repositories under `codebases/` are ignored local research references. The
actual implementation lives in `src/onestate` so experiments remain reproducible
and do not silently modify C-JEPA, VideoSAUR, or stable-worldmodel.
