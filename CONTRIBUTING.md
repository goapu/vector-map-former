# Contributing

VectorMapFormer is a research prototype with a deliberately strict boundary
between verified functionality and planned experiments.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[geometry,dev]'
```

Run the complete local quality gate before committing:

```bash
make check
```

## Change requirements

- Add or update tests for behavioral changes.
- Preserve padding and cyclic-order invariants for sequence models.
- Reject malformed data instead of truncating or silently coercing it.
- Keep raw datasets, checkpoints, and generated outputs out of source control.
- Record the resolved configuration, random seed, runtime, and data audit for
  every reported experiment.
- Distinguish smoke tests from converged benchmarks and hypotheses from
  implemented components.
- Use MOVE-only masking for movement loss unless the target contract changes
  explicitly.

## Pull requests

Keep changes focused and describe:

1. the research or engineering question;
2. the data and configuration used;
3. the tests and metrics affected;
4. known limitations or failure cases.

CI must pass Ruff linting and format checks, strict MyPy, and Pytest.
