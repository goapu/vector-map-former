# End-to-end smoke test

## Purpose

The smoke runs verify that the real MapGeneralizer arrays can pass through
validation, batching, training, checkpoint selection, safe reload, test
evaluation, geometry reconstruction, prediction export, and visualization.
They use one training epoch and are not converged benchmarks.

## Data audit

| Split | Buildings | Vertices | REMOVE | KEEP | MOVE | Maximum vertices |
|---|---:|---:|---:|---:|---:|---:|
| Train | 5,096 | 32,740 | 6,925 | 16,673 | 9,142 | 32 |
| Validation | 1,700 | 11,104 | 2,330 | 5,617 | 3,157 | 32 |
| Test | 1,697 | 10,950 | 2,287 | 5,583 | 3,080 | 32 |

No building ID occurs in more than one supplied split.

## One-epoch verification results

| Model | Parameters | Validation macro-F1 | Test macro-F1 | Valid predictions | Valid-only mean IoU | MOVE endpoint error |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 19,973 | 0.4073 | 0.4120 | 91.34% | 0.6306 | 0.1109 |
| Ring Transformer | 262,505 | 0.3630 | 0.3567 | 81.97% | 0.7254 | 0.0768 |

The MLP produced better action macro-F1 after one epoch, while the ring model's
valid outputs had better geometric IoU and movement endpoint error. No model
comparison should be inferred from a single epoch. A defensible benchmark
requires converged runs across multiple seeds with identical feature sets,
optimization budgets, and reported invalid-prediction rates.

## Verified quality gates

- Ruff linting and formatting checks;
- strict MyPy over the package;
- 22 unit and invariance tests;
- wheel construction without dependency resolution;
- real-data CLI inspection, training, checkpoint evaluation, and visualization.
