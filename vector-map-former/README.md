# VectorMapFormer

A production-oriented research prototype for learning cartographic building
generalization directly from ordered vector coordinates. The repository starts
with auditable supervised baselines and a ring-relative Transformer. It is
inspired by the Cart2Former research direction but is not an implementation of
the DFG project.

## Implemented scope

- Strict MapGeneralizer object-array adapter and data audit.
- Translation/scale normalization recomputed from projected coordinates.
- Variable-length batching without truncation; maximum 32 vertices for the
  verified public splits.
- Three actions: `REMOVE`, `KEEP`, and `MOVE`.
- Movement regression along the two incident edges.
- MLP, circular CNN, vanilla Transformer, and cyclic-relative Transformer.
- Balanced classification loss, MOVE-only regression loss, early stopping,
  atomic checkpoints, deterministic loaders, and JSON experiment records.
- Classification, movement, polygon validity, IoU, Hausdorff, and area metrics.
- Qualitative source/target/prediction figures.
- Unit tests for data contracts, masking, circular behaviour, cyclic shift
  equivariance, padding invariance, loss masking, and reconstruction.

Self-supervised OpenStreetMap pretraining, spatial context, hierarchical
pooling, and autoregressive editing remain the next controlled milestones. They
should not be presented as completed work.

## Environment

Python 3.10–3.13 and PyTorch 2.x are supported by the package declaration.

```bash
cd vector-map-former
python3 -m pip install -e '.[geometry,dev]'
```

The current machine already contains the required runtime packages in the
Anaconda Python environment. The commands below can also be executed with
`PYTHONPATH=src` without installing the package.

## Dataset

Clone the public source outside this repository:

```bash
git clone https://github.com/chouisgiser/MapGeneralizer.git
```

Do not commit the downloaded arrays. See [data/README.md](data/README.md) for
the verified column contract, security warning, and redistribution caveat.

## Audit the data

```bash
vmf inspect \
  --data-dir "/absolute/path/MapGeneralizer/data/input" \
  --output outputs/data_audit.json
```

Verified local arrays contain 8,493 buildings and no building-ID overlap among
the supplied splits. The associated paper reports 8,494; the discrepancy is
recorded rather than silently repaired.

## Train

MLP smoke experiment:

```bash
vmf train \
  --config configs/baseline.yaml \
  --data-dir "/absolute/path/MapGeneralizer/data/input" \
  --model mlp \
  --epochs 2 \
  --output-dir outputs/mlp_smoke
```

Ring-relative Transformer:

```bash
vmf train \
  --config configs/baseline.yaml \
  --data-dir "/absolute/path/MapGeneralizer/data/input" \
  --model ring_transformer \
  --movement-weight 0.25 \
  --output-dir outputs/ring_transformer
```

Important artifacts:

```text
data_audit.json
resolved_config.json
runtime.json
history.json
best_model.pt
test_predictions.npz
result.json
```

## Evaluate and visualize

```bash
vmf evaluate \
  --checkpoint outputs/ring_transformer/best_model.pt \
  --data-dir "/absolute/path/MapGeneralizer/data/input" \
  --split test \
  --output outputs/ring_transformer/test_metrics.json
```

```bash
vmf visualize \
  --checkpoint outputs/ring_transformer/best_model.pt \
  --data-dir "/absolute/path/MapGeneralizer/data/input" \
  --split test \
  --examples 9 \
  --output outputs/ring_transformer/predictions.png
```

## Test

```bash
ruff check src tests
mypy src/vector_map_former
pytest
```

The repository-level GitHub Actions workflow runs the same lint, type, and test
quality gates on every project change.

## Scientific limitations

- The distributed NumPy files do not preserve authoritative CRS metadata.
- The public repository does not visibly provide a dataset licence, so the
  arrays are not redistributed here.
- The supplied normalized coordinates were generated with preprocessing
  statistics; this project recomputes per-polygon normalization from projected
  coordinates.
- This benchmark addresses single-building simplification. It does not yet
  establish object aggregation, contextual conflict resolution, network
  thinning, or free-coordinate autoregressive generation.
- Architectural components are research hypotheses. The repository reports
  controlled ablations and does not claim novelty or superiority without data.

## References

- Zhou, Z., Fu, C., and Weibel, R. (2023). *Move and remove: Multi-task
  learning for building simplification in vector maps with a graph
  convolutional neural network.*
  <https://doi.org/10.1016/j.isprsjprs.2023.06.004>
- Wamhoff, M., Baerenzung, J., Kaufhold, L., and Kada, M. (2025).
  *CNN-Based Geometric Feature Embedding Using Coordinates for Cartographic
  Generalization Tasks on Building Footprints.*
  <https://doi.org/10.5194/ica-proc-7-26-2025>
