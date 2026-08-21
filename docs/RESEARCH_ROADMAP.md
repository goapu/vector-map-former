# Research roadmap

This roadmap separates implemented engineering from research hypotheses. A
milestone is complete only when its data contract, baseline, evaluation
protocol, and failure analysis are reproducible.

## Current foundation

- validated single-building MapGeneralizer adapter;
- coordinate and cyclic structural features;
- MLP, circular CNN, absolute-position Transformer, and ring-relative
  Transformer;
- joint action classification and edge-relative movement regression;
- constraint-aware reconstruction and geometric metrics;
- deterministic experiment artifacts and automated quality gates.

## Milestone 1: controlled supervised benchmark

Run every implemented architecture with identical features, training budgets,
and at least three seeds. Report per-class F1, balanced accuracy, movement
error, invalid-prediction rate, IoU, Hausdorff distance, parameter count, and
runtime. Select hyperparameters using validation data only.

**Acceptance criterion:** a machine-readable experiment table and statistical
summary that distinguish architectural effects from optimization variance.

## Milestone 2: explicit geometric constraints

Evaluate loss terms and decoding rules for minimum edge length, self-
intersection, orientation preservation, and excessive area change. Compare
hard constrained decoding with differentiable penalties.

**Acceptance criterion:** improved validity without concealing invalid cases or
materially degrading action macro-F1.

## Milestone 3: spatial-semantic context

Extend the sample from one ring to a focal building plus neighboring buildings,
nearby road segments, object classes, and target scale. Start with context
tokens before introducing a heterogeneous spatial graph.

**Acceptance criterion:** a context ablation on conflict-sensitive cases, with
explicit checks for building-road and building-building violations.

## Milestone 4: self-supervised vector pretraining

Pretrain the encoder on unlabelled building footprints using masked vertex and
edge reconstruction, cyclic-order prediction, and augmented-view contrastive
objectives. Prevent leakage by partitioning geographic regions before
pretraining and fine-tuning.

**Acceptance criterion:** better labelled-data efficiency than training the
same encoder from scratch across multiple label fractions.

## Milestone 5: hierarchical and generative generalization

Introduce segment-level pooling and an autoregressive edit decoder conditioned
on geometry, context, and target scale. Generate a structured edit sequence
before attempting unconstrained coordinate generation.

**Acceptance criterion:** valid closed-ring decoding with measurable gains over
parallel prediction and documented exposure-bias failures.

## Out of scope until the evidence exists

Object aggregation, network thinning, cross-city generalization, and full map
constraint satisfaction must not be claimed from the present single-building
benchmark. They require appropriate data and task-specific evaluation.
