# VectorMapFormer

## A context- and scale-conditioned hierarchical Transformer for geographic vector generalization

**Status:** Project design  
**Purpose:** TU Berlin Cart2Former interview preparation  
**Recommended implementation effort before interview:** 18–24 focused hours  
**Positioning:** A small research prototype inspired by Cart2Former, not an implementation or reproduction of the DFG project

---

## 1. One-sentence project description

VectorMapFormer learns directly from ordered geographic coordinates, spatial context, and a requested generalization strength; it uses self-supervised masked-ring pretraining and a Transformer to generate a sequence of `KEEP`, `REMOVE`, and later `MOVE` edit operations for building-footprint simplification.

## 2. Exact vacancy alignment

The TU Berlin vacancy VI-255/26 is a 36-month research position without teaching duties. It is not merely a Python implementation role. It explicitly requires independent scientific work, novel Transformer architectures for geographic vector data, heterogeneous training and benchmark datasets, model training/evaluation/benchmarking, scientific publication and presentation, and project reporting.

The public Cart2Former description adds the technical research directions: direct hierarchical feature extraction from geometry coordinates, positional/structural/context encodings, adaptive pooling, autoregressive generation, preservation of shape and symmetry, spatial-semantic context, and self-supervised pretraining.

### 2.1 Responsibility-to-evidence matrix

| Exact vacancy responsibility | Evidence the finished repository must contain |
|---|---|
| Independent scientific work | A research question, falsifiable hypotheses, literature comparison, experiment plan, ablations, failure analysis, and explicit limitations |
| Novel Transformer architectures | A clearly specified `RingContextFormer` block with cyclic relative attention, geometric bias, hierarchical segment pooling, and context cross-attention; novelty is evaluated, not asserted |
| Training/benchmark data from heterogeneous geodata | Reproducible adapters for Geofabrik GeoPackage and MapGeneralizer NumPy data, CRS/schema validation, provenance, data cards, and geographic split manifests |
| Train, evaluate, and benchmark AI models | MLP, circular CNN, parallel Transformer, and autoregressive edit-decoder comparisons using both learning and geometry metrics |
| Publish and present internationally | A four-page paper-style technical report, a five-slide research presentation, a three-minute live explanation, and publication-quality figures |
| Project and final reports | Reproducible experiment table, data/model limitations, milestone status, and a short final project report in the repository |

### 2.2 Candidate-profile-to-evidence matrix

| Vacancy profile | Your existing evidence | Evidence added by this project |
|---|---|---|
| Relevant Master's degree | M.Sc. Geodesy and Geoinformation Science, TU Berlin | Uses the degree's geospatial, coordinate-system, GIS, and research-method foundations |
| AI and neural networks | Vision Transformer and geometric computer-vision work | Transfers tokenization, attention, loss design, training, and ablation skills to vector geometry |
| PyTorch/TensorFlow | PyTorch experience | A tested PyTorch package with masking, attention, batching, checkpoints, and reproducible configs |
| Excellent Python | Python project work | Type-annotated modules, tests, CLI commands, logging, configuration, and clean error handling |
| Geodata processing and analysis | Geoinformatics education and GIS tools | GeoPackage ingestion, reprojection, topology checks, spatial joins, context construction, and spatial splits |
| Cartographic generalization advantageous | This is the honest current gap | Explicit operators, target scale/strength, shape constraints, baselines, and cartographic evaluation |
| Independent scientific work | Institut Pascal research and experimental evaluation | Hypothesis-driven experiments, negative-result reporting, a research report, and a defensible next-step plan |
| German and/or English | English-language degree/research communication | Clear English documentation and presentation; optional one-page German summary |

The repository cannot prove every three-year responsibility in ten days. Its purpose is to provide concrete evidence that you understand the research problem, can build the first rigorous pipeline independently, and can identify the next research questions without pretending that they are already solved.

## 3. Technical alignment with Cart2Former

This prototype implements a narrow but technically representative subset:

| Cart2Former direction | VectorMapFormer component |
|---|---|
| Direct learning from coordinates | Polygon rings are processed as ordered `(x, y)` sequences, without rasterization |
| Positional encoding | Cyclic Fourier encoding of each vertex's relative ring position |
| Structural encoding | Edge vectors, edge lengths, turning angle, convexity, and normalized coordinates |
| Self-supervised pretraining | Masked vertices and short masked spans are reconstructed |
| Heterogeneous geodata | Geofabrik GeoPackage adapters plus MapGeneralizer NumPy adapter; Berlin and New Zealand test geographic transfer |
| Shape simplification | Parallel per-vertex baseline and sequential edit-operation generation |
| Spatial-semantic context | Neighbour-building tokens, nearest-road token, semantic-type embeddings, and target generalization-strength token |
| Adaptive hierarchical pooling | Learned circular pairwise pooling creates segment-level and building-level representations |
| Generative modelling | Autoregressive decoder emits `KEEP`, `REMOVE`, `MOVE_PREV`, `MOVE_NEXT`, and `EOS` operations |
| Training and benchmarking | Rule-based, MLP, circular CNN, and Transformer comparisons |
| Geometric constraints | Validity-aware polygon reconstruction and explicit topology metrics |

The prototype does **not** claim to cover object aggregation, displacement, road-network thinning, or production-quality cartographic generalization.

## 4. Research questions

### Primary question

Does a cyclic, hierarchical, context-conditioned Transformer improve building-footprint edit prediction over local MLP/CNN and vanilla Transformer baselines?

### Secondary questions

1. Does masked-ring pretraining improve simplification performance when labelled data is limited?
2. Do cyclic relative attention and hierarchical segment pooling outperform a vanilla Transformer using raw `(x, y)` coordinates?
3. Does explicit context from neighbouring buildings and roads improve predictions near spatial conflicts?
4. Does an autoregressive edit decoder produce more coherent polygons than independent parallel vertex decisions?
5. How much representation quality transfers between Berlin and New Zealand?
6. Does a model with high classification accuracy also produce geometrically valid and cartographically usable polygons?

### Hypotheses

- Structural encoding and cyclic relative-attention bias will improve `REMOVE`-class F1 over an `(x, y)`-only Transformer.
- Hierarchical segment pooling will most benefit long or geometrically complex rings.
- Self-supervised pretraining will help most when only 10–25% of labelled samples are used.
- Context will have limited average benefit for isolated simplification but larger benefit on a deliberately selected conflict subset.
- Autoregressive action prediction will reduce invalid combinations of local edit decisions, at the cost of slower inference.
- Cross-region representation quality will be lower without pretraining on both geographic regions.
- Classification metrics alone will overestimate quality; validity, Hausdorff distance, IoU, and area change will reveal additional failures.

## 5. Data

### 5.1 Unlabelled pretraining and transfer data

Use the GeoPackage versions only. The shapefile folders duplicate the same extracts.

#### Berlin

```text
/Users/gmac/Desktop/Job Interview/gis/data/Berlin/
  berlin-260820-free.gpkg/berlin.gpkg
```

- Layer: `gis_osm_buildings_a_free`
- Features: 537,851
- Geometry: `MULTIPOLYGON`
- Source coordinates: WGS84 longitude/latitude
- Working CRS: EPSG:25833

#### New Zealand

```text
/Users/gmac/Desktop/Job Interview/gis/data/New/
  new-zealand-260820-free.gpkg/new-zealand.gpkg
```

- Layer: `gis_osm_buildings_a_free`
- Features: 2,463,985
- Geometry: `MULTIPOLYGON`
- Source coordinates: WGS84 longitude/latitude
- Working CRS for the main islands: EPSG:2193

Both extracts are based on OpenStreetMap and their supplied README files identify the licence as ODbL 1.0. Do not commit either raw dataset to GitHub. Include download and attribution instructions in `data/README.md`.

For context construction, read these additional layers where needed:

- `gis_osm_roads_free`
- `gis_osm_landuse_a_free`
- building `type` as an optional semantic value with an explicit `UNKNOWN` category

Berlin and New Zealand are geographically heterogeneous, but they are still from the same OpenStreetMap/Geofabrik source family. Do not incorrectly describe them as two independent source types.

### 5.2 Labelled fine-tuning data

#### Preferred option: MapGeneralizer

Use the public MapGeneralizer arrays locally:

- `vertex_train.npy`
- `vertex_valid.npy`
- `vertex_test.npy`
- Column 0: building/OSM ID
- Column 1: vertex order
- Columns 4–5: normalized coordinates
- Columns 6–9: turning angle, convexity, and adjacent edge lengths
- Column 10: removal label
- Columns 11–12: movement labels

The repository does not visibly provide a clear licence file. Cite the authors and repository, do not redistribute its arrays in this project, and describe how an authorized user can place them locally.

The supervised benchmark from Zhou et al. contains 8,494 Stuttgart buildings generalized from 1:5,000 to 1:10,000. Retain its provided train/validation/test divisions unless a documented reason requires otherwise.

Together, the GeoPackage and NumPy adapters demonstrate practical heterogeneity in file format, schema, coordinate representation, annotation availability, and geographic coverage. A future publication-quality benchmark should add an independently produced authoritative map source with clear redistribution terms.

#### Fallback option: synthetic proxy labels

Generate simplified targets from OSM polygons with a topology-preserving Douglas–Peucker operation at one or more tolerances, for example 0.5 m, 1.0 m, and 2.0 m.

These labels are appropriate for testing the pipeline but are **not** professional cartographic ground truth. Because the labelling algorithm itself is available, its output is an oracle rather than a baseline the learned model should be expected to beat.

Use this exact limitation statement in the report:

> The synthetic target labels encode the decisions of one rule-based simplification procedure. The experiment tests representation learning and algorithm imitation, not whether the model exceeds expert cartographic generalization.

### 5.3 Recommended sample sizes

Keep the experiment deliberately small:

| Stage | Region | Samples |
|---|---|---:|
| Self-supervised pretraining | Berlin | 30,000–50,000 |
| Self-supervised pretraining | New Zealand | 30,000–50,000 |
| Supervised training | Labelled data | 10,000–20,000 |
| Supervised validation/test | Official MapGeneralizer splits | As supplied |
| Synthetic pipeline test | Berlin | 2,000–3,000 |
| Synthetic external transfer test | New Zealand | 5,000 |

Do not process all 2.46 million New Zealand buildings for the interview prototype.

Keep the results in two clearly labelled tracks:

1. **Professional supervised track:** MapGeneralizer Stuttgart labels and official splits.
2. **OSM proxy/self-supervised track:** Berlin/New Zealand masked geometry, context denoising, and optional synthetic simplification labels.

Do not present synthetic New Zealand results as an external test of professional cartographic generalization.

## 6. Data preparation

### 6.1 Geometry cleaning

For every source:

1. Read only the building layer and required attributes.
2. Reproject into a metric coordinate reference system.
3. Explode multipolygons into individual polygon parts.
4. Remove null, empty, and invalid geometries; log all removal counts.
5. Initially exclude polygons with interior rings to keep reconstruction unambiguous.
6. Remove repeated closing coordinate before tokenization; restore it after prediction.
7. Remove consecutive duplicate vertices.
8. Normalize orientation to counter-clockwise.
9. Retain approximately 8–128 unique exterior vertices.
10. Apply defensible area limits, for example 25–10,000 square metres, and report them.

For New Zealand, first restrict the experiment to the main islands. The full extract contains remote islands and crosses the antimeridian. Do not project the complete geographic extent blindly into a single mainland CRS.

### 6.2 Per-polygon normalization

For polygon vertices `p_i`:

```text
c = polygon centroid
s = max(width, height, epsilon)
p_i_normalized = (p_i - c) / s
```

Store `c` and `s` so predictions can be transformed back to metric coordinates.

This makes the coordinate representation translation- and scale-normalized while retaining the original coordinates for metric evaluation.

### 6.3 Cyclic sequence handling

A polygon ring has no naturally privileged first vertex. Address this explicitly:

- During training, randomly rotate the starting vertex.
- Keep vertex order and counter-clockwise orientation consistent.
- During evaluation, use a deterministic starting vertex for reproducibility.
- Use cyclic relative-position features rather than relying only on an integer index.

This is an important interview discussion point: an ordinary sequence Transformer otherwise treats two cyclic shifts of the same polygon as different inputs.

### 6.4 Spatially separated splits

Do not randomly split neighbouring buildings row by row.

- Assign buildings to fixed geographic grid cells.
- Allocate entire cells to train, validation, and test sets.
- Save a split manifest with a fixed random seed.
- Keep all New Zealand samples out of Berlin training when measuring cross-region transfer.

This prevents spatial leakage from nearly identical neighbouring footprints.

### 6.5 Source adapters and data contracts

Implement two loaders that emit the same internal sample contract:

```text
geometry_id
source_id
region
metric_crs
ordered_vertices
semantic_attributes
context_objects
target_actions (optional)
target_strength_or_scale (optional)
```

Every adapter must validate schema, coordinate reference system, geometry type, sequence order, and label availability. Save a machine-readable data card containing source URL, extract date, licence, filters, counts, and rejected-geometry reasons.

## 7. Token representation

For each vertex `i`, construct a feature vector:

```text
x_i, y_i                       normalized coordinate
dx_previous, dy_previous       incoming edge vector
dx_next, dy_next               outgoing edge vector
length_previous, length_next   normalized adjacent edge lengths
sin(turn_angle), cos(turn_angle)
convexity                      local orientation sign
ring_position_fourier          cyclic position features
```

The cyclic position features can begin with:

```text
sin(2*pi*i/n), cos(2*pi*i/n),
sin(4*pi*i/n), cos(4*pi*i/n)
```

Project the feature vector to `d_model` with a two-layer MLP.

### Padding

- Pad variable-length rings to the maximum length in each batch.
- Pass a key-padding mask to every Transformer block.
- Exclude padding tokens from every loss and metric.
- Add unit tests that deliberately change padding values and confirm that valid-token outputs remain unchanged.

## 8. Model architecture

```text
target polygon vertices ---------> vertex/edge tokenizer
                                          |
neighbour buildings + roads -----> context tokenizer
                                          |
generalization strength ---------> condition token
                                          |
                                          v
              cyclic local-global RingFormer blocks
                                          |
                 learned circular segment pooling
                                          |
                     multi-resolution features
                                          |
                +-------------------------+--------------------+
                |                         |                    |
                v                         v                    v
       masked-ring head          parallel edit head    autoregressive
       (pretraining)             (strong baseline)     edit decoder
                                                               |
                         KEEP / REMOVE / MOVE / EOS operations
                                                               |
                                    constrained reconstruction
                                                               |
                           topology + cartographic evaluation
```

### 8.1 Proposed research block: RingContextFormer

The architecture candidate is not simply a standard encoder renamed for geodata. Each block tests four vector-specific design choices:

1. **Cyclic relative attention:** attention logits receive a learned bias based on shortest ring distance `min(|i-j|, n-|i-j|)`. This makes the first and final vertices local neighbours.
2. **Geometric relative bias:** normalized Euclidean distance, relative bearing, and edge relationship contribute an additional pairwise bias.
3. **Local-global attention:** alternating blocks use a small cyclic neighbourhood for efficient local shape learning and full attention for long-range symmetry and shape characteristics.
4. **Context cross-attention:** target-vertex features attend to neighbouring-building, road, semantic, and generalization-strength tokens.

Call it a **candidate architectural contribution**, not a proven novel method. The ablation study must determine whether each component improves a relevant metric.

### 8.2 Hierarchical feature extraction and adaptive pooling

Create two coordinate resolutions:

```text
Level 0: one token per original vertex
Level 1: one adaptively pooled token per adjacent vertex pair/short segment
Level 2: one attention-pooled building token
```

For each circular pair, learn a mixing weight:

```text
alpha = sigmoid(MLP([h_i, h_(i+1), edge_features]))
h_segment = alpha * h_i + (1-alpha) * h_(i+1)
```

Run attention at both vertex and segment levels, then send segment/global information back to vertex tokens through cross-attention. This is a modest, testable interpretation of hierarchical coordinate feature extraction and adaptive pooling.

The first implementation may replace learned pooling with fixed pairwise averaging. The learned version becomes one controlled ablation.

### 8.3 Target-scale or generalization-strength conditioning

Provide one condition token representing the intended output detail. With professional paired maps, encode the actual target scale. With synthetic proxy targets, call this a **generalization-strength token**, not a map scale, and encode the tolerance category.

This lets one model learn multiple output-detail levels and prepares a direct answer to: “How can one model generalize a polygon differently for different target scales?”

### 8.4 Recommended compact configuration

```yaml
d_model: 128
n_heads: 4
n_layers: 4
feedforward_dim: 384
dropout: 0.10
normalization: pre_norm
activation: gelu
max_vertices: 128
```

This is intentionally small enough to train on a laptop or modest GPU. Add architecture components one at a time only after the vanilla encoder passes tests.

### 8.5 Building-level attention pooling

Use one learned query to attend over the unpadded vertex embeddings and produce a polygon-level embedding. Use it for:

- Building-type prediction as an auxiliary pretraining task where reliable labels exist.
- A global feature concatenated back into the per-vertex simplification head.
- Visualization of building embeddings with UMAP as an optional analysis.

## 9. Self-supervised pretraining

### Core task: masked geometry modelling

Mask approximately 20% of vertices, mixing isolated masks and short contiguous spans. Replace their geometric input with a learned mask token while preserving the padding mask.

Predict:

- Normalized `(x, y)` coordinate or local edge deltas.
- `sin(angle)` and `cos(angle)`.
- Adjacent normalized edge lengths.

Recommended loss:

```text
L_ssl = SmoothL1(coordinates)
      + 0.25 * SmoothL1(edge_lengths)
      + 0.25 * angle_loss
```

Evaluate reconstruction error only at masked, non-padding positions.

### Context-denoising task

Create a second self-supervised task from OSM scenes:

1. Keep neighbouring buildings and roads fixed.
2. Apply a small translation to the target building, sometimes creating an overlap or minimum-distance conflict.
3. Ask the context-conditioned model to predict the inverse displacement or a discrete correction direction plus magnitude.
4. Evaluate displacement error and the percentage of corrected scenes that are conflict-free.

This is synthetic denoising, not expert displacement generalization, but it gives the spatial-context encoder a task for which context is genuinely necessary.

### Cross-region pretraining protocol

Compare the same downstream supervised model initialized from:

```text
random weights
Berlin-only masked-ring pretraining
Berlin + New Zealand masked-ring pretraining
```

Fine-tune all three on the same Stuttgart labelled split. This tests whether geographically broader unlabelled vector data improves transfer without pretending that New Zealand has professional target labels.

### Augmentations

Use transformations that preserve the footprint's identity:

- Random cyclic shift of the starting vertex.
- Small coordinate jitter before normalization.
- Rotation augmentation if the downstream task should be rotation robust.
- Mild scale augmentation before normalization.

Do not use destructive augmentations that create self-intersections without explicitly labelling them as corruptions.

## 10. Supervised simplification

### 10.1 Strong parallel baseline

For each input vertex, predict:

```text
KEEP or REMOVE
```

Use class-weighted cross-entropy or focal loss because most vertices may be retained.

This head predicts all vertex decisions simultaneously. It is the stable baseline against which the sequential decoder is compared.

### 10.2 Movement task

If authorized MapGeneralizer movement labels can be used, add a second head predicting vertex displacement:

```text
delta_x, delta_y
```

The MapGeneralizer formulation predicts whether a retained vertex is kept or moved along one of its incident edges. Preserve that structure rather than converting every movement into unconstrained Cartesian regression:

```text
action class: KEEP, REMOVE, MOVE_PREV, MOVE_NEXT
movement magnitude: one scalar along the selected incident edge
```

Apply the regression loss only to vertices with valid movement targets. If label semantics cannot be verified, omit movement rather than guessing.

### 10.3 Combined loss

```text
L = L_action + lambda_move * L_magnitude
```

Tune `lambda_move` on validation data and report both task-specific and reconstructed-geometry results. Establish the `KEEP`/`REMOVE` pipeline first, but the final research blueprint retains the movement task because displacement is central to cartographic generalization and to the Kada group's recent line of work.

## 11. Constraint-aware polygon reconstruction

Convert vertex probabilities into a polygon using the original cyclic order.

1. Keep vertices whose `KEEP` probability exceeds the selected validation threshold.
2. If fewer than three unique vertices remain, restore vertices with the highest `KEEP` probabilities.
3. Close the exterior ring.
4. Test polygon validity and self-intersection.
5. If invalid, reinsert removed vertices in decreasing uncertainty order until the polygon becomes valid.
6. Record whether repair was necessary; never silently hide failures.

This is not a complete cartographic constraint solver. It demonstrates that geometric validity is part of the model system rather than an afterthought.

## 12. Spatial-semantic context module

Context is part of the job/project title, so it must be represented in the project design and in at least one controlled experiment. Implement it only after the geometry-only model works, but do not leave it as an unexplained “future idea.”

For each target building, collect up to 16 neighbours within a fixed metric radius. Encode each neighbour as a summary token containing:

```text
relative centroid x/y
log area
perimeter
vertex count
optional building type
```

From the Geofabrik `gis_osm_roads_free` layer, add a nearest-road context token containing:

```text
relative nearest-point x/y
distance to road
road class embedding
road orientation near the building
```

Add an `UNKNOWN` semantic category because many OSM building types are missing, especially in the New Zealand extract. Do not let semantic missingness remove samples.

Allow target vertices to cross-attend to context tokens. Compare against the identical model with context tokens masked out. Also construct a **conflict subset** containing buildings close to roads or close neighbours; average performance alone may conceal the context module's actual effect.

This supports a modest context experiment without attempting full multi-object aggregation or displacement.

## 13. Autoregressive edit decoder

The full Cart2Former direction emphasizes sequential generation. Free coordinate generation is too fragile for a short interview prototype, so use an autoregressive **edit sequence** aligned with the input ring:

```text
<BOS>, KEEP, REMOVE, MOVE_PREV(delta), MOVE_NEXT(delta), ..., <EOS>
```

The encoder processes the detailed polygon and context. The causal decoder predicts the action at vertex `i` conditioned on the detailed geometry, target strength, context, and previous edit decisions. Use teacher forcing during training and greedy decoding for the first evaluation.

This formulation is deliberately connected to prior keep/remove/move work while adding a sequential generative dependency. It avoids the target-start and continuous-coordinate matching problem of a completely free polygon decoder.

Compare:

```text
parallel head:       p(action_i | full input)
autoregressive head: p(action_i | full input, action_<i)
```

Measure whether the autoregressive decoder improves geometry coherence or validity enough to justify slower inference.

### 13.1 Free-coordinate generation as future work

A later decoder may generate `<BOS>, (x_1,y_1), ..., (x_k,y_k), <EOS>`. That requires cyclic target alignment, continuous-coordinate loss design, stopping criteria, topology constraints, and exposure-bias analysis. Explain these challenges in the report; do not implement them before the edit decoder and benchmarks are reliable.

## 14. Baselines

Implement at least three comparisons:

1. **Rule-based:** Douglas–Peucker or Shapely simplification.
2. **Vertex MLP:** processes every vertex independently using the same engineered local features.
3. **Circular 1D CNN:** kernel size 3 with circular padding, inspired by the Kada-group coordinate-embedding work.
4. **Vanilla Transformer:** absolute/learned position encoding and no special ring bias.
5. **RingContextFormer:** cyclic/geometric bias, hierarchy, context, and the same parameter budget where practical.
6. **Published GCN reference:** the MapGeneralizer MT_GCNN result or a carefully documented rerun when its older software environment can be reproduced safely.

If synthetic labels were generated by Douglas–Peucker, identify the rule-based result as the label generator/oracle, not as a fair learned baseline.

Do not spend more than two hours repairing the historical MT_GCNN environment. If it cannot be reproduced, explain the compatibility issue and distinguish published numbers from your own runs.

## 15. Experiments and ablations

Run experiments in this order:

| ID | Model | Purpose |
|---|---|---|
| E0 | Rule-based generator | Establish target behaviour and geometry metrics |
| E1 | MLP + local features | Non-contextual learned baseline |
| E2 | Circular CNN | Strong local-sequence baseline |
| E3 | Transformer with `(x, y)` only | Basic attention model |
| E4 | E3 + cyclic/structural bias | Test ring-specific representation design |
| E5 | E4 + hierarchical segment pooling | Test multi-resolution coordinate features |
| E6 | E5 + masked-ring pretraining | Test self-supervised transfer |
| E7 | E6 + neighbour/road context | Test spatial-semantic context, especially on conflict subset |
| E8 | E7 + autoregressive edit decoder | Compare sequential and parallel edit decisions |

For E5 and E6, also compare 10%, 25%, and 100% of the labelled training set. This directly tests whether self-supervised pretraining helps under label scarcity.

Do not claim `RingContextFormer` is novel or better unless the relevant baselines, parameter counts, training budget, and ablations support the claim. A negative result is scientifically useful if the experiment is controlled and the explanation is thoughtful.

## 16. Evaluation

### Vertex classification

- Macro-F1
- Per-class precision, recall, and F1 for `KEEP`, `REMOVE`, and verified movement classes
- Balanced accuracy
- Confusion matrix

### Movement and context

- Movement magnitude mean absolute error in metres
- Displacement-vector endpoint error
- Conflict-free percentage after context denoising
- Minimum building-building and building-road clearance after correction

### Polygon geometry

- Vertex reduction percentage
- Intersection-over-Union
- Hausdorff distance in metres
- Relative area change
- Relative perimeter change
- Valid polygon percentage
- Self-intersection percentage

### Generalization and efficiency

- Stuttgart professional supervised performance using the official splits
- Masked-ring reconstruction error by region
- Downstream difference between random, Berlin-only, and Berlin-plus-New-Zealand pretraining
- Separately labelled synthetic Berlin-to-New-Zealand proxy performance, if run
- Inference milliseconds per polygon
- Trainable parameter count
- Peak memory use

Always report the number of polygons for which reconstruction failed or required repair.

## 17. Repository design

```text
vector-map-former/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── configs/
│   ├── pretrain.yaml
│   ├── finetune.yaml
│   ├── context_denoising.yaml
│   └── evaluate.yaml
├── data/
│   ├── README.md                 # downloads, licences, checksums
│   ├── data_card.md
│   └── processed/              # ignored by Git
├── src/vector_map_former/
│   ├── data/
│   │   ├── geofabrik_adapter.py
│   │   ├── mapgeneralizer_adapter.py
│   │   ├── geometry.py
│   │   ├── features.py
│   │   ├── context.py
│   │   ├── splits.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── mlp.py
│   │   ├── circular_cnn.py
│   │   ├── vanilla_transformer.py
│   │   ├── ring_context_former.py
│   │   ├── pooling.py
│   │   ├── edit_decoder.py
│   │   └── heads.py
│   ├── pretrain.py
│   ├── finetune.py
│   ├── evaluate.py
│   ├── reconstruct.py
│   └── visualize.py
├── tests/
│   ├── test_geometry.py
│   ├── test_cyclic_features.py
│   ├── test_padding_mask.py
│   ├── test_context.py
│   ├── test_source_adapters.py
│   ├── test_splits.py
│   └── test_reconstruction.py
├── reports/
│   ├── technical_report.pdf
│   ├── experiment_summary.md
│   └── limitations.md
├── presentation/
│   └── vector_map_former_interview.pdf
└── outputs/
    ├── figures/
    └── metrics/                # generated results, mostly ignored
```

### Reproducibility requirements

- One command per preprocessing, training, and evaluation stage.
- All seeds set from configuration.
- Configuration copied into every experiment output directory.
- Dataset source, extraction date, CRS, filters, and sample counts logged.
- Every results row linked to a configuration and source-control revision.
- No absolute personal paths in committed configuration files.
- Unit tests for geometry orientation, cyclic shifts, padding masks, source adapters, context joins, spatial splits, and reconstruction.
- Model checkpoints and raw datasets excluded from Git.
- Automated formatting, linting, tests, and a small CPU smoke test in continuous integration.

## 18. README result presentation

The first screen of the README should contain:

1. A two-sentence research question.
2. One architecture diagram.
3. One before/target/prediction figure.
4. A compact result table.
5. Exact reproduction commands.
6. A limitations box.

Example result table structure:

| Model | Remove F1 | IoU | Hausdorff m | Valid % | Parameters |
|---|---:|---:|---:|---:|---:|
| MLP | TBD | TBD | TBD | TBD | TBD |
| Circular CNN | TBD | TBD | TBD | TBD | TBD |
| Vanilla Transformer | TBD | TBD | TBD | TBD | TBD |
| RingContextFormer | TBD | TBD | TBD | TBD | TBD |
| RingContextFormer + SSL | TBD | TBD | TBD | TBD | TBD |

Never invent or pre-fill metrics.

### Research communication package

Prepare the outputs that mirror the vacancy's publication, presentation, and reporting responsibilities:

- A four-page paper-style report: problem, related work, method, data, experiments, limitations, next work.
- A five-slide deck: motivation, representation, architecture, evidence, research roadmap.
- One architecture figure and one qualitative error-analysis figure suitable for a paper.
- A data card and a compact model card.
- A 150-word conference-style abstract.
- A one-page milestone/final report distinguishing completed, failed, and planned work.

## 19. Implementation schedule

### Session 1: heterogeneous data pipeline — 4 hours

- Create repository and environment.
- Implement the Geofabrik and MapGeneralizer adapters.
- Implement projection, schema checks, cleaning, ring extraction, and split validation.
- Generate a small cached processed dataset.
- Produce data cards and ten geometry QA plots.

### Session 2: defensible supervised benchmark — 4 hours

- Implement MLP and circular CNN baselines.
- Implement the vanilla Transformer and parallel action head.
- Implement batching and padding masks.
- Add classification and geometric metrics.
- Add reconstruction tests.

### Session 3: vector-specific architecture — 4–5 hours

- Add cyclic/geometric relative-attention bias.
- Add fixed then learned segment pooling.
- Run E3–E5 with matched training budgets.
- Record an ablation table and one negative/failure result.

### Session 4: self-supervised and cross-region learning — 4 hours

- Add Berlin/New Zealand masked-geometry data.
- Pretrain the encoder.
- Fine-tune pretrained and randomly initialized models under identical settings.
- Compare random, Berlin-only, and Berlin-plus-New-Zealand initialization.

### Session 5: context/generation signature experiment — 3–5 hours

- Build neighbour/road tokens and one context-denoising smoke test.
- If stable, implement a binary autoregressive `KEEP`/`REMOVE` decoder.
- Keep movement classes and full context ablations as the documented next milestone if labels or time block them.

### Session 6: scientific communication — 3 hours

- Create result table and qualitative figures.
- Write the four-page report, data/model limitations, and 150-word abstract.
- Prepare five slides.
- Rehearse a three-minute project explanation.

The non-negotiable interview deliverable is Sessions 1–3 plus honest results. Sessions 4–6 add much stronger vacancy alignment, but a reliable benchmark with a thoughtful architecture ablation is preferable to an unfinished claim of a complete generative system.

## 20. Definition of done

The project is interview-ready when:

- [ ] One preprocessing command creates a documented processed dataset.
- [ ] GeoPackage and NumPy source adapters validate their data contracts.
- [ ] Official supervised and geographic proxy splits are not confused.
- [ ] MLP, circular CNN, vanilla Transformer, and one ring-specific variant run end to end.
- [ ] Padding and cyclic sequence handling have unit tests.
- [ ] Predictions reconstruct into polygons.
- [ ] Classification and geometric metrics are reported.
- [ ] At least twenty prediction figures are generated automatically.
- [ ] Limitations distinguish synthetic targets from expert ground truth.
- [ ] Cross-region pretraining comparison is included or clearly marked as pending.
- [ ] Context denoising or autoregressive edit generation has at least a tested smoke experiment or an explicit blocked-status explanation.
- [ ] A four-page report and five-slide presentation summarize the research evidence.
- [ ] README commands run from a clean environment.

## 21. Three-minute interview narrative

Use completed-work language only for components that actually run. Replace “built” with “designed” for unfinished components.

> I developed a controlled research prototype inspired by the position rather than trying to reproduce the full Cart2Former project. It works directly on ordered building-footprint coordinates, so the representation problem is different from the image patches I used with Vision Transformers. I introduced cyclic relative attention because a polygon has local structure but no naturally privileged starting vertex, and I tested a small hierarchical pooling block to learn both vertex- and segment-level features.
>
> I separated the data into a professional supervised track and a self-supervised OSM track. The pipeline reads heterogeneous GeoPackage and NumPy schemas, validates CRS and geometry, and records provenance. I used unlabelled Berlin and New Zealand footprints for masked-ring pretraining, then fine-tuned on the labelled Stuttgart simplification data. I compared an MLP, a circular CNN, a vanilla Transformer, and the ring-specific variant under controlled settings.
>
> To explore the generative aspect without pretending that free coordinate generation is already solved, I formulated simplification as an autoregressive sequence of keep, remove, and movement operations. I also designed context tokens for neighbouring buildings and roads and a synthetic context-denoising task where context is genuinely necessary. I evaluated classification together with polygon validity, IoU, Hausdorff distance, area change, movement error, and inference cost.
>
> The project connects my existing strengths in geoinformatics, PyTorch, Vision Transformers, geometric registration, and quantitative evaluation to this new vector domain. Its main limitation is that OSM synthetic labels are not professional cartographic ground truth. I therefore keep those results separate and use them only for pipeline or self-supervised experiments. The next scientific step is a larger, licensed multiscale benchmark and fully autoregressive coordinate generation with explicit topology constraints.

## 22. Likely questions this project prepares you to answer

1. Why is a polygon not equivalent to a normal text sequence?
2. How do you encode cyclic order and remove dependence on the starting vertex?
3. Why process coordinates directly instead of rasterizing polygons?
4. What is the purpose of masked-geometry pretraining?
5. How do you handle polygons with different numbers of vertices?
6. How do padding masks interact with self-attention and the loss?
7. Why is a geographic split preferable to a random split?
8. How would you include neighbouring buildings and roads?
9. How do you prevent invalid or self-intersecting outputs?
10. Why are vertex-level classification metrics insufficient?
11. What does the New Zealand transfer test reveal?
12. Why can a Transformer fail to beat a circular CNN on this task?
13. What additional problems arise in an autoregressive coordinate decoder?
14. How would professional multiscale targets change the experiment?
15. How would you extend the model from polygons to road-network thinning?

## 23. What to show Prof. Kada in five minutes

Do not begin with code. Present evidence in this order:

1. **Research question — 30 seconds:** explain why vector rings need cyclic, geometric, and contextual representations rather than ordinary ViT patch tokenization.
2. **Data responsibility — 45 seconds:** show the two source adapters, CRS/topology validation, professional-versus-synthetic result separation, and data provenance.
3. **Architecture idea — 60 seconds:** sketch cyclic relative attention, segment pooling, context tokens, and the parallel-versus-autoregressive comparison.
4. **Evidence — 90 seconds:** show one controlled ablation table and three representative predictions, including one failure.
5. **Scientific judgement — 45 seconds:** explain why OSM proxy labels cannot establish professional cartographic quality and what data would be required.
6. **Your fit — 30 seconds:** connect ViT implementation to attention/positional encoding, registration to geometric constraints and failure analysis, and the TU Berlin geoinformatics degree to CRS/GIS/data processing.

The strongest signal is not the largest architecture. It is a compact repository in which the data assumptions, baseline fairness, failed cases, and next experiment are immediately clear.

## 24. Sources to cite in the project

- TU Berlin vacancy VI-255/26: https://www.jobs.tu-berlin.de/en/job-postings/204958
- Cart2Former DFG project record: https://bib-pubdb1.desy.de/record/647622
- Kada, Baerenzung, and Kaufhold (2024), *MLP Feature Extraction from Coordinates for Building Footprint Simplification using Graph Convolutional Networks*: https://doi.org/10.5194/ica-abs-8-11-2024
- Wamhoff, Baerenzung, Kaufhold, and Kada (2025), *CNN-Based Geometric Feature Embedding Using Coordinates for Cartographic Generalization Tasks on Building Footprints*: https://doi.org/10.5194/ica-proc-7-26-2025
- Zhou, Fu, and Weibel (2023), *Move and remove: Multi-task learning for building simplification in vector maps with a graph convolutional neural network*: https://doi.org/10.1016/j.isprsjprs.2023.06.004
- MapGeneralizer repository: https://github.com/chouisgiser/MapGeneralizer
- Geofabrik OpenStreetMap extracts and documentation: https://download.geofabrik.de/

---

## Final scope rule

Before the interview, prioritize a correct heterogeneous data pipeline, controlled baselines, one tested ring-specific Transformer contribution, masked-ring pretraining, and rigorous geometry evaluation. Add a context-denoising smoke test and binary autoregressive edit decoder only after that core is reproducible. Present every unimplemented component as a research plan, never as completed work.
