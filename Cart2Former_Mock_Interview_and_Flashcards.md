# Cart2Former Mock Interview and Flashcards

Use this with the full `Cart2Former_Interview_Study_Guide.md`.

## Practice method

For each question:

1. Answer aloud without notes.
2. Keep normal answers to 60-120 seconds.
3. Score yourself:
   - 0: could not answer;
   - 1: definitions only;
   - 2: technically correct with project relevance;
   - 3: correct, concise, includes tradeoffs and evaluation.
4. Repeat all questions scoring below 2 the next day.

Do not memorize exact paragraphs. Memorize structures, distinctions, and evidence.

---

## Round 1: opening and motivation

### 1. Tell us about yourself.

Answer structure:

- TU Berlin M.Sc. and specialization.
- Geometric research strength.
- Institut Pascal registration example.
- Current work.
- Bridge to Cart2Former.
- Immediate contribution.

Do not mention financial pressure, visa anxiety, or that this is your only option.

### 2. Why did you apply for Cart2Former?

Key points:

- Intersection of geoinformatics, geometry, and deep learning.
- New research rather than routine application of an existing model.
- Interest in vector representations, context, and validity.
- Contribution through PyTorch, data pipelines, and evaluation.

### 3. Why move from medical/3D vision to cartography?

Model answer:

> The application domain changes, but the scientific core remains highly relevant: geometric representation, spatial-semantic integration, coordinate transformations, model implementation, and quantitative evaluation. I am not assuming that medical or 3D methods transfer unchanged. I am specifically studying polygon representation, cartographic operators, topology, and multi-scale constraints to make the transition rigorously.

### 4. What is your biggest gap?

Model answer:

> My biggest gap is direct research experience in cartographic generalization. I am addressing it through the group’s recent papers, classic generalization operators, vector representation, and benchmark design. My strongest transferable evidence is geometric learning, geoinformatics education, Python/PyTorch, and independent research evaluation.

### 5. Why should we hire you?

Answer structure:

- Strong degree-role alignment.
- Geometry plus AI rather than AI alone.
- Reproducible implementation and evaluation.
- Honest, fast domain learning.
- Concrete first-90-days approach.

Avoid claiming you are the best candidate.

### 6. Where do you want to be after three years?

Good answer:

> I want to have developed scientifically defensible vector-geometry models and reusable benchmarks, published the results, and grown into an independent GeoAI researcher. I would like the work to contribute both methodological advances and tools or datasets useful to the cartographic community.

### 7. Are you interested in a doctorate?

If true:

> Yes. The project’s three-year research scope, publication responsibilities, and methodological novelty align strongly with doctoral research. I would like to understand how the chair structures doctoral supervision and project milestones.

Do not imply the doctorate is guaranteed; the vacancy does not explicitly state it.

---

## Round 2: Professor Kada and group research

### 8. What do you know about Professor Kada’s research?

Key points:

- Head of Methods of Geoinformation Science.
- 3D buildings/city models, algorithms, reconstruction, visualization.
- Cartographic 3D generalization and aggregation.
- Recent direct coordinate learning, self-supervision, vector extraction, Transformers.
- Cart2Former as progression toward generative context-aware vector models.

### 9. Which group paper did you find most relevant?

Answer:

> The 2025 paper on CNN-based geometric feature embeddings is the most directly relevant. It learns directly from building-footprint coordinates, uses self-supervised geometric pretraining, and evaluates building classification and simplification. It also identifies self-supervised Transformers and additional vector types as future directions, which appears closely connected to Cart2Former.

### 10. Summarize that 2025 paper.

Include:

- problem with rasterization and handcrafted features;
- circular 1D CNN on ordered polygon vertices;
- self-supervised local geometry regression;
- keep/remove/move plus displacement;
- 8,494 Stuttgart footprints;
- comparisons and metrics;
- limitations and Transformer extension.

### 11. Why can a 1D CNN work on a building polygon?

Answer:

> A simple polygon ring has a fixed local degree of two: each vertex has a predecessor and successor. A circular 1D convolution with kernel size three can aggregate those neighbours while respecting the cyclic boundary. It is simpler than a general graph convolution for this constrained topology. The limitation is that irregular sampling, holes, multipolygons, and object-to-object context require additional design.

### 12. What limitation would you address first?

Strong options:

- cross-city geographic generalization;
- irregular vertex sampling and loop origin;
- broader validity/cartographic metrics;
- multiple object classes and spatial context;
- multipolygons, polylines, and networks;
- limited labelled data.

Explain why and how to test it.

### 13. What is important about the DETR building-model paper?

Answer:

> It demonstrates a hybrid design: a Transformer predicts meaningful geometric primitives, building planes, and deterministic Boolean geometry constructs a watertight model. The broader lesson is that a neural network need not generate every final coordinate directly; it can predict structured primitives while geometric algorithms guarantee validity.

---

## Round 3: cartographic generalization

### 14. Define cartographic generalization.

> Cartographic generalization is the scale- and purpose-dependent transformation of map content to maintain legibility, important characteristics, and spatial-semantic relationships when the available display space or intended use changes.

### 15. Is generalization just simplification?

> No. Simplification is one operator. Generalization also includes selection, aggregation, displacement, collapse, typification, smoothing, exaggeration, and network thinning, together with semantic and contextual decisions.

### 16. Explain Douglas-Peucker.

Expected points:

- recursively retains the point with maximum perpendicular distance from a baseline;
- stops when deviations are below tolerance;
- efficient geometric baseline;
- no semantic/context awareness;
- may damage building characteristics and orthogonality.

### 17. Explain aggregation.

> Aggregation combines nearby objects into a single representation while preserving the broader occupied area, pattern, or semantic meaning. It is not simply a geometric union: minimum gaps, building groups, roads, density, and target scale influence the result.

### 18. Explain displacement.

> Displacement moves map objects to resolve conflicts or maintain legibility while attempting to preserve relative position, alignment, connectivity, and semantic relationships. It is a constrained multi-object problem rather than independent coordinate regression.

### 19. Explain network thinning.

> Network thinning removes less important edges while retaining connectivity, hierarchy, reachability, and characteristic structure. For roads, semantic class and network centrality may matter as much as geometry.

### 20. Why is scale important?

> The same geometry can be appropriate at one map scale and illegible at another. Minimum visible size, separation, symbol width, and retained semantic detail depend on target scale and use.

### 21. What should building simplification preserve?

- characteristic corners;
- dominant orientation and orthogonality where appropriate;
- area and overall shape;
- symmetry;
- topology and validity;
- relations with neighbouring objects;
- minimum length/width constraints;
- important semantic identity.

### 22. Why are traditional algorithms not enough?

Do not dismiss them.

> Classical algorithms are essential baselines and may guarantee constraints. The challenge is formalizing all interacting geometric, semantic, contextual, and perceptual decisions across diverse environments and scales. Learned models can capture recurring patterns, while hybrid methods can retain hard validity guarantees.

---

## Round 4: vector representation

### 23. How would you represent a simple polygon?

> As an ordered cyclic sequence of unique boundary vertices or edges, with an implicit closing edge from the final vertex to the first. I would maintain ring orientation and metadata rather than treating it as an unordered point set.

### 24. How would you represent a polygon with holes?

> Use separate ring sequences with exterior/interior identifiers and orientation conventions, plus object-level pooling or hierarchy that associates all rings with the same polygon. The model and evaluation must preserve hole containment and ring validity.

### 25. How would you represent a multipolygon?

> Encode each component and ring separately, attach part and object identifiers, pool hierarchically, and make the object embedding invariant to arbitrary part ordering while preserving spatial relations among parts.

### 26. Which polygon invariances matter?

- loop-origin invariance;
- trivial-vertex invariance;
- part-permutation invariance;
- topology awareness;
- carefully selected translation/rotation/scale invariance.

Add that orientation and absolute location may matter through context.

### 27. How do vectors differ from point clouds?

> Point clouds are typically treated as unordered samples of a surface. Polygon and polyline vertices have explicit sequence order, edges, rings, topology, and semantic object identity. A map also contains inter-object relationships and target-scale constraints. Point-cloud intuition helps with coordinate learning but cannot simply replace vector structure.

### 28. How do you handle variable-length geometry?

> Padding and masks, length-bucketed batching, adaptive pooling, or hierarchical encoding. Fixed resampling is possible but may alter meaningful corners and must be evaluated.

### 29. What is lost when centring and scaling a polygon?

> Absolute location, real-world size, and potentially orientation or neighbourhood meaning. I would preserve those as separate object/context features when they matter.

### 30. Vertex tokens or edge tokens?

> Both are plausible. Vertex tokens directly support coordinate and operation prediction. Edge tokens naturally encode length, direction, and adjacency. I would treat this as an empirical design decision and include a hybrid or ablation.

---

## Round 5: Transformers

### 31. Explain self-attention.

> Each token produces a query, key, and value. Query-key similarities determine attention weights, and the output is a weighted combination of values. This lets a token combine information from relevant tokens regardless of distance in the sequence.

### 32. Why divide by `sqrt(d_k)`?

> As key dimension grows, dot-product variance grows. Scaling keeps logits in a range where softmax is less likely to saturate, improving optimization.

### 33. Why multi-head attention?

> Different learned subspaces can model different relationships, such as local adjacency, long-range shape, or contextual object relations. It increases representation capacity, although individual heads are not guaranteed to have simple interpretations.

### 34. Why positional encoding?

> Self-attention alone does not encode token order. Geometry additionally requires coordinate, relative, cyclic, and structural information, so sequence position alone is insufficient.

### 35. Absolute or relative positional encoding?

> Absolute encoding helps preserve global location and sequence index. Relative encoding naturally captures displacement, adjacency, and local geometry. I would likely combine normalized local/relative geometry with separate global/context features and test both through ablation.

### 36. Encoder-only, decoder-only, or encoder-decoder?

> Encoder-only suits classification, vertex labelling, or representation learning. Decoder-only suits unconditional or context-prefix autoregressive generation. Encoder-decoder is natural when detailed source geometry and context are encoded and a generalized sequence is generated. I would compare an encoder operation-prediction baseline with an encoder-decoder generative model.

### 37. Why not always use a Transformer?

> Transformers can be data hungry and computationally expensive, and they lack some geometric inductive biases. A 1D CNN or GNN may be simpler and more efficient for local ring structure. Model choice should follow benchmarks, not fashion.

### 38. How do you handle long city-scale sequences?

- local/window attention;
- object hierarchy;
- neighbourhood graph;
- adaptive pooling;
- sparse attention;
- processing spatial tiles with overlap;
- efficient attention variants.

### 39. How does your ViT experience transfer?

> It transfers through token embeddings, attention, positional encoding, training, regularization, and fine-tuning. Cart2Former adds continuous coordinates, variable length, cyclic topology, multiple objects, and constrained generation.

---

## Round 6: generative and autoregressive models

### 40. What does autoregressive generation mean?

> The model factors the output probability into conditional next-step probabilities and generates each coordinate, token, or operation using the input and all previously generated outputs.

### 41. Why is autoregression useful for polygons?

> It naturally handles variable length and lets each new element depend on the partial shape, which may help preserve sequential structure and symmetry. It also supports explicit end-of-ring and end-of-object tokens.

### 42. What are the weaknesses?

- slow sequential inference;
- exposure bias;
- accumulated coordinate error;
- sensitivity to starting order;
- invalid partial or final geometry;
- difficult multimodal continuous prediction.

### 43. Continuous or quantized coordinates?

> Quantized coordinates permit categorical next-token prediction but introduce resolution error. Continuous regression preserves precision but can struggle with multimodal outputs. A hybrid coarse-cell-plus-offset or operation-plus-displacement formulation may offer a useful compromise.

### 44. What is teacher forcing?

> During training, the decoder receives the ground-truth previous output. During inference, it receives its own previous prediction. The mismatch causes exposure bias and error accumulation.

### 45. How does generation stop?

> Predict explicit end-of-ring and end-of-object tokens, with validity-aware minimum/maximum length rules and masks. For operation-based models, the output length may be tied to the input instead.

### 46. Operation prediction or free coordinate generation?

> Operation prediction is easier to constrain and aligns with existing keep/remove/move labels. Free generation is more flexible for aggregation and new topology but harder to train and validate. I would start with operation prediction as a baseline and extend only where the task requires free output topology.

---

## Round 7: self-supervised learning

### 47. Why self-supervision here?

> Paired generalization examples are scarce and expensive, whereas unlabelled vector data is abundant. Self-supervision can learn geometric representations before task-specific fine-tuning.

### 48. Propose three pretext tasks.

- masked vertex/edge reconstruction;
- denoising perturbed geometry;
- prediction of edge length, angle, local area, or next vertex.

Optional: contrastive loop-origin/orientation views or multi-scale reconstruction.

### 49. What is dangerous about contrastive augmentation?

> A transformation assumed to preserve meaning may change orientation, scale, context, topology, or cartographic importance. Positive pairs must match the intended invariances of the downstream task.

### 50. How would you prove pretraining helps?

> Compare random versus pretrained initialization across several label fractions, cities, scales, and downstream operators. Report convergence, final performance, robustness, and multiple seeds with ablations for each pretext task.

### 51. Could self-supervision hurt?

> Yes. A pretext task can encourage the wrong invariances or overemphasize local reconstruction rather than generalization. Negative transfer should be tested through controlled fine-tuning and representation analysis.

---

## Round 8: data and geospatial processing

### 52. How would you combine heterogeneous sources?

Answer sequence:

- provenance/licence/metadata;
- projected CRS;
- schema harmonization;
- geometry validation;
- temporal alignment;
- cross-scale object matching;
- labels;
- geographic splits;
- versioning and audit reports.

### 53. Why use a projected CRS?

> Distance, area, displacement, and tolerances need meaningful metric units. Latitude/longitude degrees are not uniform metric distances. Projection distortion still has to be considered for large regions.

### 54. What is spatial leakage?

> Nearby or nearly identical geometries appear across train and test sets, allowing spatial memorization and inflating performance. Split by blocks, districts, or cities rather than randomly by object.

### 55. How do you separate temporal change from generalization?

> Compare acquisition dates and authoritative identifiers, detect additions/removals or changed footprints, and exclude or separately label changes not caused by scale transformation. Otherwise the model may learn demolition or construction as a generalization operator.

### 56. How would you match objects across scales?

> Use identifiers if available; otherwise combine spatial overlap, proximity, semantics, and topology. Account explicitly for one-to-one, one-to-many, many-to-one, additions, and eliminations. Manual audits are needed for uncertain matches.

### 57. How would you make the pipeline reproducible?

- version data and code;
- save configurations and seeds;
- record CRS/schema transformations;
- deterministic preprocessing where possible;
- audit invalid geometry repairs;
- unit tests;
- checkpoint and environment files;
- experiment tracking.

---

## Round 9: evaluation

### 58. Why is accuracy insufficient?

> Keep/remove classes may be imbalanced, and correct labels do not guarantee valid or cartographically good geometry. Use macro-F1, regression metrics, geometric similarity, validity, conflict, and robustness measures.

### 59. IoU versus Hausdorff distance?

> IoU measures area overlap and may hide local boundary errors. Hausdorff captures the maximum nearest-boundary discrepancy and is sensitive to outliers. They are complementary.

### 60. Hausdorff versus Chamfer distance?

> Hausdorff emphasizes the worst mismatch. Chamfer averages nearest-neighbour distances and is smoother but may hide a severe local error. Sampling strategy also affects boundary-distance estimates.

### 61. Which metrics would you report?

- operation macro-F1;
- displacement MAE;
- IoU and Hausdorff;
- area/orientation/vertex-reduction error;
- validity and self-intersection rate;
- conflict and minimum-separation compliance;
- cross-city and cross-scale robustness;
- runtime/memory.

### 62. What baselines would you use?

- Douglas-Peucker or appropriate rule-based system;
- move/remove GCN;
- coordinate MLP/GCN;
- group’s 1D CNN;
- simple MLP;
- geometry-only Transformer;
- context-aware Transformer;
- with and without pretraining.

### 63. How would you evaluate ambiguity?

> Some source geometries may have several acceptable generalized forms. Consider multiple references if available, constraint satisfaction, expert review, distributional metrics, and calibrated uncertainty rather than assuming one target sequence is the only correct answer.

---

## Round 10: geometric constraints

### 64. How do you prevent self-intersection?

> Use validity-aware decoding or operation masks where possible, add geometric penalties, and validate with a geometry engine. For guaranteed validity, use constrained repair or a representation whose operations preserve a valid ring.

### 65. How do you prevent buildings overlapping roads?

> Encode road context and minimum-separation requirements, penalize or mask conflict-producing moves, and run deterministic spatial validation. Displacement may need joint multi-object optimization rather than independent buildings.

### 66. Hard or soft constraints?

> Hard constraints guarantee critical validity but may restrict optimization or be non-differentiable. Soft losses are trainable but cannot guarantee compliance. Use hard constraints for essential topology/validity and soft objectives for qualities such as shape similarity, followed by explicit validation.

### 67. What if post-processing changes the model output heavily?

> That indicates the model is not learning the desired feasible distribution. Report repair magnitude and failure rates, analyze the cases, and move more constraint information into representation, loss, or decoding rather than hiding errors with post-processing.

---

## Round 11: research method and behaviour

### 68. Describe your independent research.

Use STAR plus scientific structure:

- problem and constraints;
- your specific responsibility;
- method choices and alternatives;
- experiment and evaluation;
- quantitative result;
- failure/limitation;
- lesson relevant to Cart2Former.

### 69. Tell us about a failed experiment.

Good answer pattern:

> I expected X because Y. The result contradicted it. I checked data, implementation, optimization, and metric assumptions in that order. I found Z. I changed A, reran a controlled comparison, and learned B.

Never say you have never had a failed experiment.

### 70. How do you debug a model that does not learn?

1. Verify labels and transformations visually.
2. Overfit a tiny batch.
3. Check loss, gradients, masks, normalization, and output scale.
4. Start with a simple baseline.
5. Inspect class balance and metric implementation.
6. Change one factor at a time.

### 71. How do you ensure reproducibility?

> Versioned data and code, saved configs and seeds, environment specification, deterministic preprocessing, tracked experiments, multiple runs, and documented exceptions. Reproducibility also means describing data cleaning and failed runs, not only publishing final weights.

### 72. How do you handle disagreement?

> Clarify the scientific question, identify evidence that would distinguish the alternatives, agree on a small experiment or decision criterion, document the result, and avoid making a technical disagreement personal.

### 73. How do you receive critical feedback?

> Separate the criticism from personal judgement, restate it to confirm understanding, check whether evidence supports it, and revise or explain the tradeoff. In research, critical feedback often exposes hidden assumptions.

### 74. How do you prioritize an ambiguous project?

> Clarify the first decision the project must make, establish data and baseline readiness, choose a minimum experiment that reduces the largest uncertainty, and document assumptions. Avoid beginning with the most complex architecture.

---

## Round 12: your previous work

### 75. Explain your registration pipeline.

Cover:

- preoperative/intraoperative problem;
- semantic initialization;
- scale/rigid alignment;
- RANSAC and ICP;
- coordinate spaces;
- evaluation design;
- limitations.

### 76. Why semantic initialization?

> Pure local registration can converge to the wrong local optimum when initialization is poor. Semantic landmarks or regions provide a meaningful coarse alignment before geometric refinement.

### 77. Why RANSAC before ICP?

> RANSAC provides robust coarse estimation under outliers, while ICP is local and sensitive to initialization. The sequence improves the chance that ICP begins in the correct basin.

### 78. Which registration metrics did you use?

State only real metrics. Clarify that 19,157 refers to evaluated surface vertices.

### 79. Explain your ViT work.

Prepare exact facts:

- architecture;
- input/output;
- pretrained or scratch;
- dataset size and split;
- loss;
- augmentations;
- result;
- comparison;
- limitation.

Do not use the resume’s classification results for ViT unless they truly came from that ViT experiment.

### 80. What did ViT teach you that is relevant here?

> Practical token embeddings, multi-head attention, positional information, transfer learning, optimization, regularization, and evaluation. The Cart2Former extension requires new treatment of vector order, topology, continuous coordinates, and valid generation.

---

## A 45-minute mock interview script

### Minutes 0-5: introduction

1. Tell us about yourself.
2. Why Cart2Former?
3. What is your main gap?

### Minutes 5-15: previous work

4. Explain your thesis problem and method.
5. Why semantic initialization, RANSAC, and ICP?
6. Describe a failure and how you diagnosed it.
7. Explain your ViT experience precisely.

### Minutes 15-30: Cart2Former technical

8. Define cartographic generalization.
9. How would you encode a polygon?
10. Which invariances matter?
11. Design an autoregressive decoder.
12. Design self-supervised pretraining.
13. Enforce spatial constraints.
14. Compare CNN, GNN, and Transformer baselines.

### Minutes 30-38: dataset and science

15. Build a heterogeneous data pipeline.
16. Prevent spatial leakage.
17. Select metrics and ablations.
18. What would you do in the first 90 days?

### Minutes 38-42: behaviour

19. How do you prioritize ambiguous research?
20. How do you handle criticism and collaboration?

### Minutes 42-45: your questions

Ask three prepared questions about first operator/dataset, expected six-month milestone, and project/doctoral structure.

---

## Final rapid-fire checklist

You are ready when you can answer yes to all of these:

- I can explain Cart2Former in one sentence and two minutes.
- I can name and explain the main generalization operators.
- I can summarize the 2024 and 2025 Kada-group papers.
- I can explain attention, masking, and positional encoding.
- I can distinguish ViT patches from polygon tokens.
- I can represent polygons, holes, multipolygons, polylines, and networks conceptually.
- I can compare operation prediction with coordinate generation.
- I can propose three self-supervised tasks.
- I can design a heterogeneous geospatial data pipeline.
- I can explain spatial leakage.
- I can name classical and neural baselines.
- I can justify at least six complementary metrics.
- I can explain hard, soft, and post-processing constraints.
- I can present my thesis in five minutes.
- Every result and number I mention is accurate.
- I have three intelligent questions for the panel.

