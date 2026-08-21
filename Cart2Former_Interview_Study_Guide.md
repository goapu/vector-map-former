# Cart2Former Research Assistant Interview Study Guide

Prepared for: Dilip Goswami  
Position: Research Assistant, TU Berlin, reference VI-255/26  
Project: Cart2Former  
Verified: 21 August 2026

## How to use this guide

Do not try to memorize every line. Your objective is to be able to:

1. Explain the research problem in plain language.
2. Discuss technically credible ways to represent and generate vector geometry.
3. Design a clean dataset, experiment, benchmark, and failure analysis.
4. Connect your verified experience in ViTs, geometric computer vision, registration, PyTorch, and geoinformatics to the project.
5. Admit what you have not yet done without sounding unprepared.

The highest-priority sections are marked **MUST KNOW**.

---

## 1. The position in one page - MUST KNOW

The official role is a 36-month, TV-L E13 research position in the Chair of Methods of Geoinformation Science at TU Berlin. The position has no stated teaching obligation. The work is part of the DFG-funded Cart2Former project.

### Official responsibilities

- Conduct independent research within Cart2Former.
- Design and implement novel Transformer architectures for geographic vector data.
- Create training and benchmark datasets from heterogeneous geospatial sources.
- Train, evaluate, and benchmark AI models.
- Publish and present results internationally.
- Prepare project and final reports.

### Official candidate profile

- Relevant master's degree.
- AI and deep-learning experience.
- PyTorch or TensorFlow proficiency.
- Excellent Python.
- Solid geospatial-data processing and analysis.
- Basic cartography and cartographic generalization are advantageous.
- Ability to work scientifically and independently.
- Good German and/or English, with willingness to learn the missing language.

### What the interview must establish

The panel already knows you do not have a long publication history in cartographic generalization. You were invited anyway. They are likely deciding whether you have:

- enough geometric and AI depth to enter the project quickly;
- the ability to reason about vector geometry rather than only raster images;
- scientific discipline in data splits, baselines, metrics, ablations, and reporting;
- intellectual honesty and the ability to learn cartography;
- motivation for a three-year research project, not just an urgent need for employment;
- clear spoken English and workable collaboration habits.

Official vacancy: <https://www.jobs.tu-berlin.de/en/job-postings/204958?language=en>

---

## 2. Professor Martin Kada: public professional profile

This section covers role-relevant public professional information, not private information.

### Current role

Professor Dr.-Ing. Martin Kada is head of the Chair of Methods of Geoinformation Science at TU Berlin's Institute of Geodesy and Geoinformation Science. The chair works on modelling, acquisition, processing, analysis, and presentation of spatial data.

Official group profile: <https://www.tu.berlin/en/gis>  
Official management page: <https://www.tu.berlin/en/gis/about/management-and-administration>

### Publicly listed expertise

His ResearchGate profile lists expertise in:

- 3D;
- buildings;
- algorithms;
- segmentation;
- 3D computer graphics;
- 3D modelling and reconstruction;
- visualization and real-time rendering;
- C++.

ResearchGate profile: <https://www.researchgate.net/profile/Martin-Kada>

### Broader chair research

The chair's official profile emphasizes:

- spatial modelling and algorithms;
- data acquisition and integration;
- 3D geodatabases and visualization;
- geodata infrastructures;
- the transition from 2D to 3D geodata and time as a fourth dimension;
- automated acquisition of 3D buildings and city models;
- integration of uncertain, fuzzy, and redundant geometry into GIS.

This matters because the Cart2Former interview may extend beyond generic deep learning. Expect interest in geometrically valid, GIS-usable, explainable, and scalable results.

### Teaching context

TU Berlin course listings associate Professor Kada with Geo Data Science, Deep Learning for Geographical Data, and Geographical Information Systems. This suggests he may expect you to explain AI concepts rigorously and connect them to geospatial practice.

Course source: <https://www.static.tu.berlin/fileadmin/www/10004219/INT_SB/Dokumente/Incomings/English_Courses_MTS.pdf>

### Relevant team members

The current group page includes Julien Baerenzung, Lilli Kaufhold, Matthias Wamhoff, Andreas Fuls, and others. Baerenzung, Kaufhold, and Wamhoff are especially relevant because they co-authored the recent building-generalization papers. They may participate in the project or interview, but that is not confirmed.

Team page: <https://www.tu.berlin/gis/ueber-uns/team>

### Research trajectory relevant to your interview

Professor Kada's work shows a coherent trajectory:

1. Longstanding 3D city-model reconstruction and cartographic 3D generalization.
2. Simplification and aggregation under geometric and perceptual constraints.
3. Deep learning for building reconstruction, segmentation, and vector extraction.
4. Direct feature learning from building-footprint coordinates with GNN/CNN models.
5. Self-supervised geometric embeddings for building simplification.
6. Generative, context-aware Transformer models for polygons, polylines, and networks in Cart2Former.

This trajectory is the best predictor of the scientific discussion in the interview.

---

## 3. Cart2Former: exact project concept - MUST KNOW

The DFG project identifier is 577232574, coordinated by Martin Kada, with a grant period starting in 2026.

Project record: <https://bib-pubdb1.desy.de/record/647622>

### Problem statement

Cartographic generalization changes map content so it remains readable, meaningful, and spatially coherent at a smaller scale or for a different use. Individual operators can be formalized, but the complete decision process is difficult because geometric, semantic, contextual, and perceptual constraints interact.

### Planned technical components

- Generative Transformer models for vector map data.
- Feature extraction directly from geometry coordinates.
- Positional, structural, and contextual encodings.
- Adaptive pooling.
- Hierarchical feature extraction.
- Shape simplification.
- Object aggregation.
- Displacement to resolve spatial conflicts.
- Polygon and polyline geometries.
- Network thinning.
- Autoregressive sequential geometry generation.
- Preservation of shape characteristics and symmetry.
- Explicit spatial and semantic context.
- Self-supervised pretraining on large unlabelled vector datasets.
- Transfer to downstream tasks without requiring large annotated datasets.

### A one-sentence explanation for the interview

> Cart2Former aims to learn cartographic generalization directly on vector geometries by developing hierarchical, context-aware Transformer representations and generative decoders that preserve geometric and semantic properties across simplification, aggregation, displacement, and network-thinning tasks.

### A plain-language explanation

> When a detailed map is shown at a smaller scale, objects cannot simply be shrunk. Some details must be removed, some objects combined, and some moved to avoid conflicts, while important shapes and relationships remain recognizable. Cart2Former investigates whether a Transformer can learn these decisions directly from vector coordinates and their spatial-semantic context.

---

## 4. Cartographic generalization - MUST KNOW

### Why generalization is necessary

A map at 1:5,000 can display more geometric detail than a map at 1:50,000. At a smaller display scale:

- narrow gaps disappear;
- nearby symbols overlap;
- tiny objects become illegible;
- excessive vertices add noise;
- maintaining every object can hide the important spatial structure.

Generalization is not merely compression. It is purpose- and scale-dependent abstraction.

### Core operators

#### Selection and elimination

Retain important objects and remove less important ones based on scale, function, hierarchy, or density.

#### Simplification

Reduce geometric detail while preserving characteristic shape. Douglas-Peucker is a classic polyline simplification method, but building footprints often require stronger constraints such as orthogonality, characteristic corners, minimum edge length, and area preservation.

#### Smoothing

Remove small fluctuations while preserving the overall trend or character of a line.

#### Aggregation or amalgamation

Merge several nearby objects into a single representation, such as a group of buildings becoming one built-up block.

#### Collapse

Replace an area object by a line or point when its width or area becomes too small at the target scale.

#### Displacement

Move objects to resolve symbol overlap or spatial conflict while retaining important relative relationships.

#### Typification

Reduce the number of repeated objects but maintain their density, pattern, and distribution.

#### Exaggeration

Enlarge an important feature that would otherwise disappear.

#### Network thinning

Remove less important edges from a road, river, or other network while preserving connectivity and hierarchy.

### Important constraints

- Topological validity: connectivity, containment, adjacency, no unintended crossing.
- Geometric validity: closed rings, no self-intersection, valid holes.
- Shape preservation: orientation, symmetry, characteristic corners, orthogonality.
- Size constraints: minimum area, length, width, and separation.
- Spatial relationships: distance, alignment, proximity, relative position.
- Semantic importance: hospital versus shed, motorway versus local road.
- Legibility: no overlapping symbols or imperceptible gaps.
- Target scale and application.

### Strong answer: Why is deep learning useful?

> Traditional algorithms remain valuable and should be used as baselines and constraint mechanisms. The difficulty is not that no operator can be formalized; it is that many operators and constraints interact across diverse geometries, contexts, and target scales. Learning-based methods can infer recurring patterns from paired examples and large datasets. However, valid cartographic output still requires explicit constraints, careful evaluation, and often a hybrid learned-plus-rule-based system.

### Strong answer: Why direct vectors instead of rasterization?

> Vector data preserves exact coordinates, topology, object identity, and semantic attributes. Rasterization introduces resolution-dependent aliasing and can produce boundary artifacts. Direct vector learning avoids an image-to-vector conversion step and makes outputs easier to integrate into GIS. The tradeoff is that vector geometries are irregular, variable in length, cyclic, and topologically structured, so the architecture must handle those properties explicitly.

---

## 5. Vector geometry representation - MUST KNOW

### Polygon representation

A simple polygon can be represented by an ordered sequence of vertices:

`[(x1, y1), (x2, y2), ..., (xn, yn)]`

The final edge connects the last vertex back to the first. Complex geometries may contain:

- an exterior ring;
- one or more interior rings or holes;
- multiple disconnected polygons;
- semantic attributes;
- neighbouring objects and spatial relations.

### Polygon properties a model should consider

- Loop-origin invariance: changing the first vertex should not change the geometry.
- Trivial-vertex invariance: adding a redundant collinear vertex should not change meaning.
- Part-permutation invariance: reordering parts of a multipolygon should not change the object.
- Topology awareness: holes, connectivity, and component relationships must be represented.
- Orientation handling: clockwise versus counterclockwise order.
- Translation, rotation, and scale treatment: decide which should be invariant and which should remain meaningful.

Do not casually claim full rotation invariance is always desirable. Building orientation relative to roads or neighbouring structures may be meaningful. The desired invariance depends on the task and context.

### Token choices

A token could represent:

- a vertex;
- an edge;
- a local geometric primitive;
- an entire object;
- a generalization operation such as keep, remove, or move.

Potential vertex/edge features:

- absolute or normalized coordinates;
- relative coordinates to centroid or bounding box;
- previous and next edge vectors;
- edge lengths;
- turning angles;
- local triangle area;
- curvature estimate;
- ring identifier;
- geometry type;
- semantic class;
- target map scale;
- local density or neighbourhood features.

### Normalization

A useful design is to separate:

- local geometry features, normalized around the object's centroid and scale;
- global/context features, retaining projected coordinates, orientation, neighbourhood, and map scale.

This helps the model learn shape without discarding spatial context.

### Variable length

Options include:

- padding plus attention masks;
- batching by similar sequence lengths;
- resampling to a fixed number of vertices;
- adaptive pooling;
- hierarchical encoding;
- sparse/local attention for large scenes.

Resampling makes batching easier but can change meaningful corners or introduce redundant points. It must be evaluated, not assumed harmless.

### Strong answer: How would you encode a polygon?

> I would begin with an ordered vertex or edge sequence in a projected coordinate system. I would embed normalized local coordinates together with edge vectors, lengths, turning angles, ring identity, semantic class, and target scale. I would use padding masks for variable length and explicitly test loop-origin and orientation handling. A hierarchical model could first encode each geometry and then use contextual attention across neighbouring objects. I would compare continuous coordinate embeddings with quantized tokens and include ablations for each structural feature.

---

## 6. Transformer foundations - MUST KNOW

### Scaled dot-product attention

Given queries Q, keys K, and values V:

`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`

Interpretation:

- Query: what information does this token seek?
- Key: what information does another token advertise?
- Value: what content is retrieved if that token is relevant?

The scaling by `sqrt(d_k)` helps prevent very large dot products and saturated softmax gradients.

### Multi-head attention

Multiple heads learn different relationships. In geometry, possible relationships include:

- adjacent vertices;
- long-range symmetry;
- parallel or perpendicular edges;
- context from neighbouring buildings or roads;
- semantic compatibility.

Do not claim that individual heads will necessarily be interpretable in exactly this way; it is an intuition, not a guarantee.

### Encoder versus decoder

- Encoder: bidirectional attention creates contextual representations of the entire input.
- Autoregressive decoder: causal masking prevents access to future output tokens and predicts the next token conditioned on previous outputs.
- Encoder-decoder: input geometry/context is encoded, and a decoder generates a generalized output sequence.

### Positional encoding

Self-attention is otherwise insensitive to token order. Position can be represented through:

- learned absolute embeddings;
- sinusoidal encodings;
- relative positional bias;
- rotary encodings;
- graph or structural encodings;
- continuous coordinate embeddings.

For Cart2Former, sequence position alone is insufficient. The model needs geometric and structural relations.

### Feed-forward network and residual structure

Each Transformer block typically includes attention, residual connections, normalization, and a token-wise multilayer perceptron. Be ready to explain layer normalization, dropout, and residual connections at a conceptual level.

### Complexity

Full self-attention is quadratic in sequence length. For a city scene with many vertices and objects, consider:

- local or windowed attention;
- hierarchical attention;
- object-level pooling;
- sparse attention;
- neighbourhood graphs;
- efficient attention variants.

### Connection to your ViT experience

ViT converts fixed image patches into tokens. Cart2Former may convert irregular vertices, edges, and objects into tokens. Transferable knowledge includes:

- token embeddings;
- attention and positional encoding;
- pretrained backbones and fine-tuning;
- regularization and training stability;
- error analysis.

New challenges include:

- continuous coordinates;
- variable length;
- cyclic rings;
- topology;
- multiple objects and relations;
- valid generative output.

---

## 7. Generative and autoregressive geometry - MUST KNOW

### Autoregressive factorization

For an output sequence `y1 ... yT`:

`p(y1, ..., yT | x) = product_t p(yt | y<t, x)`

The model generates one element at a time, conditioned on the input and all previous outputs.

### Possible output formulations

#### Operation prediction

For each input vertex, predict:

- keep;
- remove;
- move;
- displacement vector or movement distance.

This is easier to constrain and aligns with recent building-simplification work, but output topology is tied to the input vertices.

#### Coordinate sequence generation

Generate output vertices one at a time, followed by an end-of-ring or end-of-object token. This is more flexible but makes validity and exposure bias harder.

#### Quantized coordinates

Map x and y values to discrete bins and treat them like vocabulary tokens. Advantages include categorical likelihood and straightforward autoregression. Disadvantages include quantization error and scale sensitivity.

#### Continuous coordinates

Regress x-y values or predict a probabilistic continuous distribution. Advantages include precision; disadvantages include multimodality and potentially unstable sequential regression.

#### Hybrid

Predict a coarse discrete spatial cell followed by a continuous offset, or generate operation tokens plus continuous displacement.

### Teacher forcing and exposure bias

During training, the decoder often receives the true previous output. During inference, it receives its own predictions. Errors can accumulate. Possible mitigation includes scheduled sampling, beam search, constrained decoding, or non-autoregressive refinement.

### Validity during generation

Possible methods:

- causal masks and operation masks;
- grammar-like constraints;
- self-intersection or topology-aware losses;
- rejection or repair with computational geometry;
- predict-and-refine architectures;
- differentiable geometric constraints when available.

### Strong answer: Why autoregressive?

> Autoregression naturally handles variable-length output and allows each generated coordinate or operation to depend on the existing partial geometry. That may help preserve sequential structure, shape characteristics, and symmetry. The disadvantages are slower inference, exposure bias, and accumulated geometric error, so I would benchmark it against operation-classification and parallel prediction baselines.

Recommended reading: PolyGen, <https://proceedings.mlr.press/v119/nash20a.html>

---

## 8. Positional, structural, and contextual encoding - MUST KNOW

### Positional encoding

Represents order or location. Possible inputs:

- vertex index in ring;
- normalized x-y coordinates;
- projected/global coordinates;
- relative coordinate to object centroid;
- relative coordinate to previous vertex.

### Structural encoding

Represents geometry and topology:

- ring membership;
- hole/exterior indicator;
- edge adjacency;
- object membership;
- graph distance;
- geometry type;
- road-network degree;
- building hierarchy or cluster membership.

### Contextual encoding

Represents surrounding spatial-semantic conditions:

- nearby objects;
- distance and direction to neighbours;
- road/building relationships;
- land-use class;
- building function;
- map scale;
- density;
- spatial conflicts;
- administrative or planning context.

### Hierarchical architecture example

1. Vertex-level encoder learns local geometry.
2. Pooling creates an object embedding.
3. Object-level Transformer models neighbouring objects.
4. Decoder combines object embedding and context to generate generalized geometry.

### Adaptive pooling

Instead of fixed average or max pooling, learn which vertices or features are most important. This could help preserve characteristic corners or reduce long geometries to compact object tokens.

### Interview design principle

Always state what information could be lost. For example, centring and scaling improve invariance but remove absolute location and size. Preserve the lost information in a separate context channel if it remains relevant.

---

## 9. Self-supervised learning for vector maps - MUST KNOW

### Why self-supervision?

High-quality paired source and generalized maps are scarce, expensive, and may reflect different cartographic specifications. Unlabelled vector maps are much more abundant.

### Candidate pretext tasks

#### Masked vertex or edge reconstruction

Hide vertices or edge attributes and predict them from visible geometry and context.

#### Geometry denoising

Perturb coordinates, add redundant vertices, delete vertices, or distort angles; reconstruct the original geometry.

#### Next-vertex prediction

Predict the next coordinate or operation from a partial sequence.

#### Geometric-property prediction

Predict edge length, turning angle, local triangle area, orientation, curvature, or relation to neighbours.

The 2025 Kada-group paper uses a related idea: it learns geometric embeddings by approximating properties of local triangles before fine-tuning.

#### Contrastive learning

Bring embeddings of equivalent views together and unrelated shapes apart. Possible positive transformations include a changed loop origin, orientation reversal, small non-semantic perturbations, or safe coordinate transformations.

Do not use augmentations that silently change cartographic meaning.

#### Multi-scale prediction

Predict a coarser representation from detailed geometry or recover detailed structure from a coarser/noisy version.

### How to test whether pretraining helps

- Fine-tune with 1%, 10%, 50%, and 100% of labels.
- Compare random initialization with pretrained initialization.
- Test transfer across cities, object types, operators, and scales.
- Run ablations on each pretext task.
- Evaluate geometry, topology, semantic constraints, and convergence speed.

Recommended reading: Masked Autoencoders, <https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html>

---

## 10. Geospatial dataset construction - MUST KNOW

### Heterogeneous-source challenges

- Different coordinate reference systems.
- Different map scales and generalization specifications.
- Schema and semantic-class differences.
- Positional offsets and temporal mismatch.
- Duplicate or missing objects.
- Invalid geometries.
- Inconsistent ring orientation and vertex density.
- Different levels of completeness and accuracy.
- Licensing and provenance.

### Suggested pipeline

1. Record source, licence, scale, date, CRS, accuracy, and schema.
2. Transform to an appropriate projected CRS for metric geometry.
3. Harmonize schema and semantic classes.
4. Validate and repair geometries with an audit trail.
5. Normalize ring orientation and representation conventions.
6. Match source and target objects across scales.
7. Identify additions, deletions, splits, merges, and temporal changes.
8. Derive labels or output sequences.
9. Create geographic train/validation/test splits.
10. Store reproducible metadata, versions, and quality reports.

### Spatial leakage

Random object-level splitting can place neighbouring or nearly identical buildings in train and test sets. This inflates performance. Prefer splits by spatial blocks, districts, cities, or regions.

### Dataset questions to ask

- Are there authoritative paired multi-scale datasets?
- Are generalization operations explicitly labelled or inferred?
- Which target scales and map specifications apply?
- How are temporal changes separated from cartographic changes?
- Which object classes and semantic attributes are available?
- Can the data be published as a benchmark?

### Useful tools to mention

Only mention tools you can use:

- GeoPandas;
- Shapely/GEOS;
- GDAL/OGR;
- pyproj;
- PostGIS;
- QGIS for visual inspection;
- PyTorch and PyTorch Geometric if graph models are used;
- experiment tracking and reproducible configuration tools.

---

## 11. Evaluation and benchmarking - MUST KNOW

No single metric is enough.

### Task metrics

- Accuracy, macro-F1, precision, and recall for keep/remove/move classification.
- MAE or RMSE for displacement regression.
- Sequence likelihood or coordinate error for generation.

### Geometric metrics

- Hausdorff distance: worst-case boundary discrepancy; sensitive to outliers.
- Chamfer distance: average nearest-neighbour discrepancy; can hide local extremes.
- Intersection over Union: area overlap; insensitive to some boundary details.
- Area and perimeter error.
- Vertex-reduction ratio.
- Turning-angle and orientation difference.
- Symmetry, rectangularity, compactness, or shape-descriptor differences where relevant.

### Topological and validity metrics

- Percentage of valid output geometries.
- Self-intersection rate.
- Correct ring and hole structure.
- Preserved adjacency, containment, and connectivity.
- Number of new or unresolved spatial conflicts.
- Network connectivity and reachability.

### Cartographic metrics

- Minimum-separation compliance.
- Symbol conflict count.
- Preservation of characteristic shape.
- Pattern/density preservation after aggregation or typification.
- Semantic hierarchy preservation.
- Expert cartographer assessment where automated metrics are insufficient.

### Generalization and robustness

- Cross-city evaluation.
- Cross-scale evaluation.
- Performance by geometry complexity and semantic class.
- Performance on rare shapes.
- Sensitivity to different vertex sampling and loop origins.
- Calibration or uncertainty for ambiguous outputs.

### Efficiency

- Training and inference time.
- Memory.
- Parameters and FLOPs.
- Throughput by number of vertices/objects.

### Benchmark design

Compare with:

- classical algorithms such as Douglas-Peucker and domain-specific rule-based methods;
- the 2023 move/remove GCN baseline;
- the group’s coordinate-based MLP/GCN and CNN approaches;
- simple MLP, 1D CNN, GNN, and Transformer baselines;
- pretrained versus non-pretrained models;
- geometry-only versus geometry-plus-context models.

---

## 12. Enforcing geometric and cartographic constraints

### Four levels of control

1. **Input representation:** include topology, semantics, target scale, and neighbourhood.
2. **Architecture/decoding:** masks, hierarchical structure, constrained operation vocabulary.
3. **Losses:** penalties for self-intersection, overlap, displacement, shape distortion, or broken connectivity.
4. **Post-processing:** validate and repair using computational geometry or optimization.

### Strong answer

> I would not rely on the network to learn every constraint implicitly. I would combine context-aware inputs, constraint-aware decoding, differentiable penalties where possible, and deterministic GIS validation or repair. I would report both model accuracy and the percentage of outputs satisfying topology and cartographic constraints. A hybrid approach is scientifically defensible because learned models capture complex patterns while established geometry methods guarantee hard validity conditions.

### Tradeoff

Hard constraints guarantee validity but can restrict flexibility or be non-differentiable. Soft constraints allow optimization but do not guarantee valid output. Explain which constraints must be hard and which can be learned or penalized.

---

## 13. Professor Kada's most relevant recent papers

### Paper 1 - Highest priority

**CNN-Based Geometric Feature Embedding Using Coordinates for Cartographic Generalization Tasks on Building Footprints**  
Wamhoff, Baerenzung, Kaufhold, and Kada, 2025  
<https://ica-proc.copernicus.org/articles/7/26/2025/>

#### Research question

Can a relatively simple CNN learn useful geometric features directly from ordered building-footprint coordinates, avoiding rasterization and handcrafted geometric features?

#### Approach

- Circular 1D convolution for polygon sequences with fixed degree two.
- Self-supervised local geometric-property regression to learn embeddings.
- Deeper U-Net-like encoder-decoder for classification and simplification.
- Building simplification as keep/remove/move classification plus displacement regression.
- Comparison with GCN, GraphSAGE, SplineCNN, and prior work.

#### Dataset

- 8,494 Stuttgart building footprints with paired simplifications.
- 60/20/20 train/test/validation split as reported.

#### Metrics

- Accuracy and macro-F1 for vertex classification.
- MAE for movement regression.
- IoU and Hausdorff distance for reconstructed footprints.

#### Key result

Self-supervised pretraining improved the CNN result over the same model without pretraining on several reported metrics. The paper concludes that direct coordinate learning can be effective and identifies Transformer-based self-supervision as a future direction.

#### Critique points to prepare

- Geographic splitting and cross-city generalization are not the main focus.
- The dataset is relatively small.
- CNN performance may depend on vertex sampling and sequence shifts.
- Building rings are structurally simpler than arbitrary multipolygons, polylines, and networks.
- Macro-F1 and accuracy can rank models differently; explain why.
- Validity and cartographic constraints need broader evaluation.

### Paper 2 - Direct predecessor

**MLP Feature Extraction from Coordinates for Building Footprint Simplification using Graph Convolutional Networks**  
Kada, Baerenzung, and Kaufhold, 2024  
<https://ica-abs.copernicus.org/articles/8/11/2024/>

#### Core idea

Instead of feeding manually computed angles, triangle areas, or line lengths to a GCN, add an MLP that learns features from point coordinates. The task predicts keep/remove/move classes and movement along incident edges. The learned feature approach performs on par with precomputed features.

#### Why it matters

This paper marks the move from handcrafted geometry toward end-to-end coordinate feature learning. The 2025 CNN paper deepens this direction, and Cart2Former extends it toward Transformers, context, multiple geometry types, and self-supervision.

### Paper 3 - Transformer and your 3D bridge

**Generating Watertight 3D Building Models from Airborne LiDAR Point Clouds using Detection Transformer (DETR)**  
Kaufhold and Kada, 2025  
<https://isprs-archives.copernicus.org/articles/XLVIII-G-2025/773/2025/>

#### Core idea

A modified DETR predicts building planes directly from airborne LiDAR point clouds. Boolean operations over half-spaces then produce watertight 3D models.

#### Why it matters to you

This connects your geometric computer-vision and 3D-registration background to Kada’s use of Transformers. Discuss the value of predicting meaningful geometric primitives and combining neural prediction with deterministic geometry construction.

### Paper 4 - Generative spatial context

**Generating Realistic Urban Patterns: A Controllable cGAN Approach with Hybrid Loss Optimization**  
Agoub and Kada, 2025  
<https://doi.org/10.3390/ijgi14100375>

#### Core idea

The work generates urban patterns conditioned on spatial and statistical inputs, using Berlin geospatial data. It discusses context, controllability, hybrid losses, spatial constraints, evaluation, and limitations of generated urban layouts.

#### Interview relevance

- Generative models need more than visual realism.
- Context and conditioning strongly influence output.
- Spatial, legal, historical, and typological constraints may need explicit modelling.
- Quantitative metrics, ablations, and practitioner evaluation matter.

### Paper 5 - Vector fidelity and GIS usability

**Benchmarking Vectorized Building Footprint Extraction from Very High Resolution Aerial Imagery**  
Büyükdemircioğlu et al., including Kada, 2025  
<https://doi.org/10.5194/isprs-archives-XLVIII-1-W6-2025-47-2025>

#### Core idea

Line-segment detection networks are compared with semantic segmentation. Pixel scores can look strong while boundaries remain broken or unsuitable for GIS. Vector-oriented methods may provide cleaner topology and more usable footprints.

#### Interview relevance

Always evaluate whether output is usable as geospatial vector data, not only whether a neural metric is high.

### Paper 6 - Current 2026 vector extraction

**From Pixels to Polylines: Extracting City-scale Vectorized Roof Structures with Line Segment Detection Networks**  
Büyükdemircioğlu, Remondino, Kada, and Kocaman, 2026  
<https://doi.org/10.5194/isprs-annals-XI-2-2026-439-2026>

#### Interview relevance

The group remains interested in scalable extraction of explicit vector geometry, small or occluded structures, and city-scale evaluation.

---

## 14. Essential external reading

Read in this order:

1. Cart2Former project description: <https://bib-pubdb1.desy.de/record/647622>
2. Kada group 2025 coordinate-CNN paper: <https://ica-proc.copernicus.org/articles/7/26/2025/>
3. Kada group 2024 MLP/GCN paper: <https://ica-abs.copernicus.org/articles/8/11/2024/>
4. Move and Remove GCN paper: <https://doi.org/10.1016/j.isprsjprs.2023.06.004>
5. Polygon representation learning: <https://arxiv.org/abs/2209.15458>
6. Attention Is All You Need: <https://arxiv.org/abs/1706.03762>
7. Vision Transformer: <https://research.google/pubs/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale/>
8. Masked Autoencoders: <https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html>
9. PolyGen: <https://proceedings.mlr.press/v119/nash20a.html>
10. PolyFormer sequential polygon generation: <https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_PolyFormer_Referring_Image_Segmentation_As_Sequential_Polygon_Generation_CVPR_2023_paper.pdf>

For each paper, prepare five notes:

- problem;
- representation;
- model;
- metrics/results;
- limitation and how Cart2Former could extend it.

---

## 15. Research design and independent scientific work

The role explicitly requires independent research. Demonstrate a repeatable process.

### A strong research workflow

1. Define the scientific question and falsifiable hypothesis.
2. Audit dataset quality and possible leakage.
3. Select simple classical and neural baselines.
4. Define metrics before experiments.
5. Implement the simplest viable model.
6. Verify on synthetic or small data.
7. Track configurations, random seeds, code, and data versions.
8. Run ablations and multiple seeds.
9. Analyze failures by meaningful subsets.
10. Report negative results and limitations.

### Example hypothesis

> A hierarchical Transformer using geometric and contextual encodings will improve cross-city cartographic validity over a geometry-only vertex Transformer, particularly for objects involved in spatial conflicts.

### Example ablations

- no pretraining versus self-supervised pretraining;
- absolute versus relative coordinates;
- no structure encoding versus ring/edge encoding;
- geometry-only versus semantic context;
- object-only versus neighbourhood attention;
- autoregressive versus operation-based decoder;
- learned versus fixed pooling;
- random versus geographic split.

### What counts as a useful negative result?

If context does not improve performance, determine whether:

- context labels are noisy;
- neighbourhood definition is wrong;
- architecture ignores context;
- metric does not measure contextual quality;
- train/test regions have different semantics;
- geometry alone already solves the selected task.

---

## 16. PyTorch and programming areas to revise

Be ready to discuss:

- Dataset and DataLoader design for variable-length sequences.
- Collate functions, padding, and attention masks.
- Cross-entropy, weighted losses, focal loss, MAE, and multi-task loss weighting.
- Optimizers such as AdamW.
- Learning-rate schedules and warm-up.
- Mixed precision and gradient accumulation.
- Gradient clipping for sequential models.
- Checkpointing and resuming.
- Reproducibility and random seeds.
- Training/validation/test separation.
- Distributed training at a conceptual level.
- Profiling memory and sequence-length bottlenecks.
- Unit tests for geometry transformations and label generation.

### Multi-task loss example

`L = lambda_cls * L_classification + lambda_reg * L_displacement + lambda_geo * L_geometry + lambda_topo * L_topology`

Explain that weights can be tuned, normalized by scale, or learned, and that competing objectives must be monitored separately.

### Variable-length batching

> I would store each geometry as a variable-length tensor, pad within each batch, and pass a padding mask to the Transformer. I would consider length-bucketed batching for efficiency. For multiple objects, I would maintain object and ring identifiers so padding does not destroy hierarchy.

---

## 17. Tailoring your experience to the role

### Your verified strengths

- M.Sc. in Geodesy and Geoinformation Science from TU Berlin.
- Python and PyTorch.
- Vision Transformer experience.
- Geometric registration and 3D reconstruction.
- Semantic initialization guiding geometric processing.
- Large, reproducible evaluation pipelines.
- Quantitative benchmarking and failure analysis.
- Geospatial processing background.
- English communication and familiarity with TU Berlin.

### Your honest gap

You have not yet led a research project specifically on cartographic generalization or generative Transformers for vector maps.

### Best bridge statement

> My recent application domain has been medical and 3D computer vision, but the transferable scientific core is geometric representation, spatial-semantic integration, reproducible PyTorch implementation, and quantitative evaluation. Cart2Former changes the data structure and domain constraints, so I am studying polygon representation, cartographic operators, and vector validity directly rather than assuming that image or point-cloud methods transfer unchanged.

### How ViT experience transfers

> My ViT work gives me practical experience with tokenization, attention, positional encodings, transfer learning, and Transformer training. In Cart2Former the tokens would be vertices, edges, objects, or operations rather than fixed image patches. The new research challenges are cyclic order, topology, variable length, continuous coordinates, and spatial-semantic context.

### How registration transfers

> Registration taught me to reason carefully about coordinate systems, transformations, initialization, geometric error metrics, outliers, and failure cases. Those habits are directly useful for vector-map learning, even though the operators and cartographic constraints are different.

### Accuracy note

When discussing your 19,157 figure, call them evaluated surface vertices, not independent cases or patients.

---

## 18. High-probability interview answers

### Tell us about yourself

> Thank you for inviting me. I recently completed my master's degree in Geodesy and Geoinformation Science at TU Berlin, specializing in photogrammetric computer vision and remote sensing. My main strength is developing reproducible methods for geometric problems using Python and PyTorch. During my research at Institut Pascal, I developed a coarse-to-fine 3D registration framework combining semantic initialization, geometric alignment, and ICP refinement, together with quantitative evaluation and failure analysis. I currently work on monocular 3D facial reconstruction. Although my recent applications have focused on medical and 3D data, the underlying research skills - geometric representation, spatial-semantic reasoning, model implementation, and benchmarking - connect closely to Cart2Former. I am particularly interested in learning directly from vector coordinates and developing context-aware Transformer models that produce valid, usable geospatial outputs.

### Why this job?

> This position brings together the three areas I want to develop long term: geoinformatics, geometric deep learning, and independent research. Cart2Former is attractive because it is not simply applying a standard model; it requires new representations, encodings, datasets, and evaluation methods for vector geometry. I can contribute immediately to Python/PyTorch implementation, data pipelines, and experimental evaluation while deepening my cartographic-generalization expertise.

### What is your main gap?

> My main gap is direct research experience in cartographic generalization. I do not want to overstate it. I am addressing it by studying the operators, constraints, paired multi-scale data problem, and your group’s recent coordinate-based work. The vacancy treats basic cartographic-generalization knowledge as advantageous, while my relevant strengths are geometric learning, geospatial education, PyTorch, and independent experimental work. I believe this is a focused and learnable transition.

### What would you do first?

> First, I would clarify the first target operator, dataset, map scales, and success criteria. I would audit geometry and metadata quality, reproduce the strongest classical and recent neural baselines, and create geographic train/validation/test splits. Then I would implement a small Transformer baseline with transparent coordinate and structural encodings. Only after validating the baseline would I add hierarchy, context, self-supervised pretraining, and constrained generation, with ablations at each stage.

### How would you encode a polygon?

> I would represent it as an ordered sequence of vertex or edge tokens in a projected coordinate system. Each token could combine normalized local coordinates, edge vectors, lengths, turning angles, ring identity, semantic attributes, and target scale. I would preserve global location and neighbourhood separately. Padding masks handle variable length, while loop-origin, orientation, holes, and multipolygons require explicit design and tests. A hierarchical architecture could encode vertices within objects and then spatial context between objects.

### How would you design self-supervision?

> I would start with masked vertex or edge reconstruction and geometry denoising, because both use unlabelled vector maps. I would also test prediction of local geometric properties such as edge lengths and turning angles, similar in spirit to the group’s geometric pretraining. Contrastive pairs could use safe transformations such as changed loop origin, but augmentations must not alter cartographic meaning. I would evaluate pretraining through low-label fine-tuning and cross-city transfer.

### How would you enforce validity?

> I would combine explicit structural inputs, constrained output operations or decoding masks, geometry-aware losses, and deterministic GIS validation or repair. Hard constraints such as valid ring structure may be better guaranteed algorithmically, while softer qualities such as characteristic shape can be optimized through losses and learned representations. I would report validity and conflict rates, not only prediction accuracy.

### Why not only use a GNN?

> GNNs are a strong baseline because vector maps have explicit graph structure. But polygon rings have ordered cyclic structure and context may require long-range interactions across objects. A Transformer can model global dependencies and autoregressive output, while a GNN provides useful locality and topology bias. I would compare them fairly and consider a hybrid rather than assuming the Transformer must always win.

### Why not only use Douglas-Peucker?

> Douglas-Peucker is efficient and useful as a baseline, but it optimizes geometric deviation rather than the complete cartographic objective. It may remove characteristic corners, fail to preserve orthogonality, ignore semantics and neighbours, and cannot perform aggregation or context-aware displacement by itself.

### How would you evaluate a generated footprint?

> I would combine operation metrics, geometric similarity, topological validity, cartographic constraints, and robustness. Examples include macro-F1 for vertex operations, displacement MAE, IoU, Hausdorff distance, area and orientation error, self-intersection rate, minimum-separation conflicts, and cross-city evaluation. I would also inspect failures visually and, where possible, include expert cartographic assessment.

---

## 19. Questions Professor Kada or the panel may ask

### Motivation and fit

- Tell us about yourself.
- Why Cart2Former and why this group?
- Why change from medical/3D vision to cartographic generalization?
- Which parts of your experience transfer immediately?
- What is your main skill gap?
- Why should we select you over a candidate with more cartography experience?
- What do you want to achieve in three years?
- Are you interested in a doctorate?

### Professor/group knowledge

- Which of our papers have you read?
- Explain the 2025 coordinate-CNN paper.
- Why did the simple CNN perform competitively with GNNs?
- What limitations do you see in that work?
- How would you extend it with a Transformer?
- How does the DETR building-plane paper relate to your experience?

### Cartography

- Define cartographic generalization.
- Explain simplification, aggregation, displacement, and network thinning.
- What should be preserved during building simplification?
- Why is map scale important?
- Why is cartographic generalization difficult to formalize completely?
- What is the difference between geometric and semantic generalization?
- How would you evaluate legibility or characteristic shape?

### Vector geometry

- How would you encode a polygon, polyline, multipolygon, or road network?
- Which invariances should a polygon encoder have?
- How do you represent holes?
- How do you handle variable numbers of vertices?
- What is lost by normalizing coordinates?
- How would you include neighbouring objects?
- What makes vector data different from point clouds?

### Transformers and generation

- Explain self-attention.
- Why scale dot products by the square root of key dimension?
- Why are positional encodings necessary?
- Absolute versus relative positional encoding?
- Encoder-only, decoder-only, or encoder-decoder for Cart2Former?
- Continuous versus quantized coordinates?
- How would an autoregressive polygon decoder stop?
- What is exposure bias?
- How would you reduce quadratic attention cost?

### Self-supervision

- Why is self-supervision useful here?
- Propose three pretext tasks.
- How do you prevent augmentation from changing semantics?
- How would you prove that pretraining transfers?
- What negative transfer risks exist?

### Dataset and evaluation

- How would you combine heterogeneous geospatial sources?
- How would you match objects across scales?
- How would you separate temporal change from generalization?
- What is spatial leakage?
- Which baselines would you choose?
- Why are accuracy and IoU insufficient?
- How do Hausdorff and Chamfer distance differ?
- How would you test cross-city generalization?

### Research and behaviour

- Describe a research problem you solved independently.
- Describe a failed experiment.
- How do you debug a model that does not learn?
- How do you make experiments reproducible?
- How do you prioritize when the project is ambiguous?
- How do you respond to critical feedback?
- Describe a disagreement with a collaborator.
- How would you prepare a publication from negative results?

---

## 20. Ten-day preparation schedule

### Day 1 - Role and project

- Read the vacancy and project description.
- Recite the one-sentence project explanation.
- Prepare tell-me-about-yourself and why-this-job answers.

### Day 2 - Cartographic foundations

- Learn every core operator and constraint.
- Draw simplification, aggregation, displacement, and thinning examples.
- Explain why scale changes the correct output.

### Day 3 - Kada papers

- Deep-read the 2024 and 2025 building-simplification papers.
- Prepare a two-minute summary, one limitation, and one extension for each.

### Day 4 - Transformer foundations

- Review attention, masking, encoder/decoder, positional encoding, and complexity.
- Connect ViT patches to vector tokens without claiming they are identical.

### Day 5 - Vector representation

- Study polygon invariances, rings, holes, variable length, and context.
- Sketch two candidate Cart2Former architectures.

### Day 6 - Generation and self-supervision

- Compare operation prediction, discrete coordinate tokens, continuous regression, and hybrid generation.
- Design three self-supervised tasks and their evaluation.

### Day 7 - Dataset and benchmark

- Design an end-to-end data pipeline.
- Prepare metrics, baselines, geographic splits, and ablations.

### Day 8 - Your evidence

- Rehearse thesis, ViT work, registration, and one research failure.
- Verify every number and technical claim.

### Day 9 - Full mock interview

- 10-minute introduction/research presentation.
- 40 minutes of technical and behavioural questions.
- Review recording for clarity and answer length.

### Day 10 - Light review

- Review flashcards and diagrams.
- Print CV and one-page research idea.
- Confirm location and route.
- Sleep normally; do not learn a new field overnight.

---

## 21. Questions you should ask them

Choose three or four:

- Which generalization operator and geometry type will be the first project milestone?
- What datasets and map scales are planned initially?
- Will the first benchmark use authoritative paired multi-scale maps, synthetic labels, or both?
- How do you currently envision spatial-semantic context being represented?
- Is the project expected to begin from the group’s recent CNN/GNN simplification work?
- What would successful progress look like after six months?
- How is Cart2Former work divided across the current team?
- Is doctoral research expected or supported within this position?
- Which conferences or journals are the primary targets?

Avoid asking questions whose answers are plainly written in the vacancy.

---

## 22. Final interview principles

1. Lead with the research problem, not a list of software.
2. Distinguish verified experience from ideas you would investigate.
3. Never describe polygons as merely unordered point clouds.
4. Do not assume a Transformer is automatically superior to a GNN or classical method.
5. Mention baselines, ablations, geographic splits, and failure analysis.
6. Pair learned methods with explicit geometric validation.
7. Use the phrase "evaluated surface vertices" for the 19,157 figure.
8. Do not mention desperation or that this is your only option.
9. If you do not know, state what you know, identify the uncertainty, and propose how you would test it.
10. Finish answers with relevance to Cart2Former.

### A strong way to handle an unknown question

> I have not implemented that specific method, so I do not want to claim direct experience. My current understanding is [...]. I would compare it against [...] using [...], and the main risk I would check is [...].

That is a research answer, not a weak answer.

