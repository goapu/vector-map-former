# Data

Raw and processed datasets are deliberately excluded from source control.

## MapGeneralizer labelled benchmark

Source repository: <https://github.com/chouisgiser/MapGeneralizer>

Clone it outside this repository, or place its `data/input` directory at:

```text
data/raw/mapgeneralizer/data/input/
```

Required files:

```text
vertex_train.npy
vertex_valid.npy
vertex_test.npy
```

The public arrays are NumPy object arrays and therefore require
`allow_pickle=True`. Loading pickled arrays can execute malicious code. This
project refuses arbitrary filenames and documents the risk, but users must only
load files obtained from a trusted source.

Verified public label contract:

```text
column 0: building ID
column 1: vertex order
columns 2-3: projected source coordinates
columns 4-5: supplied normalized coordinates (not used by default)
columns 6-9: supplied geometric features (not used by default)
column 10: action, where 0=REMOVE, 1=KEEP, 2=MOVE
columns 11-12: signed movement components along incident edges
```

The repository does not visibly contain a dataset licence. Do not redistribute
the arrays. Cite the associated paper and provide download instructions.
