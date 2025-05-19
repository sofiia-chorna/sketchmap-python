# sketchmap-python

Reimplementation of sketch-map dimentionality reduction algorithm in Python.

## CLI Commands

The following cli commands are available:

| Command            | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `analyse`            | Analyse the density of the pairwise distances in the high dimension data                      |
| `select-landmarks`             | Sample points using Farthest-Point selection (FPS)  |


### analyse

Analyse the density of the pairwise distances in the high dimension.

```bash
python3 -u main.py analyse --P high_dimension_data.pt
```
Replace high_dimension_data.pt with your filename.

### select-landmarks

Samples representative points from a high-dimensional dataset using FPS.

```bash
python3 main.py select-landmarks --hdim-filepath high_dimension_data.pt --output_filepath sampled.pt
```
Replace high_dimension_data.pt with your filename to select points from.

By default `min-max` selection algorithm is used (currently the only one implemented): the first point is selected randomly, and each consequative point is selected being the farthest from the already selected points. The algorithm stops once the desired number of points is reached.

| Option            | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `--hdim-filepath`/ `--P` | Path to input high-dimensional data file (required)                       |
| `--num`/ `--n`             | Number of landmarks to select (default: 1000)  |
| `--select-mode`             | Selection mode (currently only "minmax" supported)  |
| `--save-indices` / `--i`             | Save original indices of selected landmarks to the first column  |
| `--output-filepath`             | Path to save output landmarks  |
| `--compute-weights`              | Calculate Voronoi weights for landmarks. Saved to the last column if true  |
| `--metric`             | Distance metric ("euclidean" - default, "dot", "pbc", or "sphere")  |
| `--period`             | Periodicity for pbc metric  |
| `--sphere-period`             |  Periodicity for spherical distance metric  |
| `--numpy`             |  Save in numpy format  |

