# sketchmap-python

Reimplementation of sketch-map dimentionality reduction algorithm in Python.

## CLI Commands

The following cli commands are available:

| Command            | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `select-landmarks`             | Sample points using Farthest-Point selection (FPS)  |
| `analyse`            | Analyse the density of the pairwise distances in the high dimension data                      |


### select-landmarks

Samples representative points from a high-dimensional dataset using FPS.

```bash
python3 main.py select-landmarks --hdim-filepath high_dimension_data.pt --run-check
```
Replace `high_dimension_data.pt` with your filename to select points from.

`--run-check` is **optional** but highly recommended. It calculates the coverage and separation statistics on the selected landmarks and generates a plot with selected points highlighed over the PCA of all highdimentional data. It help to evaluate quantitavely and qualitatively the sampling.

By default `min-max` selection algorithm is used (currently the only one implemented): the first point is selected randomly, and each consequative point is selected being the farthest from the already selected points. The algorithm stops once the desired number of points is reached.


| Option            | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `--hdim-filepath` / `--P`              | Path to input high-dimensional data file (required)                       |
| `--num`/ `--n`             | Number of landmarks to select (default: 1000)  |
| `--select-mode`             | Selection mode (currently only "minmax" supported)  |
| `--save-indices` / `--i`             | Save original indices of selected landmarks to the first column  |
| `--compute-weights`              | Save a weight of landmark proportional to the number of points in its Voronoi cell ("region of influence") as the last column |
| `--metric`             | Distance metric ("euclidean" - default, "dot", "pbc", or "sphere")  |
| `--period`             | Periodicity for pbc metric  |
| `--sphere-period`             |  Periodicity for spherical distance metric  |
| `--numpy`             |  Save in numpy format  |
| `--output-filepath`             | Path to the custom output landmarks  |
| `--run-check`             |  Run and save coverage and separation statistics on the selected points as well as a plot with PCA with on all data with highlighted selected points  |


When `--compute-weights` is enabled, the algorithm assigns to each landmark a Voronoi like weight that reflects the number of points that are closer to that landmark than to any other one.


### analyse

Analyse the density of the pairwise distances in the high dimension. This command generates a histogram and suggests the parameters to run sketch-map dimentionality reduction.

```bash
python3 -u main.py analyse --hdim-filepath high_dimension_data.pt
```
Replace `high_dimension_data.pt` with your filename.

| Option            | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `--hdim-filepath` / `--P`              | Path to input high-dimensional data file (required)                       |
| `--n-bins`             | Number of bins for histogram  |
| `--max-distance`             | Maximum distance to consider for generating the histogram  |
| `--high-dimension`             | Dimentionality of the data in case it does not correspond to shape[2] of data|
| `--metric`             | Distance metric ("euclidean" - default, "dot", "pbc", or "sphere")  |
| `--period`             | Periodicity for pbc metric  |
| `--sphere-period`             |  Periodicity for spherical distance metric  |
| `--output-filepath`             | Path to the custom output landmarks  |
| `--weighted`             | If the points are weighted in the input high-dimentional data |
