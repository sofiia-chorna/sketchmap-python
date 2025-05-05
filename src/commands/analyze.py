import numpy as np
import torch
from scipy.spatial.distance import pdist

from src.utils.const import DEVICE

"""
! dimdist -D 1024 -P data/mad_llfs/merged_llfs/mad_combined_val.dat -maxd 100 -nbin 500 > mad_map/histo_mad_combined_val.dat

import matplotlib.pyplot as plt
import numpy as np


data_combined = np.loadtxt('histo_mad_combined_val.dat', skiprows=1)

distances_combined = data_combined[:, 0] 
histogram_values_combined = data_combined[:, 1]

plt.plot(distances_combined, histogram_values_combined, color="red", label='combined')

plt.xlabel('distance')
plt.ylabel('prob density')
plt.legend()
plt.xlim(0,20)
plt.show()

"""


def compute_distances(
    points: torch.Tensor, weights: torch.Tensor, weighted: bool = False
):
    if torch.is_tensor(points):
        if DEVICE == "gpu":
            points, weights = points.to(DEVICE), weights.to(DEVICE)

        # pairwise distances
        dist_matrix = torch.cdist(points, points)

        # here we get upper trianglular part of the matrix to exlude duplications and self-distances
        rows, cols = torch.triu_indices(len(points), len(points), offset=1)
        distances = dist_matrix[rows, cols]

        if weighted:
            dweights = torch.outer(weights, weights)[rows, cols]
        else:
            dweights = torch.ones_like(distances)

        return distances.cpu().numpy(), dweights.cpu().numpy()

    distances = pdist(points, "euclidean")

    if weighted:
        indices = np.triu_indices(len(weights), k=1)
        dweights = torch.outer(weights, weights)[indices]
    else:
        dweights = torch.ones_like(distances)

    return distances, dweights
