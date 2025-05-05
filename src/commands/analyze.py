import numpy as np
import torch
from scipy.spatial.distance import pdist

from src.utils.const import DEVICE


def compute_distances(
    points: torch.Tensor, weights: torch.Tensor, weighted: bool = False
):
    # TODO: implement other distance measures, for now, only euclidean
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
