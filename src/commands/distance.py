from typing import Literal

import numpy as np
import torch
from scipy.spatial.distance import pdist

from src.utils.const import DEVICE


class DistanceCalculator:
    def __init__(
        self,
        metric: Literal["euclidean", "dot", "pbc", "sphere"] = "euclidean",
        period: float = 0.0,
        sphere_period: float = 0.0,
    ):
        self.metric = metric
        self.period = period
        self.sphere_period = sphere_period

    def _single_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        match self.metric:
            case "euclidean":
                return np.linalg.norm(x - y)
            case "dot":
                return -np.dot(x, y)
            case "pbc":
                diff = np.abs(x - y)
                diff = np.where(diff > self.period / 2, self.period - diff, diff)
                return np.linalg.norm(diff)
            case "sphere":
                # simplified spherical distance, TODO: fix
                return np.linalg.norm(x - y)
            case _:
                raise ValueError(f"Unknown distance metric: {self.metric}")

    def pairwise_distances(
        self, points: torch.Tensor, weights: torch.Tensor = None, weighted: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if torch.is_tensor(points):
            if DEVICE == "gpu":
                points = points.to(DEVICE)
                if weights is not None:
                    weights = weights.to(DEVICE)

            # pairwise distances
            match self.metric:
                case "euclidean":
                    dist_matrix = torch.cdist(points, points)
                case "dot":
                    dist_matrix = -torch.matmul(points, points.T)
                case _:
                    raise ValueError(f"Unknown distance metric: {self.metric}")

            # here we get upper trianglular part of the matrix to exlude duplications and self-distances
            rows, cols = torch.triu_indices(len(points), len(points), offset=1)
            distances = dist_matrix[rows, cols]

            if weighted and weights is not None:
                dweights = torch.outer(weights, weights)[rows, cols]
            else:
                dweights = torch.ones_like(distances)

            return distances.cpu().numpy(), dweights.cpu().numpy()

        else:  # numpy array
            if self.metric == "euclidean":
                distances = pdist(points, "euclidean")
            else:
                # for non-euclidean metrics with numpy
                n = len(points)
                distances = np.zeros(n * (n - 1) // 2)
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        distances[idx] = self._single_distance(points[i], points[j])
                        idx += 1

            if weighted and weights is not None:
                indices = np.triu_indices(len(weights), k=1)
                dweights = np.outer(weights, weights)[indices]
            else:
                dweights = np.ones_like(distances)

            return distances, dweights
