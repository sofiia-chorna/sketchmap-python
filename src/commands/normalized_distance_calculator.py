import numpy as np
import torch

from src.commands.distance_calculator import DistanceCalculator


class NormalizedDistanceCalculator(DistanceCalculator):
    def __init__(self, normalization_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.normalization_factor = normalization_factor

    def single_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        distance = super().single_distance(x, y)
        return distance / self.normalization_factor

    def pairwise_distances(self, points_1: torch.Tensor, points_2: torch.Tensor):
        dist_matrix = super().pairwise_distances(points_1, points_2)
        return dist_matrix / self.normalization_factor


def compute_normalization_factor(
    features: torch.Tensor, metric: str = "euclidean", n_samples: int = 10000
) -> float:
    """Compute normalization factor to make mean distance = 1"""
    calculator = DistanceCalculator(metric=metric)

    n = features.shape[0]
    idx1 = torch.randint(0, n, (n_samples,))
    idx2 = torch.randint(0, n, (n_samples,))

    distances = []
    for i, j in zip(idx1, idx2):
        if i != j:
            dist = calculator.single_distance(features[i], features[j])
            distances.append(dist.item())

    return np.mean(distances)
