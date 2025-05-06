from typing import Literal, Optional, Tuple

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
        if metric == "pbc" and period <= 0:
            raise ValueError("Period must be positive for PBC metric")
        if metric == "sphere" and sphere_period <= 0:
            raise ValueError("Sphere period must be positive")

    def single_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        match self.metric:
            case "euclidean":
                return torch.norm(x - y)
            case "dot":
                return -torch.dot(x, y)
            case "pbc":
                diff = torch.abs(x - y)
                diff = torch.where(diff > self.period / 2, self.period - diff, diff)
                return torch.norm(diff)
            case "sphere":
                cos_angle = torch.dot(x, y) / (torch.norm(x) * torch.norm(y))
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                angle = torch.acos(cos_angle)
                return self.sphere_period * angle / (2 * np.pi)
            case _:
                raise ValueError(f"Unknown distance metric: {self.metric}")

    def pairwise_distances(
        self,
        points: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        weighted: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float32)

        points = points.to(DEVICE)
        n = points.shape[0]

        if weights is not None:
            weights = weights.to(DEVICE)
            if weights.shape != (n,):
                raise ValueError(
                    f"Weights shape {weights.shape} must match points ({n},)"
                )

        # Compute pairwise distances
        match self.metric:
            case "euclidean":
                dist_matrix = torch.cdist(points, points, p=2)
            case "dot":
                dist_matrix = -torch.matmul(points, points.T)
            case "pbc":
                # Pairwise differences
                diff = points.unsqueeze(1) - points.unsqueeze(0)  # Shape: (n, n, d)
                diff = torch.abs(diff)
                diff = torch.where(diff > self.period / 2, self.period - diff, diff)
                dist_matrix = torch.norm(diff, dim=-1)
            case "sphere":
                # Great-circle distances
                norms = torch.norm(points, dim=1, keepdim=True)
                cos_angles = torch.matmul(points, points.T) / (norms * norms.T)
                cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
                angles = torch.acos(cos_angles)
                dist_matrix = self.sphere_period * angles / (2 * np.pi)
            case _:
                raise ValueError(f"Unknown distance metric: {self.metric}")

        rows, cols = torch.triu_indices(n, n, offset=1)
        distances = dist_matrix[rows, cols]

        # Compute weights
        if weighted and weights is not None:
            dweights = weights[rows] * weights[cols]
        else:
            dweights = torch.ones_like(distances)

        return distances.cpu().numpy(), dweights.cpu().numpy()
