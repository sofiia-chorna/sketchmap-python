from typing import Literal, Optional, Tuple

import numpy as np
import torch

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
        return_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float32)

        points = points.to(DEVICE)
        n = points.shape[0]

        match self.metric:
            case "euclidean":
                dist_matrix = torch.cdist(points, points, p=2)

            case "dot":
                dist_matrix = -torch.matmul(points, points.T)

            case "pbc":
                diff = points.unsqueeze(1) - points.unsqueeze(0)
                diff = torch.abs(diff)
                diff = torch.where(diff > self.period / 2, self.period - diff, diff)
                dist_matrix = torch.norm(diff, dim=-1)

            case "sphere":
                norms = torch.norm(points, dim=1, keepdim=True)
                cos_angles = torch.matmul(points, points.T) / (norms * norms.T)
                cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
                angles = torch.acos(cos_angles)
                dist_matrix = self.sphere_period * angles / (2 * np.pi)

            case _:
                raise ValueError(f"Unknown distance metric: {self.metric}")

        if torch.isnan(dist_matrix).any():
            nan_count = torch.isnan(dist_matrix).sum()
            raise ValueError(f"Distance matrix contains {nan_count} NaN values")

        if return_weights and weights is not None:
            weights = weights.to(DEVICE)
            if weights.shape != (n,):
                raise ValueError(
                    f"Weights shape {weights.shape} must match points ({n},)"
                )

            # compute weight products for all pairs
            weight_matrix = weights.unsqueeze(1) * weights.unsqueeze(0)

            return dist_matrix, weight_matrix

        return dist_matrix, torch.ones(len(points), device=DEVICE)
