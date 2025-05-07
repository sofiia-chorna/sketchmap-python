from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from src.commands.distance import DistanceCalculator


def run_get_landmarks(
    points: torch.Tensor,
    weights: torch.Tensor,
    weighted: bool = False,
    num: int = 1000,
    mode: Literal["minmax", "random"] = "minmax",
    metric: Literal["euclidean", "dot", "pbc", "sphere"] = "euclidean",
    period: float = 0.0,
    sphere_period: float = 0.0,
    first_index: int = -1,
    seed: int = 42,
):
    calculator = DistanceCalculator(metric, period, sphere_period)

    if mode == "minmax":
        return _minmax_selection(
            calculator,
            points,
            weights,
            num,
            first_index,
            weighted,
            seed,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _minmax_selection(
    calculator,
    points,
    weights,
    num,
    first_index,
    weighted=False,
    seed=42,
    min_distance=1e-4,
):
    N, D = points.shape

    # 1. Normalize points using robust scaling
    points_mean = points.mean(dim=0)
    points_std = points.std(dim=0)
    points_norm = (points - points_mean) / (points_std + 1e-8)

    # 2. Initialize landmarks and distances
    landmarks = torch.zeros((num, D), device=points.device)
    landmark_indices = torch.zeros(num, dtype=torch.long, device=points.device)

    # 3. Select first point
    if first_index < 0:
        first_idx = torch.randint(0, N, (1,), device=points.device).item()
    else:
        first_idx = first_index

    landmarks[0] = points_norm[first_idx]
    landmark_indices[0] = first_idx

    # 4. Compute all distances to first landmark
    dist_matrix = torch.cdist(points_norm, landmarks[:1].unsqueeze(0)).squeeze()
    min_dists = dist_matrix.clone()

    selected_indices = {first_idx}

    # 5. Select remaining landmarks
    for i in tqdm(range(1, num), desc="Selecting landmarks"):
        # Find valid candidates
        valid_mask = torch.ones(N, dtype=torch.bool, device=points.device)
        valid_mask[list(selected_indices)] = False
        valid_mask &= min_dists > min_distance

        if not valid_mask.any():
            valid_mask = torch.ones(N, dtype=torch.bool, device=points.device)
            valid_mask[list(selected_indices)] = False

        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            raise ValueError("No valid points remaining for selection")

        # Select point with maximum minimum distance
        max_idx = valid_indices[torch.argmax(min_dists[valid_indices])].item()

        # Add to landmarks
        landmarks[i] = points_norm[max_idx]
        landmark_indices[i] = max_idx
        selected_indices.add(max_idx)

        # Update minimum distances
        new_dists = torch.cdist(
            points_norm, landmarks[i : i + 1].unsqueeze(0)
        ).squeeze()
        min_dists = torch.minimum(min_dists, new_dists)

    # 6. Convert back to original space
    landmarks = landmarks * (points_std + 1e-8) + points_mean

    if weighted:
        landmark_weights = _compute_voronoi_weights(
            points, weights, landmarks, calculator
        )
    else:
        landmark_weights = torch.ones(num, device=points.device)

    return landmarks, landmark_weights


def _compute_voronoi_weights(points, weights, landmarks, calculator):
    # Compute distances from all points to landmarks
    dists = calculator.pairwise_distances(points, landmarks)  # N x num_landmarks
    min_indices = torch.argmin(
        dists, dim=1
    )  # For each point, index of nearest landmark
    landmark_weights = torch.zeros(len(landmarks), device=points.device)
    landmark_weights.index_add_(0, min_indices, weights)
    return landmark_weights
