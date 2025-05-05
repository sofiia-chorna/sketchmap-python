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
    points_np = points.cpu().numpy() if torch.is_tensor(points) else points
    weights_np = weights.cpu().numpy() if torch.is_tensor(weights) else weights
    weights_np = weights_np if weighted else np.ones(len(points_np))

    dist_calculator = DistanceCalculator(
        metric=metric, period=period, sphere_period=sphere_period
    )

    match mode:
        case "minmax":
            return _minmax_selection(
                dist_calculator,
                points_np,
                weights_np,
                num,
                first_index,
                weighted,
                seed,
            )
        case _:
            raise ValueError(f"Unsupported mode: {mode}")


def _minmax_selection(
    calculator, points, weights, num, first_index, weighted=False, seed=42
):
    N, D = points.shape
    landmarks = np.zeros((num, D))
    landmark_indices = np.zeros(num, dtype=int)
    landmark_weights = np.zeros(num)

    # select first point
    if first_index < 0:
        rng = np.random.RandomState(seed)
        first_idx = rng.randint(N)
    else:
        first_idx = first_index

    landmarks[0] = points[first_idx]
    landmark_indices[0] = first_idx

    # init minimum distances
    min_dists = np.full(N, np.inf)
    for j in tqdm(range(N), desc="Init min distances"):
        min_dists[j] = calculator.single_distance(landmarks[0], points[j])

    # track selected indices to avoid duplicates
    selected_indices = {first_idx}

    # select remaining points
    for i in tqdm(range(1, num), desc="Selecting remaining points"):
        # find point with max min distance
        valid_indices = [j for j in range(N) if j not in selected_indices]
        if not valid_indices:
            raise ValueError("Not enough unique points to select landmarks")

        max_idx = valid_indices[np.argmax(min_dists[valid_indices])]
        landmarks[i] = points[max_idx]
        landmark_indices[i] = max_idx
        selected_indices.add(max_idx)

        # update min distances
        for j in range(N):
            d = calculator.single_distance(landmarks[i], points[j])
            if d < min_dists[j]:
                min_dists[j] = d

    # compute weights if needed
    if weighted:
        landmark_weights = _compute_voronoi_weights(
            points, weights, landmarks, calculator
        )

    return landmarks, landmark_weights


def _compute_voronoi_weights(points, weights, landmarks, calculator):
    num_landmarks = len(landmarks)
    landmark_weights = np.zeros(num_landmarks)

    for j in tqdm(range(len(points)), desc="Computing Voronoi weights"):
        min_dist = np.inf
        min_idx = 0

        for i in range(num_landmarks):
            d = calculator.single_distance(landmarks[i], points[j])
            if d < min_dist:
                min_dist = d
                min_idx = i

        landmark_weights[min_idx] += weights[j]

    return landmark_weights
