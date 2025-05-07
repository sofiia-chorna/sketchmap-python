from typing import Literal, Optional, Tuple

import torch
from tqdm import tqdm

from src.commands.distance import DistanceCalculator
from src.utils.const import DEVICE


def run_get_landmarks(
    points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    weighted: bool = False,
    num: int = 1000,
    mode: Literal["minmax"] = "minmax",
    metric: Literal["euclidean", "dot", "pbc", "sphere"] = "euclidean",
    period: float = 0.0,
    sphere_period: float = 0.0,
    first_index: int = -1,
    seed: int = 42,
    weight_gamma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    torch.manual_seed(seed)

    N, _D = points.shape

    if weights is None:
        weights = torch.ones(N, device=DEVICE)
    elif weights.device != DEVICE:
        weights = weights.to(DEVICE)

    calculator = DistanceCalculator(metric, period, sphere_period)

    if mode == "minmax":
        landmarks, landmark_indices, landmark_weights = _minmax_selection(
            calculator=calculator,
            points=points,
            weights=weights,
            num=num,
            first_index=first_index,
            weighted=weighted,
            weight_gamma=weight_gamma,
        )
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")

    return landmarks, landmark_indices, landmark_weights


def _minmax_selection(
    calculator: DistanceCalculator,
    points: torch.Tensor,
    weights: torch.Tensor,
    num: int,
    first_index: int = -1,
    weighted: bool = False,
    weight_gamma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    N, _D = points.shape

    landmark_indices = torch.zeros(num, dtype=torch.long, device=DEVICE)

    # select first indices
    if first_index < 0:
        landmark_indices[0] = torch.randint(0, N, (1,), device=DEVICE)
    else:
        landmark_indices[0] = first_index

    all_dists = calculator.pairwise_distances(points, points)
    min_dists = all_dists[landmark_indices[0]]

    pbar = tqdm(range(1, num), desc="Selecting landmarks")

    for i in pbar:
        # find point with maximum min distance
        max_idx = torch.argmax(min_dists).item()
        landmark_indices[i] = max_idx

        new_dists = all_dists[max_idx]

        min_dists = torch.minimum(min_dists, new_dists)

        pbar.set_postfix({"max_dist": min_dists.max().item()})

    landmarks = points[landmark_indices]

    if weighted:
        landmark_weights = _compute_voronoi_weights(
            points, weights, landmarks, calculator, weight_gamma
        )
    else:
        landmark_weights = torch.ones(num, device=DEVICE) / num

    return landmarks, landmark_indices, landmark_weights


def _compute_voronoi_weights(
    points: torch.Tensor,
    weights: torch.Tensor,
    landmarks: torch.Tensor,
    calculator: DistanceCalculator,
    weight_gamma: float = 1.0,
) -> torch.Tensor:
    n_landmarks = landmarks.shape[0]

    # calculate distances from all points to all landmarks
    distances = torch.stack(
        [
            torch.tensor(
                [calculator.single_distance(p, l) for l in landmarks], device=DEVICE
            )
            for p in points
        ]
    )

    # find nearest landmark for each point
    min_indices = torch.argmin(distances, dim=1)

    # vectorize weight accumulation
    landmark_weights = torch.zeros(n_landmarks, device=DEVICE)
    landmark_weights.scatter_add_(0, min_indices, weights)

    if weight_gamma != 1.0:
        landmark_weights = landmark_weights.pow(weight_gamma)

    # normalize
    weight_sum = landmark_weights.sum()
    if weight_sum > 1e-10:
        landmark_weights = landmark_weights / weight_sum
    else:
        landmark_weights = torch.ones_like(landmark_weights) / n_landmarks

    return landmark_weights
