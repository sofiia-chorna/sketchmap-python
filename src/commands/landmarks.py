from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.commands.distance import DistanceCalculator
from src.utils.const import DEVICE
from src.utils.logger import logger


def run_get_landmarks(
    points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    compute_weights: bool = False,
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

    calculator = DistanceCalculator(metric, period, sphere_period)

    if mode == "minmax":
        landmarks, landmark_indices, landmark_weights = _minmax_selection(
            calculator=calculator,
            points=points,
            weights=weights,
            num=num,
            first_index=first_index,
            compute_weights=compute_weights,
            weight_gamma=weight_gamma,
        )
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")

    if compute_weights:
        weights = landmark_weights.unsqueeze(-1)
    else:
        weights = torch.ones(len(landmarks), device=DEVICE).unsqueeze(-1) / len(
            landmarks
        )

    return landmarks, landmark_indices, landmark_weights


def _minmax_selection(
    calculator: DistanceCalculator,
    points: torch.Tensor,
    weights: torch.Tensor,
    num: int,
    first_index: int = -1,
    compute_weights: bool = False,
    weight_gamma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    N = points.shape[0]

    landmark_indices = torch.zeros(num, dtype=torch.long, device=DEVICE)

    if first_index < 0:
        landmark_indices[0] = torch.randint(0, N, (1,), device=DEVICE)
    else:
        landmark_indices[0] = first_index

    all_dists = calculator.pairwise_distances(points, points)
    min_dists = all_dists[landmark_indices[0]]

    pbar = tqdm(range(1, num), desc="Selecting landmarks")

    for i in pbar:
        # find point with maximum min distance
        max_id = torch.argmax(min_dists).item()
        landmark_indices[i] = max_id

        new_dists = all_dists[max_id]

        min_dists = torch.minimum(min_dists, new_dists)

        pbar.set_postfix({"max_dist": min_dists.max().item()})

    landmarks = points[landmark_indices]

    if compute_weights:
        landmark_weights = _compute_voronoi_weights(
            points, weights, landmarks, weight_gamma
        )
    else:
        landmark_weights = torch.ones(num, device=DEVICE) / num

    return landmarks, landmark_indices, landmark_weights


def _compute_voronoi_weights(
    points: torch.Tensor,
    weights: torch.Tensor,
    landmarks: torch.Tensor,
    weight_gamma: float = 1.0,
) -> torch.Tensor:
    n_landmarks = landmarks.shape[0]

    dist_matrix = torch.cdist(points, landmarks, p=2)

    # find nearest landmark for each point
    min_indices = torch.argmin(dist_matrix, dim=1)

    # accumulate weights to the nearest landmark
    landmark_weights = torch.zeros(n_landmarks, device=DEVICE)
    landmark_weights.scatter_add_(0, min_indices, weights)

    if weight_gamma != 1.0:
        landmark_weights = landmark_weights.pow(weight_gamma)

    # normalize
    total_weight = landmark_weights.sum()
    if total_weight > 1e-10:
        landmark_weights /= total_weight
    else:
        landmark_weights.fill_(1.0 / n_landmarks)

    return landmark_weights


def plot_pca_landmarks(
    points: torch.Tensor,
    landmarks: torch.Tensor,
    plot_filepath: str,
    title: str = "Landmark selection (PCA)",
):
    points_np = points.cpu().numpy()
    landmarks_np = landmarks.cpu().numpy()

    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points_np)
    landmarks_2d = pca.transform(landmarks_np)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        points_2d[:, 0], points_2d[:, 1], c="gray", alpha=0.3, label="original points"
    )
    plt.scatter(
        landmarks_2d[:, 0], landmarks_2d[:, 1], c="red", s=20, label="landmarks"
    )
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)


def get_landmark_stat(
    points: torch.Tensor,
    landmarks: torch.Tensor,
    calculator: DistanceCalculator,
    stat_filepath: str,
):
    # 1. verify distances from all points to landmarks
    point_to_landmark_dists = calculator.pairwise_distances(points, landmarks)
    min_dists = point_to_landmark_dists.min(dim=1).values

    coverage_stats = {
        "avg_dist_to_landmark": min_dists.mean().item(),
        "max_dist_to_landmark": min_dists.max().item(),
        "min_dist_to_landmark": min_dists.min().item(),
        "coverage_ratio": min_dists.mean().item()
        / point_to_landmark_dists.max().item(),
    }

    # 2. verify distances between landmarks
    landmark_dists = calculator.pairwise_distances(landmarks, landmarks)

    # fill diagonal with infinity to ignore self-distances
    landmark_dists.fill_diagonal_(float("inf"))

    min_landmark_dists = landmark_dists.min(dim=1).values
    separation_stats = {
        "min_landmark_separation": min_landmark_dists.min().item(),
        "avg_landmark_separation": min_landmark_dists.mean().item(),
        "max_landmark_separation": min_landmark_dists.max().item(),
        "separation_ratio": min_landmark_dists.min().item()
        / landmark_dists.max().item(),
    }

    # 3. check for duplicates or near-duplicates
    duplicate_threshold = 1e-6
    num_too_close = (landmark_dists < duplicate_threshold).sum().item() // 2
    if num_too_close > 0:
        logger.warning(
            f"Found {num_too_close} landmark pairs closer than {duplicate_threshold}"
        )

    with open(stat_filepath, "w") as file:
        file.write("Landmark coverage statistics:\n")
        for k, v in coverage_stats.items():
            file.write(f"{k:<30}: {v:.6f}\n")

        file.write("\nLandmark separation statistics:\n")
        for k, v in separation_stats.items():
            file.write(f"{k:<30}: {v:.6f}\n")
