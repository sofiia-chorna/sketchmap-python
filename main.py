from typing import Literal, Optional

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

# from src.commands.dimred import DimRed, auto_select_parameters
from src.commands.dimred import DimRed
from src.commands.distance import DistanceCalculator
from src.commands.distance_histogram import DistanceHistogram
from src.commands.landmarks import run_get_landmarks
from src.utils.cli import (
    hdim_filepath,
    high_dimension,
    low_dimension,
    max_distance,
    metric,
    n_bin,
    num,
    output_filepath,
    period,
    select_mode,
    sphere_period,
    weighted,
)
from src.utils.const import DEVICE
from src.utils.file import read_file
from src.utils.logger import get_id, logger

from src.commands.distrib_analyzer import DistanceDistributionAnalyzer


@click.group()
def main():
    pass


@main.command()
@hdim_filepath
@metric
@period
@sphere_period
@n_bin
@max_distance
@high_dimension
@output_filepath
@weighted
def analyse(
    hdim_filepath: str,
    metric: str,
    period: float,
    sphere_period: float,
    n_bin: int,
    max_distance: Optional[int] = None,
    high_dimension: Optional[int] = None,
    output_filepath: Optional[str] = None,
    weighted: bool = False,
):
    logger.info("Start running 'analyze'")

    params = locals()
    logger.info(f"Parameters: {params}")

    logger.info(f"Start reading highdim file: {hdim_filepath}")
    points, weights = read_file(hdim_filepath, high_dimension, weighted)
    logger.info(f"Read {len(points)} points")

    logger.info(f"Start computing distances")

    distance_calculator = DistanceCalculator(metric, period, sphere_period)

    distances, dweights = distance_calculator.pairwise_distances(
        points, weights, weighted
    )
    logger.info(f"Calculated {len(distances)} distances")

    logger.info(f"Start creating histogram")

    analyzer = DistanceHistogram(n_bins=150)
    analyzer.analyze_distance(distances)

    params = analyzer.suggest_sketchmap_params(dim=1024)

    id = get_id()
    output_filepath = output_filepath or f"analysis_report_{id}.txt"
    plot_filepath = f"distance_analysis_{id}.png"

    analyzer.plot_analysis(save_path=plot_filepath, params=params, input_path=hdim_filepath)
    analyzer.save_analysis_report(output_filepath, params=params, input_path=hdim_filepath)

    logger.info(f"Saving histogram to {output_filepath} and plot to {plot_filepath}")

    logger.info("End running 'analyze'")


def plot_high_dim_landmarks(
    points: torch.Tensor,
    landmarks: torch.Tensor,
    title: str = "Landmark Selection (PCA)",
):
    points_np = points.cpu().numpy()
    landmarks_np = landmarks.cpu().numpy()

    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(points_np)
    landmarks_2d = pca.transform(landmarks_np)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        points_2d[:, 0], points_2d[:, 1], c="gray", alpha=0.3, label="Original Points"
    )
    plt.scatter(
        landmarks_2d[:, 0], landmarks_2d[:, 1], c="red", s=20, label="Landmarks"
    )
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("landmarks.png", dpi=300)


def verify_landmarks(
    points: torch.Tensor, landmarks: torch.Tensor, calculator: DistanceCalculator
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
    landmark_dists = calculator.pairwise_distances(landmarks)

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
        warning = (
            f"Found {num_too_close} landmark pairs closer than {duplicate_threshold}"
        )
        if logger:
            logger.warning(warning)
        else:
            print(f"Warning: {warning}")

    print("\nLandmark Coverage Statistics:")
    for k, v in coverage_stats.items():
        print(f"{k:>25}: {v:.6f}")

    print("\nLandmark Separation Statistics:")
    for k, v in separation_stats.items():
        print(f"{k:>25}: {v:.6f}")

    return coverage_stats, separation_stats


@main.command()
@hdim_filepath
@num
@select_mode
@high_dimension
@output_filepath
@weighted
@click.option(
    "--metric",
    type=click.Choice(["euclidean", "dot", "pbc"]),
    default="euclidean",
    help="Distance metric",
)
@click.option(
    "--save-indices",
    "--i",
    type=bool,
    is_flag=True,
    default=False,
    help="Should indices of the selected landmarks be saved as a first column",
)
def select_landmarks(
    hdim_filepath: str,
    num: int = 1000,
    select_mode: str = "minmax",
    high_dimension: Optional[int] = None,
    output_filepath: str = "high_landmarks.dat",
    weighted: bool = True,
    metric: Literal["euclidean", "dot", "pbc", "sphere"] = "euclidean",
    save_indices: bool = False,
):
    weighted = True
    logger.info("Start running 'select-landmarks'")
    logger.info(f"Params: num={num}, select_mode={select_mode}, weighted={weighted}")

    logger.info(f"Start reading highdim file: {hdim_filepath}")
    points_np, weights_np = read_file(hdim_filepath, high_dimension, weighted)
    logger.info(f"Read {len(points_np)} points")

    print("Number of NaN in input points:", np.isnan(points_np.cpu().numpy()).sum())
    print("Number of NaN in input weights:", np.isnan(weights_np.cpu().numpy()).sum())

    points = torch.tensor(points_np, device=DEVICE)
    weights = torch.tensor(weights_np, device=DEVICE)

    unique_points = torch.unique(points, dim=0)
    print(f"Total points: {len(points)}, Unique points: {len(unique_points)}")

    logger.info("Start selecting high-landmarks")

    landmarks, indices, landmark_weights = run_get_landmarks(
        points=points,
        weights=weights,
        num=num,
        mode=select_mode,
        metric=metric,
        weighted=weighted,
        # period=period,
        # sphere_period=sphere_period,
        #  first_index=args.first_index,
        #   compute_weights=args.compute_weights,
        #   weight_gamma=args.weight_gamma,
    )

    print(
        f"# {len(landmarks)} landmark points selected out of {len(points)} and chosen by {select_mode}"
    )

    if weighted:
        weights = landmark_weights.unsqueeze(-1)
    else:
        weights = torch.ones(len(landmarks), device=DEVICE).unsqueeze(-1) / len(
            landmarks
        )

    combined = torch.cat([landmarks, weights], dim=-1)

    if save_indices:
        indices = indices.unsqueeze(-1).float()  # shape: (num, 1)
        combined = torch.cat([indices, combined], dim=-1)  # shape: (num, D+2)

    logger.info(f"Saving landmarks to {output_filepath}")
    np.savetxt(output_filepath, combined.cpu().numpy())

    calculator = DistanceCalculator(metric)
    verify_landmarks(points, landmarks, calculator)

    plot_high_dim_landmarks(points, landmarks)
    logger.info("Finished landmark selection")


@main.command()
@click.option(
    "--highlandmarks_filepath",
    type=click.Path(exists=True),
    required=True,
    help="Path to file with selected high landmarks",
)
@click.option(
    "--high-dimension",
    "-D",
    type=int,
    required=True,
    help="Dimensionality of input space",
)
@click.option(
    "--low-dimension", "-d", type=int, default=3, help="Dimensionality of output space"
)
@click.option(
    "--output",
    "-o",
    "output_filepath",
    default="low_landmarks.dat",
    help="Output file path",
)
@click.option(
    "--period",
    "-pi",
    type=float,
    default=0.0,
    help="Periodicity for PBC (0 for non-periodic)",
)
@click.option(
    "--weighted/--no-weighted", default=False, help="Use weights from input file"
)
@click.option("--dot", is_flag=True, help="Use dot product distance")
@click.option(
    "--preopt-steps", type=int, default=100, help="Number of pre-optimization steps"
)
@click.option(
    "--gopt-steps", type=int, default=3.0, help="Number of global optimization steps"
)
@click.option(
    "--imix", type=float, default=1.0, help="Mixing parameter for stress function"
)
@click.option(
    "--metric",
    type=click.Choice(["euclidean", "dot", "pbc"]),
    default="euclidean",
    help="Distance metric",
)
@click.option(
    "--sigma", type=float, help="Sigma parameter for high-dim and low-dim sigmoid"
)
@click.option("--a-hd", type=int, help="'a' parameter for high-dim sigmoid")
@click.option("--b-hd", type=int, help="'b' parameter for high-dim sigmoid")
@click.option("--a-ld", type=int, help="'a' parameter for low-dim sigmoid")
@click.option("--b-ld", type=int, help="'b' parameter for low-dim sigmoid")
@click.option("--grid-width", type=float, help="Grid width for global optimization")
@click.option("--coarse-pts", type=int, default=21, help="Number of coarse grid points")
@click.option("--fine-pts", type=int, default=201, help="Number of fine grid points")
@click.option("--center/--no-center", default=True, help="Center the points")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def dimred(
    highlandmarks_filepath: str,
    high_dimension: int,
    low_dimension: int = 3,
    output_filepath: str = "low_landmarks.dat",
    period: float = 0.0,
    weighted: bool = False,
    dot: bool = False,
    preopt_steps: int = 100,
    gopt_steps: int = 3,
    imix: float = 1.0,
    metric: Literal["euclidean", "dot", "pbc", "sphere"] = "euclidean",
    sigma: Optional[float] = None,
    a_hd: Optional[int] = None,
    b_hd: Optional[int] = None,
    a_ld: Optional[int] = None,
    b_ld: Optional[int] = None,
    grid_width: Optional[float] = None,
    coarse_pts: int = 21,
    fine_pts: int = 201,
    center: bool = True,
    verbose: bool = True,
):
    click.echo("Running dimred with the following parameters:")
    click.echo(f"  highlandmarks_filepath: {highlandmarks_filepath}")
    click.echo(f"  high_dimension: {high_dimension}")
    click.echo(f"  low_dimension: {low_dimension}")
    click.echo(f"  output_filepath: {output_filepath}")
    click.echo(f"  period: {period}")
    click.echo(f"  metric: {metric}")
    click.echo(f"  weighted: {weighted}")
    click.echo(f"  preopt_steps: {preopt_steps}")
    click.echo(f"  gopt_steps: {gopt_steps}")
    click.echo(f"  imix: {imix}")
    click.echo(f"  grid_width: {grid_width}")
    click.echo(f"  coarse_pts: {coarse_pts}")
    click.echo(f"  fine_pts: {fine_pts}")
    click.echo(f"  center: {center}")
    click.echo(f"  verbose: {verbose}")

    data = np.loadtxt(highlandmarks_filepath)

    if np.any(np.isinf(data)) or np.any(np.isnan(data)):
        logger.warning("Input contains inf/NaN values - cleaning data")
        data = np.nan_to_num(data)

    if weighted:
        points = data[:, :-1]
        weights = data[:, -1]
    else:
        points = data
        weights = None

    if points.shape[1] != high_dimension:
        raise ValueError(f"Expected {high_dimension} dimensions, got {points.shape[1]}")

    metric = "dot" if dot else "euclidean"
    if dot and (period != 0.0):
        raise ValueError("Cannot use periodic options with dot product distance")

    if None in [sigma, a_hd, b_hd, a_ld, b_ld]:
        logger.info("Auto-selecting sigmoid parameters")
        sigma = sigma or 13.0
        a_hd = a_hd or 4
        b_hd = b_hd or 2
        a_ld = a_ld or 2
        b_ld = b_ld or 2

    reducer = DimRed(
        high_dim=high_dimension,
        low_dim=low_dimension,
        metric=metric,
        period=period,
        center=center,
        verbose=verbose,
        grid_width=grid_width or 1.5,
        coarse_points=coarse_pts,
        fine_points=fine_pts,
    )

    reducer.set_transformation("high", "sigmoid", (sigma, a_hd, b_hd))
    reducer.set_transformation("low", "sigmoid", (sigma, a_ld, b_ld))

    try:
        logger.info("Running initial MDS")
        init_points = reducer.fit(
            X=points,
            weights=weights,
            preopt_steps=preopt_steps,
            gopt_steps=gopt_steps,
            imix=imix,
            auto_grid=False,
        )

        # iterative refinement
        current_imix = imix if imix > 0 else 0.5  # start with 0.5 if not specified
        for iteration in range(max(1, gopt_steps)):
            logger.info(f"Iteration {iteration + 1}, imix={current_imix:.2f}")

            low_dim_points = reducer.fit(
                X=points,
                weights=weights,
                init=init_points,
                preopt_steps=preopt_steps,
                gopt_steps=1 if grid_width else 0,
                imix=current_imix,
                auto_grid=True,
            )

            # update imix adaptively
            if iteration < gopt_steps - 1:
                current_imix *= 0.8  # gradually reduce mixing parameter
                current_imix = max(current_imix, 0.1)  # keep >= 0.1
            init_points = low_dim_points

        if weighted:
            output_data = np.hstack((low_dim_points, weights.reshape(-1, 1)))
        else:
            output_data = np.hstack(
                (low_dim_points, np.ones(len(low_dim_points)).reshape(-1, 1))
            )
        np.savetxt(output_filepath, output_data)

        logger.info(f"Results saved to {output_filepath}")

    except Exception as e:
        logger.error(f"Error during dimensionality reduction: {e}")
        raise click.Abort()


@main.command()
def project():
    pass


if __name__ == "__main__":
    main()
