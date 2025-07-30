import os
from typing import Literal, Optional

import click
import numpy as np
import torch

# from src.commands.dimred import DimRed, auto_select_parameters
from src.commands.dimred import DimRed
from src.commands.distance_calculator import DistanceCalculator
from src.commands.distance_histogram import DistanceHistogram
from src.commands.landmarks import (
    get_landmark_stat,
    plot_pca_landmarks,
    run_get_landmarks,
)
from src.commands.normalized_distance_calculator import (
    NormalizedDistanceCalculator,
    compute_normalization_factor,
)
from src.utils.cli import (
    compute_weights,
    hdim_filepath,
    high_dimension,
    low_dimension,
    max_distance,
    metric,
    n_bin,
    normalize,
    num,
    numpy,
    output_filepath,
    period,
    run_check,
    save_indices,
    select_mode,
    sphere_period,
    weighted,
)
from src.utils.file import read_file
from src.utils.logger import RUN_PATH, logger
from src.utils.tensor import to_tensor


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
@normalize
def analyse(
    hdim_filepath: str,
    metric: str,
    period: float,
    sphere_period: float,
    n_bin: int,
    max_distance: Optional[int],
    high_dimension: Optional[int],
    output_filepath: Optional[str],
    weighted: Optional[bool],
    normalize: Optional[bool],
):
    logger.info("Start running 'analyze'")

    params = locals()
    logger.info(f"Parameters: {params}")

    logger.info(f"Start reading highdim file: {hdim_filepath}")
    points, _weights = read_file(hdim_filepath, high_dimension, weighted)
    logger.info(f"Read {len(points)} points")

    logger.info(f"Start computing distances")

    if normalize:
        norm_factor = compute_normalization_factor(points, metric)
        print(
            f"Normalization factor: {norm_factor:.4f} (mean distance = {norm_factor:.2f})"
        )
        distance_calculator = NormalizedDistanceCalculator(
            normalization_factor=norm_factor,
            metric=metric,
            period=period,
            sphere_period=sphere_period,
        )

    else:
        distance_calculator = DistanceCalculator(metric, period, sphere_period)

    distances = distance_calculator.pairwise_distances(points, points)

    logger.info(f"Calculated {len(distances)} distances")
    logger.info(f"Start creating histogram")

    analyzer = DistanceHistogram(n_bin, max_distance)
    analyzer.analyze_distance(distances)

    high_dimension = high_dimension or points.shape[0]
    params = analyzer.suggest_sketchmap_params(dim=high_dimension)

    output_filepath = output_filepath or os.path.join(
        RUN_PATH, "distance_analysis_report.txt"
    )
    plot_filepath = os.path.join(RUN_PATH, "distance_analysis_histogram.png")
    data_filepath = os.path.join(RUN_PATH, "distance_distribution.csv")

    analyzer.plot_analysis(
        save_path=plot_filepath, params=params, input_path=hdim_filepath
    )
    analyzer.save_analysis_report(
        output_filepath, params=params, input_path=hdim_filepath
    )
    analyzer.save_distance_data(data_filepath)

    logger.info(f"Saving histogram to {output_filepath} and plot to {plot_filepath}")

    logger.info("End running 'analyze'")


@main.command()
@hdim_filepath
@num
@select_mode
@high_dimension
@output_filepath
@save_indices
@metric
@period
@sphere_period
@weighted
@compute_weights
@numpy
@run_check
def select_landmarks(
    hdim_filepath: str,
    num: int,
    select_mode: str,
    high_dimension: Optional[int],
    output_filepath: Optional[str],
    save_indices: bool,
    metric: str,
    period: float,
    sphere_period: float,
    weighted: Optional[bool],
    compute_weights: Optional[bool],
    numpy: bool,
    run_check: bool,
):
    logger.info("Start running 'select-landmarks'")

    params = locals()
    logger.info(f"Parameters: {params}")

    logger.info(f"Start reading highdim file: {hdim_filepath}")
    points_np, weights_np = read_file(hdim_filepath, high_dimension, weighted)
    logger.info(f"Read {len(points_np)} points")

    points = points_np.clone().detach()
    weights = weights_np.clone().detach()

    unique_points = torch.unique(points, dim=0)
    logger.info(f"Total points: {len(points)}, unique points: {len(unique_points)}")

    logger.info("Start selecting high-landmarks")

    landmarks, indices, landmark_weights = run_get_landmarks(
        points=points,
        weights=weights,
        num=num,
        mode=select_mode,
        metric=metric,
        compute_weights=compute_weights,
        period=period,
        sphere_period=sphere_period,
    )

    logger.info(
        f"{len(landmarks)} landmark points selected out of {len(points)} and chosen by {select_mode}"
    )

    if compute_weights:
        landmark_weights = landmark_weights.unsqueeze(-1)  # shape: (num, 1)
        combined = torch.cat([landmarks, landmark_weights], dim=-1)
    else:
        combined = landmarks

    if save_indices:
        indices = indices.unsqueeze(-1).float()  # shape: (num, 1)
        combined = torch.cat([indices, combined], dim=-1)  # shape: (num, D+1 or D+2)

    landmarks_filepath = "landmarks.dat" if numpy else "landmarks.pt"
    output_filepath = output_filepath or os.path.join(RUN_PATH, landmarks_filepath)

    logger.info(f"Saving landmarks to {output_filepath}")

    if numpy:
        np.savetxt(output_filepath, combined.cpu().numpy())
    else:
        torch.save(combined, output_filepath)

    if run_check:
        logger.info("Run landmark verification")

        stat_filepath = os.path.join(RUN_PATH, "landmarks_check_report.txt")
        plot_filepath = os.path.join(RUN_PATH, "landmarks_pca_plot.png")

        calculator = DistanceCalculator(metric, period, sphere_period)
        get_landmark_stat(points, landmarks, calculator, stat_filepath)

        plot_pca_landmarks(points, landmarks, plot_filepath)

        logger.info(f"Saving histogram to {stat_filepath} and plot to {plot_filepath}")

    logger.info("Finished landmark selection")


@main.command()
@hdim_filepath
@high_dimension
@low_dimension
@click.option(
    "--init-embedding-filepath",
    type=str,
    help="The low dimention embeddings from which to start optimisation",
)
@output_filepath
@period
@weighted
@click.option(
    "--preopt-steps", type=int, default=100, help="Number of pre-optimization steps"
)
@click.option(
    "--gopt-steps", type=int, default=0, help="Number of global optimization steps"
)
@click.option(
    "--imix", type=float, default=0.0, help="Mixing parameter for stress function"
)
@metric
@click.option(
    "--sigma", type=float, help="Sigma parameter for high-dim and low-dim sigmoid"
)
@click.option("--a-hd", type=int, help="'a' parameter for high-dim sigmoid")
@click.option("--b-hd", type=int, help="'b' parameter for high-dim sigmoid")
@click.option("--a-ld", type=int, help="'a' parameter for low-dim sigmoid")
@click.option("--b-ld", type=int, help="'b' parameter for low-dim sigmoid")
@click.option("--center/--no-center", default=True, help="Center the points")
@click.option("--verbose", "-v", is_flag=True)
def dimred(
    hdim_filepath: str,
    high_dimension: int,
    low_dimension: int = 3,
    init_embedding_filepath: str = None,
    output_filepath: str = "low_dimension.dat",
    period: float = 0.0,
    weighted: bool = False,
    preopt_steps: int = 100,
    gopt_steps: int = 0,
    imix: float = 0.0,
    metric: Literal["euclidean", "dot", "pbc", "sphere"] = "euclidean",
    sigma: Optional[float] = None,
    a_hd: Optional[int] = None,
    b_hd: Optional[int] = None,
    a_ld: Optional[int] = None,
    b_ld: Optional[int] = None,
    center: bool = True,
    verbose: bool = True,
    seed: int = 42,
):
    logger.info("Start running 'dimred'")

    params = locals()
    logger.info(f"Parameters: {params}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # TODO: add support of tensors
    raw_data = np.loadtxt(hdim_filepath)

    if weighted:
        data_points = raw_data[:, :-1]
        point_weights = raw_data[:, -1]
    else:
        data_points = raw_data
        point_weights = None

    data_points = to_tensor(data_points)
    if point_weights is not None:
        point_weights = to_tensor(point_weights)

    if None in [
        sigma,
        a_hd,
        b_hd,
        a_ld,
        b_ld,
    ]:
        # TODO implement autoselection
        sigma = sigma or 7.0
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
    )

    if init_embedding_filepath:
        logger.info(f"Using initial low embeddings from {init_embedding_filepath}")
        initial_embedding = np.loadtxt(init_embedding_filepath)
        initial_embedding = to_tensor(initial_embedding)
    else:
        initial_embedding = None

    try:
        logger.info(f"Refinement with identity function")
        low_dim_embedding = reducer.fit(
            data_points=data_points,
            point_weights=point_weights,
            initial_embedding=initial_embedding,
            local_opt_num_steps=preopt_steps,
            mixing_ratio=imix,
        )

        reducer.set_transformation("high", "sigmoid", (sigma, a_hd, b_hd))
        reducer.set_transformation("low", "sigmoid", (sigma, a_ld, b_ld))

        logger.info(f"Refinement with sigmoid function")

        low_dim_embedding = reducer.fit(
            data_points=data_points,
            point_weights=point_weights,
            initial_embedding=low_dim_embedding,
            local_opt_num_steps=preopt_steps,
            global_opt_num_steps=gopt_steps,
            mixing_ratio=imix,
        )

        low_dim_embedding = low_dim_embedding.cpu().numpy()
        np.savetxt(output_filepath, low_dim_embedding)
        logger.info(f"Saved results to {output_filepath}")

    except Exception as error:
        logger.error(f"Dimensionality reduction failed ! {error}")
        raise click.Abort()


@main.command()
def project():
    pass


if __name__ == "__main__":
    main()
