import csv
import os
from typing import Optional

import click
import numpy as np

#from src.commands.dimred import DimRed, auto_select_parameters
from src.commands.dimred import DimRed
from src.commands.distance import DistanceCalculator
from src.commands.histogram import Histogram
from src.commands.landmarks import run_get_landmarks
from src.utils.cli import (
    hdim_filepath,
    high_dimension,
    low_dimension,
    max_distance,
    n_bin,
    num,
    output_filepath,
    select_mode,
    weighted,
)
from src.utils.file import read_file
from src.utils.logger import logger
from src.utils.plot import get_analyze_plot


@click.group()
def main():
    pass


@main.command()
@hdim_filepath
@max_distance
@n_bin
@high_dimension
@output_filepath
@weighted
def analyse(
    hdim_filepath: str,
    max_distance: int,
    n_bin: int,
    high_dimension: Optional[int] = None,
    output_filepath: str = "analyse_result.csv",
    weighted: bool = False,
):
    logger.info("Start running 'analyze'")

    logger.info(f"Start reading highdim file: {hdim_filepath}")
    points, weights = read_file(hdim_filepath, high_dimension, weighted)
    logger.info(f"Read {len(points)} points")

    logger.info(f"Start computing distances")

    # TODO: add param to select metric
    distance_calculator = DistanceCalculator("euclidean")

    distances, dweights = distance_calculator.pairwise_distances(
        points, weights, weighted
    )
    logger.info(f"Calculated {len(distances)} distances")

    logger.info(f"Start creating histogram")
    max_distance = max_distance if max_distance is not None else np.max(distances)
    bins = np.linspace(0, max_distance, n_bin + 1)
    hist = Histogram(bins)
    hist.add_values(distances, dweights)

    logger.info(f"Start calculating outliers")
    out_above, out_below = hist.get_outliers()
    logger.info(f"Fraction outside: {out_above:.6f} {out_below:.6f}")

    logger.info(f"Writing results to {output_filepath}")
    centers, prob_density_func, widths = hist.get_results()

    with open(output_filepath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["center", "pdf", "width"])
        for center, pdf_val, width in zip(centers, prob_density_func, widths):
            writer.writerow([f"{center:.6e}", f"{pdf_val:.6e}", f"{width:.6e}"])

    os.makedirs("results", exist_ok=True)
    histo_plot_path = os.path.join("results")
    logger.info(f"Start saving histogram to {histo_plot_path}")
    get_analyze_plot(output_filepath, histo_plot_path)

    logger.info("End running 'analyze'")


@main.command()
@hdim_filepath
@num
@select_mode
@high_dimension
@output_filepath
@weighted
def select_landmarks(
    hdim_filepath: str,
    num: int = 1000,
    select_mode: str = "minmax",
    high_dimension: Optional[int] = None,
    output_filepath: str = "high_landmarks.dat",
    weighted: bool = False,
):
    logger.info("Start running 'select-landmarks'")
    logger.info(f"Params: num={num}, select_mode={select_mode}, weighted={weighted}")

    logger.info(f"Start reading highdim file: {hdim_filepath}")
    points, weights = read_file(hdim_filepath, high_dimension, weighted)
    logger.info(f"Read {len(points)} points")

    logger.info(f"Start selecting high-landmarks")
    highlandmarks, landmark_weights = run_get_landmarks(
        points, weights, weighted, num, select_mode
    )

    if weighted:
        combined = np.hstack((highlandmarks, landmark_weights.reshape(-1, 1)))
    else:
        combined = np.hstack(
            (highlandmarks, np.ones(len(highlandmarks)).reshape(-1, 1))
        )

    logger.info(f"Saving landmarks to {output_filepath}")
    np.savetxt(output_filepath, combined)

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
    "--low-dimension", "-d", type=int, default=2, help="Dimensionality of output space"
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
    "--metric",
    type=click.Choice(["euclidean", "dot", "pbc"]),
    default="euclidean",
    help="Distance metric",
)
@click.option(
    "--weighted/--no-weighted", default=False, help="Use weights from input file"
)
@click.option(
    "--preopt-steps", type=int, default=100, help="Number of pre-optimization steps"
)
@click.option(
    "--gopt-steps", type=int, default=0, help="Number of global optimization steps"
)
@click.option(
    "--imix", type=float, default=0.0, help="Mixing parameter for stress function"
)
@click.option("--grid-width", type=float, help="Grid width for global optimization")
@click.option("--coarse-pts", type=int, help="Number of coarse grid points")
@click.option("--fine-pts", type=int, help="Number of fine grid points")
@click.option("--center/--no-center", default=True, help="Center the points")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def dimred(
    highlandmarks_filepath: str,
    high_dimension: int,
    low_dimension: int = 2,
    output_filepath: str = "low_landmarks.dat",
    period: float = 0.0,
    metric: str = "euclidean",
    weighted: bool = False,
    preopt_steps: int = 100,
    gopt_steps: int = 0,
    imix: float = 0.0,
    grid_width: Optional[float] = None,
    coarse_pts: Optional[int] = None,
    fine_pts: Optional[int] = None,
    center: bool = True,
    verbose: bool = False,
):
    try:
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
            raise ValueError(
                f"Expected {high_dimension} dimensions, got {points.shape[1]}"
            )
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        raise click.Abort()

    #params = auto_select_parameters(points, high_dimension, low_dimension)
    params = {
        "sigma_hd": 7.0,
        "a_hd": 4,
        "b_hd": 2,
        "sigma_ld": 7.0,
        "a_ld": 2,
        "b_ld": 2,
    }

    # set up grid parameters if provided => for now not available
    grid_params = None
    if all(p is not None for p in [grid_width, coarse_pts, fine_pts]):
        grid_params = (grid_width, coarse_pts, fine_pts)

    reducer = DimRed(
        high_dim=high_dimension,
        low_dim=low_dimension,
        metric=metric,
        period=period,
        center=center,
        verbose=verbose,
    )

    reducer.set_transformation(
        "high", "sigmoid", (params["sigma_hd"], params["a_hd"], params["b_hd"])
    )
    reducer.set_transformation(
        "low", "sigmoid", (params["sigma_ld"], params["a_ld"], params["b_ld"])
    )

    try:
        logger.info("Starting dimensionality reduction")
        low_dim_points = reducer.fit(
            X=points,
            weights=weights,
            preopt_steps=preopt_steps,
            gopt_steps=gopt_steps,
            imix=imix,
            grid_params=grid_params,
        )

        if weighted:
            combined = np.hstack((low_dim_points, weights.reshape(-1, 1)))
        else:
            combined = np.hstack(
                (low_dim_points, np.ones(len(low_dim_points)).reshape(-1, 1))
            )

        logger.info(f"Saving results to {output_filepath}")
        np.savetxt(output_filepath, combined)
        logger.info("Dimensionality reduction completed successfully")

    except Exception as e:
        logger.error(f"Error during dimensionality reduction: {e}")
        raise click.Abort()


@main.command()
def project():
    pass


if __name__ == "__main__":
    main()
