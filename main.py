from typing import Optional

import click
import numpy as np

from src.commands.analyze import compute_distances

from src.commands.histogram import Histogram
from src.utils.cli import (
    dimension,
    hdim_filepath,
    max_distance,
    n_bin,
    output_filepath,
    weighted,
)
from src.utils.file import read_file
from src.utils.logger import logger


@click.group()
def main():
    pass


@main.command()
@hdim_filepath
@max_distance
@n_bin
@dimension
@output_filepath
@weighted
def analyse(
    hdim_filepath: str,
    max_distance: int,
    n_bin: int,
    dimension: Optional[int] = None,
    output_filepath: Optional[str] = "analyse_result.json",
    weighted: bool = False,
):
    logger.info("Start running 'analyze'")

    logger.info(f"Start reading highdim file: {hdim_filepath}")
    points, weights = read_file(hdim_filepath, dimension, weighted)
    logger.info(f"Read {len(points)} points")

    logger.info(f"Start computing distances")
    distances, dweights = compute_distances(points, weights, weighted)
    logger.info(f"Calculated {len(distances)} distances")

    logger.info(f"Start creating histogram")
    max_distance = max_distance if max_distance is not None else np.max(distances)
    bins = np.linspace(0, max_distance, n_bin + 1)
    hist = Histogram(bins)
    hist.add_values(distances, dweights)

    logger.info(f"Start calculating outliers")
    out_above, out_below = hist.get_outliers()
    logger.info(f"Fraction outside: {out_above:.6f} {out_below:.6f}")

    logger.info("End running 'analyze'")


@main.command()
def select_landmarks():
    pass


@main.command()
def fit_landmarks():
    pass


@main.command()
def project():
    pass


if __name__ == "__main__":
    main()
