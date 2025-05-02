import csv
import os
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
from src.utils.plot import get_analyze_plot


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
    output_filepath: str = "analyse_result.csv",
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
