import csv
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch


class DistanceHistogram:
    def __init__(self, bin_endges: np.ndarray):
        self.bin_endges = bin_endges
        self.bin_counts = np.zeros(len(bin_endges) - 1)
        self.below_range_count = 0.0
        self.above_range_count = 0.0
        self.total_weight = 0.0

    def add_data_points(
        self,
        distances: Union[np.ndarray, torch.Tensor],
        weights: Union[np.ndarray, torch.Tensor],
    ):
        if torch.is_tensor(distances):
            distances = distances.detach().cpu().numpy()
        if torch.is_tensor(weights):
            weights = weights.detach().cpu().numpy()

        if weights is None:
            weights = torch.ones_like(distances)

        min_edge, max_edge = self.bin_endges[0], self.bin_endges[-1]
        in_range_mask = (distances >= min_edge) & (distances < max_edge)

        # count outliers
        self.below_range_count += np.sum(weights[distances < min_edge])
        self.above_range_count += np.sum(weights[distances >= max_edge])

        # bin valid distances
        if np.any(in_range_mask):
            valid_distances = distances[in_range_mask]
            valid_weights = weights[in_range_mask]

            indices = (
                np.searchsorted(self.bin_endges, valid_distances, side="right") - 1
            )
            np.add.at(self.bin_counts, indices, valid_weights)

        self.total_weight += np.sum(weights)

    def get_outliers(self) -> tuple[float, float]:
        return (
            self.above_range_count / self.total_weight,
            self.below_range_count / self.total_weight,
        )

    def get_histogram_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        right_edges = self.bin_endges[1:]
        left_edges = self.bin_endges[:-1]

        bin_widths = right_edges - left_edges

        probabilities = (
            self.bin_counts / (self.total_weight * bin_widths)
            if self.total_weight > 0
            else self.bin_counts
        )

        bin_centers = 0.5 * (left_edges + right_edges)

        return bin_centers, probabilities, bin_widths

    def save_csv(
        self,
        results: tuple[np.ndarray, np.ndarray, np.ndarray],
        filepath: str = "analyse.csv",
    ):
        centers, pdf, widths = results

        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["center", "pdf", "width"])

            for c, p, w in zip(centers, pdf, widths):
                writer.writerow([f"{c:.6e}", f"{p:.6e}", f"{w:.6e}"])

    def save_plot(
        self,
        results: tuple[np.ndarray, np.ndarray, np.ndarray],
        filepath: str = "plot.png",
    ):
        distances, histogram_values, _ = results

        plt.plot(distances, histogram_values, color="red", label="original")

        plt.xlabel("distance")
        plt.ylabel("prob density")
        plt.legend()
        plt.grid()
        plt.xlim(0, 20)

        plt.tight_layout()

        plt.savefig(filepath, dpi=300)
