import csv

import matplotlib.pyplot as plt
import numpy as np
import torch


class Histogram:
    def __init__(self, bins: np.ndarray):
        self.boundaries = bins
        self.bins = np.zeros(len(bins) - 1)
        self.below = 0.0
        self.above = 0.0
        self.ndata = 0.0

    def add_values(
        self,
        values: tuple[np.ndarray, np.ndarray],
        weights: tuple[np.ndarray, np.ndarray],
    ):
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        if torch.is_tensor(weights):
            weights = weights.detach().cpu().numpy()

        valid_mask = (values >= self.boundaries[0]) & (values < self.boundaries[-1])
        self.below += np.sum(weights[values < self.boundaries[0]])
        self.above += np.sum(weights[values >= self.boundaries[-1]])

        if np.any(valid_mask):
            valid_values = values[valid_mask]
            valid_weights = weights[valid_mask]
            indices = np.searchsorted(self.boundaries, valid_values, side="right") - 1
            np.add.at(self.bins, indices, valid_weights)

        self.ndata += np.sum(weights)

    def get_outliers(self):
        return self.above / self.ndata, self.below / self.ndata

    def get_results(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        first_bin = self.boundaries[1:]
        last_bin = self.boundaries[:-1]

        prob_density_func = self.bins / self.ndata if self.ndata > 0 else self.bins
        centers = 0.5 * (last_bin + first_bin)
        widths = first_bin - last_bin

        return centers, prob_density_func, widths

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
