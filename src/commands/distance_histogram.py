from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit

from src.utils.logger import logger


class DistanceHistogram:
    def __init__(self, n_bins: int = 100, max_distance: Optional[float] = None):
        self.n_bins = n_bins
        self.max_distance = max_distance

        self.bin_edges = None
        self.bin_centers = None
        self.peak_distance = None
        self.gaussian_std = None
        self.uniform_cuttoff = None

    def analyze_distance(self, distances: Union[torch.Tensor, np.ndarray]):
        if isinstance(distances, torch.Tensor):
            distances = distances.cpu().numpy()

        if self.max_distance is None:
            self.max_distance = np.percentile(distances, 99.9)
            logger.info(f"Auto-set max_distance to {self.max_distance:.2f}")

        # create histogram
        self.bin_edges = np.linspace(0, self.max_distance, self.n_bins + 1)
        bin_counts, _ = np.histogram(distances, bins=self.bin_edges, density=True)

        self.prob_density = bin_counts

        right_edges = self.bin_edges[1:]
        left_edges = self.bin_edges[:-1]
        self.bin_centers = 0.5 * (left_edges + right_edges)

        peak_id = np.argmax(bin_counts)
        self.peak_distance = self.bin_centers[peak_id]

        # estimate gaussian shape (on left side from peak)
        left_mask = self.bin_centers < self.peak_distance
        if np.sum(left_mask) > 3:
            # fit a gaussian
            initial_guess = [np.max(bin_counts), self.peak_distance, 1.0]
            optimal_params, _ = curve_fit(
                self._gaussian_func,
                self.bin_centers[left_mask],
                bin_counts[left_mask],
                p0=initial_guess,
            )

            _amplitude, _center, std_dev = optimal_params
            self.gaussian_std = std_dev

        # estimate uniform distibution (right side from peak)
        right_mask = self.bin_centers > self.peak_distance
        right_side = bin_counts[right_mask]

        if len(right_side) > 0:
            # uniforme level : median of the last 1/3 of the right side
            approx_uniform_value = np.median(right_side[-len(right_side) // 3 :])

            # detect the last bin which is significantly above uniforme
            above_uniform = np.where(bin_counts > 2 * approx_uniform_value)[0]

            if len(above_uniform) > 0:
                self.uniform_cuttoff = self.bin_centers[above_uniform[-1]]

    def _gaussian_func(
        self, x: np.ndarray, a: float, mu: float, sigma: float
    ) -> np.ndarray:
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def suggest_sketchmap_params(self, dim: int) -> dict:
        params = {"sigma": None, "a_high": 6, "b_high": 8, "a_low": 1, "b_low": 2}

        # estimate sigma between gaussian range and uniforme range
        if self.gaussian_std is not None and self.uniform_cuttoff is not None:
            gaussian_range = 3 * self.gaussian_std
            average_range = (gaussian_range + self.uniform_cuttoff) / 2

            params["sigma"] = np.clip(
                average_range,
                gaussian_range * 1.5,  # lower bound: 1.5 * stddev
                self.uniform_cuttoff
                * 0.9,  # upper bound: 90% of right tail cuttof distance
            )
        else:
            logger.warning(
                f"Range is not correct: gaussian_std: {self.gaussian_std}, uniform_cuttoff: {self.uniform_cuttoff}"
            )

        # dimention-based parameter scaling
        if dim > 1000:
            params.update(
                {
                    "a_high": 12,  # max suppression of short-distance noise
                    "b_high": 12,  # max suppression of uniform long distances
                    "a_low": 1,  # keep low-D local structure flexible
                    "b_low": 1,  # very gentle long-distance treatment in low-D
                }
            )
        elif dim > 100:
            params.update(
                {
                    "a_high": min(12, 6 + int(np.log10(dim)) * 2),  # log scaling
                    "b_high": min(12, 8 + dim // 25),  # more graduate increase
                    "a_low": 1,
                    "b_low": 1,
                }
            )
        elif dim > 10:
            params.update(
                {"a_high": min(12, 6 + dim // 10), "b_high": min(12, 8 + dim // 10)}
            )

        return params

    def plot_analysis(
        self,
        input_path: str,
        show: bool = True,
        save_path: Optional[str] = None,
        params: Optional[dict] = {},
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            self.bin_centers, self.prob_density, "b-", label="distance distribution"
        )

        if self.peak_distance is not None:
            ax.axvline(
                self.peak_distance,
                color="r",
                linestyle="--",
                label=f"peak distance: {self.peak_distance:.2f}",
            )

        if self.gaussian_std is not None:
            gaussian_range = 3 * self.gaussian_std
            ax.axvspan(
                0,
                gaussian_range,
                color="g",
                alpha=0.1,
                label="gaussian fluctuation range",
            )

        if self.uniform_cuttoff is not None:
            ax.axvspan(
                self.uniform_cuttoff,
                self.bin_edges[-1],
                color="y",
                alpha=0.1,
                label="uniform distribution range",
            )

        if params["sigma"] is not None:
            ax.axvline(
                params["sigma"],
                color="m",
                linestyle="-.",
                label=f'suggested sigma: {params["sigma"]:.2f}',
            )

        ax.set_xlabel("distance")
        ax.set_ylabel("prob density")
        ax.set_title(f"High-dim distance distribution analysis for {input_path}")
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return fig

    def save_analysis_report(
        self, filepath: str, params: dict, input_path: str
    ) -> None:
        with open(filepath, "w") as f:
            f.write("High-dimentional distance distribution analysis report\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"File: {input_path}\n\n")

            f.write(f"- Peak distance: {self.peak_distance:.4f}\n")
            if self.gaussian_std is not None:
                f.write(f"- Estimated gaussian std: {self.gaussian_std:.4f}\n")
                f.write(f"- 3 sigma gaussian range: {3*self.gaussian_std:.4f}\n")
            if self.uniform_cuttoff is not None:
                f.write(
                    f"- Uniform distribution starts at: {self.uniform_cuttoff:.4f}\n"
                )

            f.write("\nSuggested sketch-map parameters:\n")
            f.write(f"- sigma: {params['sigma']:.4f}\n")
            f.write(f"- a (high-dim): {params['a_high']}\n")
            f.write(f"- b (high-dim): {params['b_high']}\n")
            f.write(f"- a (low-dim): {params['a_low']}\n")
            f.write(f"- b (low-dim): {params['b_low']}\n")
