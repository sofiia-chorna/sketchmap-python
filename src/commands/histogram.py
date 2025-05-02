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
        self, values: tuple[torch.Tensor], weights: tuple[torch.Tensor, torch.Tensor]
    ):
        values = values.cpu().numpy()
        weights = weights.cpu().numpy()

        valid_mask = (values >= self.boundaries[0]) & (values < self.boundaries[-1])
        self.below += np.sum(weights[values < self.boundaries[0]])
        self.above += np.sum(weights[values >= self.boundaries[-1]])

        if np.any(valid_mask):
            valid_values = values[valid_mask]
            valid_weights = weights[valid_mask]
            indices = np.searchsorted(self.boundaries, valid_values, side="right") - 1
            np.add.at(self.bins, indices, valid_weights)

        self.ndata += np.sum(weights)
