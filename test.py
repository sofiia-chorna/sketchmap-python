import numpy as np
import argparse
from scipy.spatial.distance import pdist
import torch


class Histogram:
    def __init__(self, bins):
        self.boundaries = bins
        self.bins = np.zeros(len(bins) - 1)
        self.below = 0.0
        self.above = 0.0
        self.ndata = 0.0

    def add_values(self, values, weights):
        # Convert to numpy if they're torch tensors
        if torch.is_tensor(values):
            values = values.cpu().numpy()
        if torch.is_tensor(weights):
            weights = weights.cpu().numpy()

        # Vectorized binning
        valid_mask = (values >= self.boundaries[0]) & (values < self.boundaries[-1])
        self.below += np.sum(weights[values < self.boundaries[0]])
        self.above += np.sum(weights[values >= self.boundaries[-1]])

        if np.any(valid_mask):
            valid_values = values[valid_mask]
            valid_weights = weights[valid_mask]
            indices = np.searchsorted(self.boundaries, valid_values, side="right") - 1
            np.add.at(self.bins, indices, valid_weights)

        self.ndata += np.sum(weights)

    def get_results(self):
        pdf = self.bins / self.ndata if self.ndata > 0 else self.bins
        centers = 0.5 * (self.boundaries[:-1] + self.boundaries[1:])
        widths = self.boundaries[1:] - self.boundaries[:-1]
        return centers, pdf, widths

    def get_outliers(self):
        return self.above / self.ndata, self.below / self.ndata


def read_data(filename, dim, weighted=False, device="cpu"):
    """Read data supporting both .pt files and text files"""
    if filename.endswith(".pt"):
        # Handle PyTorch .pt file
        data = torch.load(filename, map_location=device)
        if isinstance(data, (tuple, list)):
            points = data[0]
            weights = data[1] if len(data) > 1 else torch.ones(len(points))
        else:
            points = data
            weights = torch.ones(len(points))

        if not dim:
            dim = points.shape[1]
        elif points.shape[1] < dim:
            raise ValueError(
                f"Data has {points.shape[1]} dimensions, but requested {dim}"
            )
        else:
            print("Data dim: {points.shape[1]}, requested dim: {dim}")

        points = points[:, :dim]

        return points, weights
    else:
        # Handle text file
        data = np.loadtxt(filename)
        points = data[:, :dim]
        weights = (
            data[:, dim] if weighted and data.shape[1] > dim else np.ones(len(points))
        )

        if torch.cuda.is_available() and device != "cpu":
            return torch.tensor(points, device=device), torch.tensor(
                weights, device=device
            )

        return points, weights


def compute_distances(points, weights, weighted=False, device="cpu"):
    """Compute distances using either numpy or torch depending on input type"""
    if torch.is_tensor(points):
        # Torch implementation
        if device != "cpu" and torch.cuda.is_available():
            points = points.to(device)
            weights = (
                weights.to(device)
                if weighted
                else torch.ones(len(points), device=device)
            )

        # Compute pairwise distances
        dist_matrix = torch.cdist(points, points)

        # Get upper triangular part
        rows, cols = torch.triu_indices(len(points), len(points), offset=1)
        distances = dist_matrix[rows, cols]

        if weighted:
            dweights = torch.outer(weights, weights)[rows, cols]
        else:
            dweights = torch.ones_like(distances)

        return distances.cpu().numpy(), dweights.cpu().numpy()
    else:
        # Numpy implementation
        distances = pdist(points, "euclidean")
        if weighted:
            dweights = np.outer(weights, weights)[np.triu_indices(len(weights), k=1)]
        else:
            dweights = np.ones_like(distances)
        return distances, dweights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-P",
        "--highdim",
        required=True,
        help="Input high-dimensional file (.pt or text)",
    )
    parser.add_argument("-d", "--dim", type=int, help="Dimensionality")
    parser.add_argument("-nbin", type=int, default=100, help="Number of bins")
    parser.add_argument("-maxd", type=float, help="Max distance")
    parser.add_argument("-w", "--weighted", action="store_true", help="Use weights")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    points, weights = read_data(args.highdim, args.dim, args.weighted, device)

    distances, dweights = compute_distances(points, weights, args.weighted, device)

    # Setup histogram
    maxd = args.maxd if args.maxd is not None else np.max(distances)
    bins = np.linspace(0, maxd, args.nbin + 1)
    hist = Histogram(bins)

    # Process all distances at once
    hist.add_values(distances, dweights)

    # Output results
    out_above, out_below = hist.get_outliers()
    print(f"# Fraction outside: {out_above:.6f} {out_below:.6f}")

    centers, pdf, widths = hist.get_results()
    for c, p, w in zip(centers, pdf, widths):
        print(f"{c:.12e} {p:.12e} {w:.12e}")


if __name__ == "__main__":
    main()
