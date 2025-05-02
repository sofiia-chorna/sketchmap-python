import numpy as np
import argparse
from scipy.spatial.distance import pdist
from itertools import combinations


class Histogram:
    def __init__(self, bins):
        self.boundaries = bins
        self.bins = np.zeros(len(bins) - 1)
        self.below = 0.0
        self.above = 0.0
        self.ndata = 0.0  # Total weight of all added samples

    def add(self, value, weight=1.0):
        if value < self.boundaries[0]:
            self.below += weight
        elif value >= self.boundaries[-1]:
            self.above += weight
        else:
            idx = np.searchsorted(self.boundaries, value, side='right') - 1
            self.bins[idx] += weight
        self.ndata += weight

    def get_results(self):
        pdf = self.bins / self.ndata if self.ndata > 0 else self.bins
        centers = 0.5 * (self.boundaries[:-1] + self.boundaries[1:])
        widths = self.boundaries[1:] - self.boundaries[:-1]
        return centers, pdf, widths

    def get_outliers(self):
        return self.above / self.ndata, self.below / self.ndata


def read_matrix(filename, dim, weighted=False):
    data, weights = [], []
    with open(filename) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) >= dim:
                data.append(parts[:dim])
                weights.append(parts[dim] if weighted and len(parts) > dim else 1.0)
    return np.array(data), np.array(weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--highdim", required=True, help="Input high-dimensional file")
    parser.add_argument("-d", "--dim", type=int, required=True, help="Dimensionality")
    parser.add_argument("-nbin", type=int, default=100, help="Number of bins")
    parser.add_argument("-maxd", type=float, help="Max distance")
    parser.add_argument("-w", "--weighted", action="store_true", help="Use weights")
    args = parser.parse_args()

    points, weights = read_matrix(args.highdim, args.dim, args.weighted)

    distances = pdist(points, 'euclidean')

    if args.weighted:
        dweights = np.outer(weights, weights)
        dweights = dweights[np.triu_indices(len(weights), k=1)]
    else:
        dweights = np.ones_like(distances)

    distances = np.array(distances)
    dweights = np.array(dweights)
    maxd = args.maxd if args.maxd is not None else distances.max()
    bins = np.linspace(0, maxd, args.nbin + 1)
    hist = Histogram(bins)

    for d, w in zip(distances, dweights):
        hist.add(d, w)

    out_above, out_below = hist.get_outliers()
    print(f"# Fraction outside: {out_above:.0f} - {out_below:.0f}")

    centers, pdf, widths = hist.get_results()
    for c, p, w in zip(centers, pdf, widths):
        print(f"{c:.12e} {p:.12e} {w:.12e}")


if __name__ == "__main__":
    main()
