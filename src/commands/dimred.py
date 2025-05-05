from typing import Literal, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from src.utils.logger import logger


class DimRed:
    def __init__(
        self,
        high_dim: int,
        low_dim: int = 2,
        metric: Literal["euclidean", "dot", "pbc"] = "euclidean",
        period: float = 0.0,
        center: bool = True,
        verbose: bool = False,
    ):
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.metric = metric
        self.period = period
        self.center = center
        self.verbose = verbose

        self.tfun_hd = self._identity_function
        self.tfun_ld = self._identity_function

        logger.info(
            f"Initialized DimRed with high_dim={high_dim}, low_dim={low_dim}, "
            f"metric={metric}, period={period}, center={center}, verbose={verbose}"
        )

    def _identity_function(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x, np.ones_like(x)

    def set_transformation(
        self,
        space: Literal["high", "low"],
        fun_type: Literal["identity", "sigmoid", "gamma"],
        params: Tuple[float, ...],
    ):
        logger.info(
            f"Setting {space}-dim transformation to {fun_type} with params={params}"
        )
        if fun_type == "identity":
            func = self._identity_function
        elif fun_type == "sigmoid":
            if len(params) != 3:
                raise ValueError("Sigmoid function requires 3 parameters (sigma, a, b)")
            func = lambda x: self._sigmoid_function(x, *params)
        elif fun_type == "gamma":
            if len(params) != 2:
                raise ValueError("Gamma function requires 2 parameters (sigma, n)")
            func = lambda x: self._gamma_function(x, *params)
        else:
            raise ValueError(f"Unknown function type: {fun_type}")

        if space == "high":
            self.tfun_hd = func
        elif space == "low":
            self.tfun_ld = func
        else:
            raise ValueError("Space must be 'high' or 'low'")

    def _sigmoid_function(self, x, sigma, a, b):
        y = 1 / (1 + (x / sigma) ** a) ** b
        dy = -a * b * (x / sigma) ** (a - 1) * y ** (1 + 1 / b) / sigma
        return y, dy

    def _gamma_function(self, x, sigma, n):
        y = (x / sigma) ** n
        dy = n * (x / sigma) ** (n - 1) / sigma
        return y, dy

    def _compute_distance_matrix(
        self, X: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        logger.info(f"Computing distance matrix with metric='{self.metric}'")

        if self.metric == "euclidean":
            if weights is not None:
                logger.info("Applying weighted Euclidean distance")
                if len(weights) != len(X):
                    raise ValueError("Number of weights must match number of points")
                weighted_X = X * np.sqrt(weights[:, np.newaxis])
                return squareform(pdist(weighted_X, "euclidean"))
            return squareform(pdist(X, "euclidean"))

        elif self.metric == "dot":
            return -np.dot(X, X.T)

        elif self.metric == "pbc":
            if self.period <= 0:
                raise ValueError("Period must be positive for PBC metric")
            logger.warning("Using periodic boundary conditions for distance")
            diff = np.abs(X[:, None] - X)
            diff = np.where(diff > self.period / 2, self.period - diff, diff)
            return np.linalg.norm(diff, axis=-1)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def fit(
        self,
        X,
        weights=None,
        init=None,
        preopt_steps=100,
        gopt_steps=0,
        imix=0.0,
        grid_params=None,
    ):
        logger.info("Starting fit process")

        if self.center:
            logger.info("Centering the data")
            X = X - np.mean(X, axis=0)

        D = (
            X
            if (self.metric == "dot" and X.shape[0] == X.shape[1])
            else self._compute_distance_matrix(X, weights)
        )

        if init is None:
            if self.verbose:
                logger.info("Computing initial coordinates using classical MDS")
            init = self._classical_mds(D)

        if self.verbose:
            logger.info("Beginning optimization")

        result = self._optimize_embedding(
            D, init, weights, preopt_steps, gopt_steps, imix, grid_params
        )

        logger.info("Finished fit process")
        return result

    def _classical_mds(self, D):
        logger.info("Performing classical MDS")
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (D**2) @ H

        vals, vecs = eigh(B)
        idx = np.argsort(vals)[::-1][: self.low_dim]
        if self.verbose:
            logger.info("Classical MDS eigenvalues: %s", vals[idx])
        return vecs[:, idx] * np.sqrt(vals[idx])

    def _stress_function(self, y_flat, D, weights, imix):
        n = D.shape[0]
        Y = y_flat.reshape(n, self.low_dim)
        d = squareform(pdist(Y, "euclidean"))

        fD, dfD = self.tfun_hd(D)
        fd, dfd = self.tfun_ld(d)

        chi_id = np.sum((D - d) ** 2)
        chi_fun = np.sum((fD - fd) ** 2)

        stress = imix * chi_id + (1 - imix) * chi_fun
        if self.verbose:
            logger.info(
                f"Stress: chi_id={chi_id:.4f}, chi_fun={chi_fun:.4f}, total={stress:.4f}"
            )
        return stress

    def _optimize_embedding(
        self, D, init, weights, preopt_steps, gopt_steps, imix, grid_params
    ):
        n = D.shape[0]
        y0 = init.flatten()

        if preopt_steps > 0:
            logger.info(f"Running pre-optimization for {preopt_steps} steps")
            res = minimize(
                fun=self._stress_function,
                x0=y0,
                args=(D, weights, imix),
                method="L-BFGS-B",
                options={"maxiter": preopt_steps, "disp": self.verbose},
            )
            y0 = res.x
            logger.info("Pre-optimization completed")

        if gopt_steps > 0 and grid_params is not None:
            logger.info(f"Running global optimization for {gopt_steps} steps")
            grid_width, coarse_pts, fine_pts = grid_params
            # Placeholder for a real grid search strategy
            res = minimize(
                fun=self._stress_function,
                x0=y0,
                args=(D, weights, imix),
                method="L-BFGS-B",
                options={"maxiter": gopt_steps, "disp": self.verbose},
            )
            y0 = res.x
            logger.info("Global optimization completed")

        return y0.reshape(n, self.low_dim)
