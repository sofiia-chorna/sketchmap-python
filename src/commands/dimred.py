from typing import Literal, Optional, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import torch
from src.utils.logger import logger
from src.utils.const import DEVICE


def auto_select_parameters(points, high_dim, low_dim=2):
    """Automatically determine optimal sigmoid parameters"""
    # pairwise distances
    distances = pdist(points)
    hist, bins = np.histogram(distances, bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # key distribution features
    # find the first peak (gaussian correlations)
    peak_idx = np.argmax(hist)
    first_peak = bin_centers[peak_idx]

    # find where the histogram drops significantly (high-dim effects)
    q90 = np.percentile(distances, 90)

    # set sigma between the Gaussian peak and high-dim effects
    sigma_hd = np.clip((first_peak + q90) / 2, first_peak * 1.5, q90 * 0.8)

    # high-dim parameters (based on dimension)
    a_hd = max(2, min(6, 2 + np.log10(high_dim)))  # Logarithmic scaling
    b_hd = max(4, min(10, 4 + np.log10(high_dim * 2)))

    # low-dim parameters (more relaxed)
    a_ld = max(1, a_hd * (low_dim / high_dim))
    b_ld = max(1, min(2, b_hd * (low_dim / high_dim) * 2))  # Target 1-2

    params = {
        "sigma_hd": sigma_hd,
        "a_hd": a_hd,
        "b_hd": b_hd,
        "sigma_ld": sigma_hd,
        "a_ld": a_ld,
        "b_ld": b_ld,
    }

    logger.info(f"Params used: {params}")

    return params


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

        if DEVICE == "cuda":
            X_tensor = torch.from_numpy(X).float().to(self.device)

            if self.metric == "euclidean":
                if weights is not None:
                    weights_tensor = torch.from_numpy(weights).float().to(self.device)
                    weighted_X = X_tensor * torch.sqrt(weights_tensor).unsqueeze(1)
                    dists = torch.cdist(weighted_X, weighted_X)
                else:
                    dists = torch.cdist(X_tensor, X_tensor)
                return dists.cpu().numpy()

            elif self.metric == "dot":
                return -torch.matmul(X_tensor, X_tensor.T).cpu().numpy()

            elif self.metric == "pbc":
                if self.period <= 0:
                    raise ValueError("Period must be positive for PBC metric")
                diff = torch.abs(X_tensor.unsqueeze(1) - X_tensor.unsqueeze(0))
                diff = torch.where(diff > self.period / 2, self.period - diff, diff)
                return torch.norm(diff, dim=2).cpu().numpy()

        else:  # Original CPU implementation
            if self.metric == "euclidean":
                if weights is not None:
                    if len(weights) != len(X):
                        raise ValueError(
                            "Number of weights must match number of points"
                        )
                    weighted_X = X * np.sqrt(weights[:, np.newaxis])
                    return squareform(pdist(weighted_X, "euclidean"))
                return squareform(pdist(X, "euclidean"))
            elif self.metric == "dot":
                return -np.dot(X, X.T)
            elif self.metric == "pbc":
                if self.period <= 0:
                    raise ValueError("Period must be positive for PBC metric")
                diff = np.abs(X[:, None] - X)
                diff = np.where(diff > self.period / 2, self.period - diff, diff)
                return np.linalg.norm(diff, axis=-1)

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

        if DEVICE == "cuda":
            D_tensor = torch.from_numpy(D).float().to(self.device)
            n = D_tensor.shape[0]
            H = (
                torch.eye(n, device=self.device)
                - torch.ones((n, n), device=self.device) / n
            )
            B = -0.5 * H @ (D_tensor**2) @ H

            # Using torch.symeig for older versions, or torch.linalg.eigh for newer
            try:
                vals, vecs = torch.linalg.eigh(B)
            except AttributeError:
                vals, vecs = torch.symeig(B, eigenvectors=True)

            idx = torch.argsort(vals, descending=True)[: self.low_dim]
            if self.verbose:
                logger.info(f"Classical MDS eigenvalues: {vals[idx].cpu().numpy()}")
            return (vecs[:, idx] * torch.sqrt(vals[idx])).cpu().numpy()
        else:
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

        if DEVICE == "cuda":
            Y = (
                torch.from_numpy(y_flat.reshape(n, self.low_dim))
                .float()
                .to(self.device)
            )
            D_tensor = torch.from_numpy(D).float().to(self.device)

            # Compute low-dimensional distances
            d = torch.cdist(Y, Y).cpu().numpy()

            # Compute transformed distances
            fD, dfD = self.tfun_hd(D)
            fd, dfd = self.tfun_ld(d)

            chi_id = torch.sum(
                (D_tensor - torch.from_numpy(d).to(self.device)) ** 2
            ).item()
            chi_fun = torch.sum(
                (
                    torch.from_numpy(fD).to(self.device)
                    - torch.from_numpy(fd).to(self.device)
                )
                ** 2
            ).item()
        else:
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

        # Multi-stage optimization
        results = []

        # Stage 1: Coarse optimization
        res = minimize(
            fun=self._stress_function,
            x0=y0,
            args=(D, weights, imix),
            method="L-BFGS-B",
            options={"maxiter": preopt_steps // 2, "disp": self.verbose},
        )
        results.append(res.fun)
        y0 = res.x

        # Stage 2: Refined optimization with momentum
        res = minimize(
            fun=self._stress_function,
            x0=y0,
            args=(D, weights, imix),
            method="CG",  # Conjugate gradient often works better
            options={"maxiter": preopt_steps // 2, "disp": self.verbose},
        )
        results.append(res.fun)

        # Stage 3: Final polish
        if gopt_steps > 0:
            res = minimize(
                fun=self._stress_function,
                x0=res.x,
                args=(D, weights, imix),
                method="L-BFGS-B",
                options={"maxiter": gopt_steps, "gtol": 1e-6, "disp": self.verbose},
            )
            results.append(res.fun)

        if self.verbose:
            logger.info(f"Optimization progression: {results}")

        return res.x.reshape(n, self.low_dim)
