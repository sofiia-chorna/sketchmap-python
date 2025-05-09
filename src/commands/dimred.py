from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
from src.utils.logger import logger
from src.utils.const import DEVICE


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
            f"metric={metric}, period={period}, center={center}, verbose={verbose}, device={DEVICE}"
        )

    def _identity_function(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, torch.ones_like(x)

    def set_transformation(
        self,
        space: Literal["high", "low"],
        fun_type: Literal["identity", "sigmoid"],
        params: Tuple[float, ...],
    ):
        logger.info(
            f"Setting {space}-dim transformation to {fun_type} with params={params}"
        )
        if fun_type == "sigmoid":
            func = self._identity_function
            if len(params) != 3:
                raise ValueError("Sigmoid function requires 3 parameters (sigma, a, b)")
            func = lambda x: self._sigmoid_function(x, *params)
        else:
            raise ValueError(f"Unknown function type: {fun_type}")

        if space == "high":
            self.tfun_hd = func
        else:
            self.tfun_ld = func

    def _sigmoid_function(self, x, sigma, a, b):
        sigma = torch.tensor(sigma, device=DEVICE)
        a = torch.tensor(a, device=DEVICE)
        b = torch.tensor(b, device=DEVICE)

        y = 1 / (1 + (x / sigma) ** a) ** b
        dy = -a * b * (x / sigma) ** (a - 1) * y ** (1 + 1 / b) / sigma
        return y, dy

    def _to_tensor(self, data, dtype=torch.float32):
        if isinstance(data, torch.Tensor):
            return data.to(device=DEVICE, dtype=dtype)
        elif isinstance(data, np.ndarray) or isinstance(data, list):
            return torch.tensor(data, device=DEVICE, dtype=dtype)
        else:
            raise TypeError(f"Cannot convert {type(data)} to torch.Tensor")

    def _compute_distance_matrix(
        self, X: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logger.info(f"Computing distance matrix with metric='{self.metric}'")

        if self.metric == "euclidean":
            if weights is not None:
                logger.info("Applying weighted Euclidean distance")

                weighted_X = X * torch.sqrt(weights.unsqueeze(1))
                return torch.cdist(weighted_X, weighted_X)

            return torch.cdist(X, X)

        elif self.metric == "dot":
            return -torch.matmul(X, X.T)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _classical_mds(self, D: torch.Tensor) -> torch.Tensor:
        logger.info("Performing classical MDS")
        n = D.shape[0]

        # Create centering matrix H
        H = torch.eye(n, device=DEVICE) - torch.ones((n, n), device=DEVICE) / n

        # Double-center the squared distance matrix
        B = -0.5 * H @ (D**2) @ H

        # Eigendecomposition
        vals, vecs = torch.linalg.eigh(B)

        # Sort eigenvalues in descending order and select top k
        idx = torch.argsort(vals, descending=True)[: self.low_dim]

        logger.info(f"Classical MDS eigenvalues: {vals[idx].cpu().numpy()}")

        # Return scaled eigenvectors
        return vecs[:, idx] * torch.sqrt(vals[idx])

    def _stress_function(
        self,
        Y: torch.Tensor,
        D: torch.Tensor,
        weights: torch.Tensor,
        imix: float = 0.0,
    ) -> torch.Tensor:
        # compute pairwise distances in the lowdim space
        d = torch.cdist(Y, Y)

        # apply transformations
        fD, _dfD = self.tfun_hd(D)
        fd, _dfd = self.tfun_ld(d)

        # calculate stress components
        chi_id = torch.sum(weights * (D - d) ** 2)
        chi_fun = torch.sum(weights * (fD - fd) ** 2)
        # chi_id = torch.sum((fD - d) ** 2)
        # chi_fun = torch.sum((fD - fd) ** 2)

        # combine stresses
        stress = imix * chi_id + (1 - imix) * chi_fun

        if self.verbose:
            logger.info(
                f"Stress: chi_id={chi_id.item():.4f}, chi_fun={chi_fun.item():.4f}, total={stress.item():.4f}"
            )

        return stress

    def _optimize_embedding(
        self,
        D: torch.Tensor,
        init: torch.Tensor,
        weights: Optional[torch.Tensor],
        preopt_steps: int,
        gopt_steps: int,
        imix: float,
        grid_params,
        learning_rate: float,
    ) -> torch.Tensor:
        # init with the given starting points
        Y = init.clone().detach().requires_grad_(True)

        # gradient-based optimization
        optimizer = torch.optim.Adam([Y], lr=learning_rate)

        # pre-optimization
        if preopt_steps > 0:
            logger.info(f"Running pre-optimization for {preopt_steps} steps")

            for step in range(preopt_steps):
                optimizer.zero_grad()

                loss = self._stress_function(Y, D, weights, imix)
                loss.backward()

                optimizer.step()

                if self.verbose and (step + 1) % 10 == 0:
                    logger.info(
                        f"Step {step + 1}/{preopt_steps}, Loss: {loss.item():.6f}"
                    )

            logger.info("Pre-optimization completed")

        # global optimization
        if gopt_steps > 0 and grid_params is not None:
            logger.info(f"Running global optimization for {gopt_steps} steps")
            grid_width, coarse_pts, fine_pts = grid_params

            # LBFGS for global optimization
            optimizer = torch.optim.LBFGS([Y], lr=learning_rate)

            # define closure for LBFGS
            def closure():
                optimizer.zero_grad()
                loss = self._stress_function(Y, D, weights, imix)
                loss.backward()
                return loss

            # perform global optimization
            for step in range(gopt_steps):
                optimizer.step(closure)

                if self.verbose and (step + 1) % 10 == 0:
                    loss = self._stress_function(Y, D, weights, imix)
                    logger.info(
                        f"Global opt step {step + 1}/{gopt_steps}, Loss: {loss.item():.6f}"
                    )

            logger.info("Global optimization completed")

        return Y.detach()

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        init: Optional[Union[np.ndarray, torch.Tensor]] = None,
        preopt_steps: int = 100,
        gopt_steps: int = 0,
        imix: float = 0.0,
        grid_params=None,
        learning_rate: float = 0.001,
    ):
        logger.info("Starting fit process")

        X = self._to_tensor(X)
        if weights is not None:
            weights = self._to_tensor(weights)

        if self.center:
            logger.info("Centering the data")
            X = X - X.mean(dim=0, keepdim=True)

        # compute distance matrix
        D = (
            X
            if (self.metric == "dot" and X.shape[0] == X.shape[1])
            else self._compute_distance_matrix(X, weights)
        )

        # init lowdim embedding
        if init is None:
            if self.verbose:
                logger.info("Computing initial coordinates using classical MDS")
            init = self._classical_mds(D)
        else:
            init = self._to_tensor(init)

        if self.verbose:
            logger.info("Beginning optimization")

        result = self._optimize_embedding(
            D, init, weights, preopt_steps, gopt_steps, imix, grid_params, learning_rate
        )

        logger.info("Finished fit process")

        return result.cpu().numpy()
