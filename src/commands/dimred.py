from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.metric = metric
        self.period = period
        self.center = center
        self.verbose = verbose
        self.device = device

        self.tfun_hd = self._identity_function
        self.tfun_ld = self._identity_function

        logger.info(
            f"Initialized DimRedGPU with high_dim={high_dim}, low_dim={low_dim}, "
            f"metric={metric}, period={period}, center={center}, verbose={verbose}, device={device}"
        )

    def _identity_function(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, torch.ones_like(x)

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
        sigma = torch.tensor(sigma, device=self.device)
        a = torch.tensor(a, device=self.device)
        b = torch.tensor(b, device=self.device)

        y = 1 / (1 + (x / sigma) ** a) ** b
        dy = -a * b * (x / sigma) ** (a - 1) * y ** (1 + 1 / b) / sigma
        return y, dy

    def _gamma_function(self, x, sigma, n):
        sigma = torch.tensor(sigma, device=self.device)
        n = torch.tensor(n, device=self.device)

        y = (x / sigma) ** n
        dy = n * (x / sigma) ** (n - 1) / sigma
        return y, dy

    def _compute_distance_matrix(
        self, X: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logger.info(f"Computing distance matrix with metric='{self.metric}'")

        if self.metric == "euclidean":
            if weights is not None:
                logger.info("Applying weighted Euclidean distance")
                if len(weights) != len(X):
                    raise ValueError("Number of weights must match number of points")
                weighted_X = X * torch.sqrt(weights.unsqueeze(1))
                D = torch.cdist(weighted_X, weighted_X)
                return D
            return torch.cdist(X, X)

        elif self.metric == "dot":
            return -torch.matmul(X, X.T)

        elif self.metric == "pbc":
            if self.period <= 0:
                raise ValueError("Period must be positive for PBC metric")
            logger.warning("Using periodic boundary conditions for distance")
            period = torch.tensor(self.period, device=self.device)
            # Broadcasting to compute pairwise differences
            diff = torch.abs(X.unsqueeze(1) - X.unsqueeze(0))
            diff = torch.where(diff > period / 2, period - diff, diff)
            return torch.norm(diff, dim=2)

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _to_tensor(self, data, dtype=torch.float32):
        """Convert numpy array or list to torch tensor on the specified device."""
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=dtype)
        elif isinstance(data, np.ndarray) or isinstance(data, list):
            return torch.tensor(data, device=self.device, dtype=dtype)
        else:
            raise TypeError(f"Cannot convert {type(data)} to torch.Tensor")

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        init: Optional[Union[np.ndarray, torch.Tensor]] = None,
        preopt_steps: int = 100,
        gopt_steps: int = 0,
        imix: float = 0.0,
        grid_params=None,
        learning_rate: float = 0.01,
    ):
        logger.info("Starting fit process on device: " + self.device)

        # Convert to tensors
        X = self._to_tensor(X)
        if weights is not None:
            weights = self._to_tensor(weights)

        n = X.shape[0]

        if self.center:
            logger.info("Centering the data")
            X = X - X.mean(dim=0, keepdim=True)

        # Compute distance matrix
        D = (
            X
            if (self.metric == "dot" and X.shape[0] == X.shape[1])
            else self._compute_distance_matrix(X, weights)
        )

        # Initialize embedding
        if init is None:
            if self.verbose:
                logger.info("Computing initial coordinates using classical MDS")
            init = self._classical_mds(D)
        else:
            init = self._to_tensor(init)

        if self.verbose:
            logger.info("Beginning optimization")

        # Optimize embedding
        result = self._optimize_embedding(
            D, init, weights, preopt_steps, gopt_steps, imix, grid_params, learning_rate
        )

        logger.info("Finished fit process")

        # Return as numpy array for compatibility
        return result.cpu().numpy()

    def _classical_mds(self, D: torch.Tensor) -> torch.Tensor:
        logger.info("Performing classical MDS")
        n = D.shape[0]

        # Create centering matrix H
        H = (
            torch.eye(n, device=self.device)
            - torch.ones((n, n), device=self.device) / n
        )

        # Double-center the squared distance matrix
        B = -0.5 * H @ (D**2) @ H

        # Eigendecomposition
        vals, vecs = torch.linalg.eigh(B)

        # Sort eigenvalues in descending order and select top k
        idx = torch.argsort(vals, descending=True)[: self.low_dim]

        if self.verbose:
            logger.info(f"Classical MDS eigenvalues: {vals[idx].cpu().numpy()}")

        # Return scaled eigenvectors
        return vecs[:, idx] * torch.sqrt(vals[idx])

    def _stress_function(
        self,
        Y: torch.Tensor,
        D: torch.Tensor,
        weights: Optional[torch.Tensor],
        imix: float,
    ) -> torch.Tensor:
        d = torch.cdist(Y, Y)

        # Apply transformations
        fD, dfD = self.tfun_hd(D)
        fd, dfd = self.tfun_ld(d)

        # Calculate stress components
        chi_id = torch.sum((D - d) ** 2)
        chi_fun = torch.sum((fD - fd) ** 2)

        # Combine stresses
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
        """Optimize the embedding using PyTorch's automatic differentiation."""
        n = D.shape[0]

        Y = init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([Y], lr=learning_rate)

        # Pre-optimization
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

        # Global optimization (if requested)
        if gopt_steps > 0 and grid_params is not None:
            logger.info(f"Running global optimization for {gopt_steps} steps")
            grid_width, coarse_pts, fine_pts = grid_params

            optimizer = torch.optim.LBFGS([Y], lr=learning_rate)

            def closure():
                optimizer.zero_grad()
                loss = self._stress_function(Y, D, weights, imix)
                loss.backward()
                return loss

            for step in range(gopt_steps):
                optimizer.step(closure)

                if self.verbose and (step + 1) % 10 == 0:
                    loss = self._stress_function(Y, D, weights, imix)
                    logger.info(
                        f"Global opt step {step + 1}/{gopt_steps}, Loss: {loss.item():.6f}"
                    )

            logger.info("Global optimization completed")

        # Return the optimized embedding (detach to remove from computation graph)
        return Y.detach()
