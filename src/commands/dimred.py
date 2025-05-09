from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch

from tqdm import tqdm

from src.utils.const import DEVICE
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
        grid_width: float = 1.5,
        coarse_points: int = 21,
        fine_points: int = 201,
    ):
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.metric = metric
        self.period = period
        self.center = center
        self.verbose = verbose

        self.grid_width = grid_width
        self.coarse_points = coarse_points
        self.fine_points = fine_points

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

        # create centering matrix H
        H = torch.eye(n, device=DEVICE) - torch.ones((n, n), device=DEVICE) / n

        # double-center the squared distance matrix
        B = -0.5 * H @ (D**2) @ H

        # eigendecomposition
        vals, vecs = torch.linalg.eigh(B)
        idx = torch.argsort(vals, descending=True)[: self.low_dim]

        logger.info(f"Classical MDS eigenvalues: {vals[idx].cpu().numpy()}")

        # scaled eigenvectors
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
        learning_rate: float,
        auto_grid: bool,
    ) -> torch.Tensor:
        device = D.device
        N, d = init.shape
        Y = init.clone().to(device).detach().requires_grad_(True)

        # ======= Step 1: Local Optimization with LBFGS =======
        if preopt_steps > 0:
            optimizer = torch.optim.LBFGS(
                [Y],
                lr=learning_rate,
                max_iter=preopt_steps,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad()
                loss = self._stress_function(Y, D, weights, imix)
                loss.backward()
                return loss

            optimizer.step(closure)

        Y = Y.detach().requires_grad_(False)  # clean up grads after local step

        # ======= Step 2: Global Optimization with Adaptive Grid Search =======
        if gopt_steps > 0:
            eye_d = torch.eye(d, device=device)
            last_loss = float("inf")
            stagnation_counter = 0
            grid_width = self.grid_width

            # Auto-set base width for grid search
            if auto_grid:
                with torch.no_grad():
                    current_radius = torch.max(torch.norm(Y, dim=1)).item()
                    grid_width = current_radius * 1.2

            # Fixed grid parameters matching C++ version (gw,g1,g2 = grid_width,21,201)
            coarse_grid_points = 21
            fine_grid_points = 201

            for global_step in tqdm(range(gopt_steps), desc="Global optimisation"):
                if stagnation_counter >= 5:
                    if self.verbose:
                        print(
                            f"Early stopping global search at step {global_step} due to stagnation."
                        )
                    break

                # Adaptive grid width reduction
                current_grid_width = grid_width * (1.0 - 0.5 * global_step / gopt_steps)

                # Compute per-point errors
                with torch.no_grad():
                    point_errors = torch.stack(
                        [
                            self._stress_function(Y[i : i + 1], D, weights, imix)
                            for i in range(N)
                        ]
                    )

                    # Error threshold calculation (matches C++ version)
                    threshold = point_errors.mean() + 0.7 * point_errors.std()
                    process_points = (point_errors > threshold).nonzero(as_tuple=True)[
                        0
                    ]

                    # Ensure we process at least 20% of points (or 1 point)
                    if len(process_points) < max(int(0.2 * N), 1):
                        k = max(int(0.2 * N), 1)
                        process_points = torch.topk(point_errors, k=k).indices

                updated = False

                for point_idx in tqdm(process_points, desc="Processing error points"):
                    point_idx = int(point_idx)
                    original_point = Y[point_idx].clone()

                    for dim in range(d):
                        # Coarse grid search (21 points)
                        coarse_vals = torch.linspace(
                            -current_grid_width,
                            current_grid_width,
                            coarse_grid_points,
                            device=device,
                        )
                        offsets = coarse_vals.view(-1, 1) * eye_d[dim : dim + 1]
                        test_Y = (
                            Y.unsqueeze(0).expand(coarse_grid_points, -1, -1).clone()
                        )
                        test_Y[:, point_idx] = original_point + offsets

                        with torch.no_grad():
                            losses = torch.stack(
                                [
                                    self._stress_function(y, D, weights, imix)
                                    for y in test_Y
                                ]
                            )

                        best_idx = torch.argmin(losses)
                        best_offset = coarse_vals[best_idx]

                        # Fine grid search (201 points around best coarse point)
                        fine_vals = torch.linspace(
                            max(
                                best_offset - current_grid_width / 10,
                                -current_grid_width,
                            ),
                            min(
                                best_offset + current_grid_width / 10,
                                current_grid_width,
                            ),
                            fine_grid_points,
                            device=device,
                        )
                        offsets = fine_vals.view(-1, 1) * eye_d[dim : dim + 1]
                        test_Y = Y.unsqueeze(0).expand(fine_grid_points, -1, -1).clone()
                        test_Y[:, point_idx] = original_point + offsets

                        with torch.no_grad():
                            losses = torch.stack(
                                [
                                    self._stress_function(y, D, weights, imix)
                                    for y in test_Y
                                ]
                            )

                        best_fine_idx = torch.argmin(losses)
                        new_point = original_point + offsets[best_fine_idx]

                        if not torch.allclose(Y[point_idx], new_point):
                            Y[point_idx] = new_point
                            updated = True

                # Final LBFGS polishing step
                if global_step == gopt_steps - 1 or (not updated):
                    Y = Y.detach().requires_grad_(True)
                    optimizer = torch.optim.LBFGS(
                        [Y],
                        lr=learning_rate * 0.1,
                        max_iter=50,
                        line_search_fn="strong_wolfe",
                    )

                    def closure():
                        optimizer.zero_grad()
                        loss = self._stress_function(Y, D, weights, imix)
                        loss.backward()
                        return loss

                    optimizer.step(closure)
                    Y = Y.detach().requires_grad_(False)

                # Check for stagnation
                with torch.no_grad():
                    current_loss = self._stress_function(Y, D, weights, imix).item()
                    if abs(last_loss - current_loss) < 1e-5:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0
                    last_loss = current_loss

                if self.verbose:
                    print(f"Step {global_step}: loss = {current_loss:.6f}")

        return Y

    def fit(
        self,
        X: Union[np.ndarray, torch.Tensor],
        weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        init: Optional[Union[np.ndarray, torch.Tensor]] = None,
        preopt_steps: int = 100,
        gopt_steps: int = 0,
        imix: float = 0.0,
        learning_rate: float = 0.001,
        auto_grid: bool = True,
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
            logger.info("Computing initial coordinates using classical MDS")
            init = self._classical_mds(D)
        else:
            init = self._to_tensor(init)

        logger.info("Beginning optimization")

        result = self._optimize_embedding(
            D, init, weights, preopt_steps, gopt_steps, imix, learning_rate, auto_grid
        )

        logger.info("Finished fit process")

        return result.cpu().numpy()
