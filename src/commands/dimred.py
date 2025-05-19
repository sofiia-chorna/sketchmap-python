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
        # C++: 1 - (1 + (2^(a/b) - 1)(x/s)^a)^(-b/a)
        # term = 1 + (2.0 ** (a / b) - 1) * (x / sigma) ** a
        # y = 1 - term ** (-b / a)
        # dy = (2.0 ** (a / b) - 1) * (b / sigma) * (x / sigma) ** (a - 1) * term ** (-b / a - 1)

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
        #return vecs[:, idx]

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
        torch.manual_seed(42)

        device = D.device
        N, d = init.shape
        Y = init.clone().to(device).detach().requires_grad_(True)

        if weights is None:
            weights = torch.ones(N, device=device)

        if self.verbose:
            print("\n=== Starting Optimization ===")
            print(f"Points: {N}, Dimensions: {d}")
            print(f"Pre-opt steps: {preopt_steps}, Global steps: {gopt_steps}")
            print(f"Learning rate: {learning_rate}, Mixing: {imix}")

        # ===== Step 1: Local Optimization =====
        if preopt_steps > 0:
            if self.verbose:
                print("\n--- Local Optimization Phase ---")
                pbar = tqdm(total=preopt_steps, desc="LBFGS Pre-optimization")

            optimizer = torch.optim.LBFGS(
                [Y],
                lr=learning_rate,
                max_iter=preopt_steps,
                line_search_fn="strong_wolfe",
                history_size=100,
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = self._stress_function(Y, D, weights, imix)
                loss.backward()
                if self.verbose:
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                return loss

            optimizer.step(closure)
            if self.verbose:
                pbar.close()
                with torch.no_grad():
                    final_loss = self._stress_function(Y, D, weights, imix).item()
                    print(f"Final pre-opt loss: {final_loss:.6f}")

        Y = Y.detach().requires_grad_(False)

        # ===== Step 2: Global Grid Search =====
        if gopt_steps > 0:
            eye_d = torch.eye(d, device=device)
            last_loss = float("inf")
            stagnation_counter = 0

            # Calculate initial grid width
            with torch.no_grad():
                current_radius = torch.max(torch.norm(Y, dim=1)).item()
                grid_width = current_radius * 1.2 if auto_grid else self.grid_width
                if self.verbose:
                    print(f"\n--- Global Optimization Phase ---")
                    print(f"Initial grid width: {grid_width:.4f}")
                    pbar = tqdm(total=gopt_steps, desc="Global optimization")

            # Pre-allocate memory for grid searches
            test_Y_coarse = torch.empty((self.coarse_points, N, d), device=device)
            test_Y_fine = torch.empty((self.fine_points, N, d), device=device)

            for global_step in range(gopt_steps):
                if stagnation_counter >= 5:
                    if self.verbose:
                        print(
                            f"\nEarly stopping at step {global_step} due to stagnation"
                        )
                    break

                current_grid_width = grid_width * (1.0 - 0.5 * global_step / gopt_steps)

                # Compute per-point errors
                with torch.no_grad():
                    point_errors = torch.zeros(N, device=device)
                    for i in range(N):
                        point_errors[i] = self._stress_function(
                            Y[i : i + 1], D, weights, imix
                        )

                    threshold = point_errors.mean() + 0.7 * point_errors.std()
                    mask = point_errors > threshold
                    process_points = mask.nonzero().view(-1)

                    if len(process_points) < max(int(0.2 * N), 1):
                        k = max(int(0.2 * N), 1)
                        _, process_points = torch.topk(point_errors, k=k)

                    if self.verbose and global_step % 5 == 0:
                        pbar.write(
                            f"Step {global_step}: Processing {len(process_points)} points "
                            f"(threshold: {threshold:.4f})"
                        )

                updated = False

                for point_idx in tqdm(
                    process_points.sort().values, desc="Processing points"
                ):
                    point_idx = point_idx.item()
                    original_point = Y[point_idx].clone()

                    for dim in range(d):
                        # Coarse grid search
                        coarse_vals = torch.linspace(
                            -current_grid_width,
                            current_grid_width,
                            self.coarse_points,
                            device=device,
                        )
                        offsets = coarse_vals.view(-1, 1) * eye_d[dim : dim + 1]

                        test_Y_coarse[:] = Y
                        test_Y_coarse[:, point_idx] += offsets

                        with torch.no_grad():
                            losses = torch.stack(
                                [
                                    self._stress_function(
                                        test_Y_coarse[i], D, weights, imix
                                    )
                                    for i in range(self.coarse_points)
                                ]
                            )

                        best_idx = losses.argmin()
                        best_offset = coarse_vals[best_idx]

                        # Fine grid search
                        fine_start = max(
                            best_offset - current_grid_width / 10, -current_grid_width
                        )
                        fine_end = min(
                            best_offset + current_grid_width / 10, current_grid_width
                        )
                        fine_vals = torch.linspace(
                            fine_start, fine_end, self.fine_points, device=device
                        )
                        offsets = fine_vals.view(-1, 1) * eye_d[dim : dim + 1]

                        test_Y_fine[:] = Y
                        test_Y_fine[:, point_idx] += offsets

                        with torch.no_grad():
                            losses = torch.stack(
                                [
                                    self._stress_function(
                                        test_Y_fine[i], D, weights, imix
                                    )
                                    for i in range(self.fine_points)
                                ]
                            )

                        best_fine_idx = losses.argmin()
                        new_point = original_point + offsets[best_fine_idx]

                        if not torch.allclose(Y[point_idx], new_point, rtol=1e-6):
                            Y[point_idx] = new_point
                            updated = True

                # Final polishing
                if global_step == gopt_steps - 1 or not updated:
                    if self.verbose:
                        pbar.write(f"Step {global_step}: Final polishing with LBFGS")

                    Y = Y.detach().requires_grad_(True)
                    optimizer = torch.optim.LBFGS(
                        [Y],
                        lr=learning_rate * 0.1,
                        max_iter=50,
                        line_search_fn="strong_wolfe",
                    )

                    def polish_closure():
                        optimizer.zero_grad(set_to_none=True)
                        loss = self._stress_function(Y, D, weights, imix)
                        loss.backward()
                        return loss

                    optimizer.step(polish_closure)
                    Y = Y.detach().requires_grad_(False)

                # Stagnation check
                with torch.no_grad():
                    current_loss = self._stress_function(Y, D, weights, imix).item()
                    if abs(last_loss - current_loss) < 1e-5:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0
                    last_loss = current_loss

                    if self.verbose:
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "loss": f"{current_loss:.6f}",
                                "grid": f"{current_grid_width:.4f}",
                                "stagnation": stagnation_counter,
                            }
                        )

            if self.verbose:
                pbar.close()
                print(f"\nOptimization completed. Final loss: {current_loss:.6f}")

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
