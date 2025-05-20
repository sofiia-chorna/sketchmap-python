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

        self.high_dim_transform = self._identity_transform
        self.low_dim_transform = self._identity_transform

        logger.info(
            f"Initialized DimRed with high_dim={high_dim}, low_dim={low_dim}, "
            f"metric={metric}, period={period}, center={center}, verbose={verbose}, device={DEVICE}"
        )

    def set_transformation(
        self,
        space: Literal["high", "low"],
        transform_type: Literal["identity", "sigmoid"],
        parameters: Tuple[float, ...],
    ) -> None:
        """
        Configure the distance transformation function for either high or low dimensional space.
        """
        logger.info(
            f"Configuring {space}-dim transform: "
            f"type={transform_type}, parameters={parameters}"
        )

        match transform_type:
            case "sigmoid":
                if len(parameters) != 3:
                    raise ValueError(
                        "Sigmoid transform requires exactly 3 parameters: "
                        "(sigma, a, b)"
                    )
                transform_func = lambda x: self._sigmoid_transform(x, *parameters)

            case "identity":
                transform_func = self._identity_transform

            case _:
                raise ValueError(
                    f"Unknown transform type: {transform_type}. "
                    f"Must be 'identity' or 'sigmoid'"
                )

        match space:
            case "high":
                self.high_dim_transform = transform_func
            case "low":
                self.low_dim_transform = transform_func
            case _:
                raise ValueError(f"Invalid space: {space}. Must be 'high' or 'low'")

    def _sigmoid_transform(
        self, x: torch.Tensor, sigma: float, a: float, b: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized sigmoid transformation for distance scaling

        Implements the function:
            y(x) = 1 - [1 + (2^(a/b) - 1)*(x / sigma)^a] ^ (-b / a)

        where:
        - sigma controls location of so called infection point
        - a controls the steepness of the rise
        - b controls the asymptotic behavior
        """
        exponent_ratio = a / b
        scaling_factor = (2.0**exponent_ratio) - 1  # (2^(a/b) - 1)
        normalized_x = x / sigma

        # 1 + scaling_factor * (x / sigma) ^ a
        transformation_term = 1 + scaling_factor * (normalized_x**a)

        # sigmoid function: 1 - term ^ (-b / a)
        transformed_distances = 1 - transformation_term ** (-exponent_ratio)

        # dy/dx = scaling_factor * (b/ sigma) * (x/sigma)^(a-1) * term^(-b/a - 1)
        derivative = (
            scaling_factor
            * (b / sigma)
            * (x ** (a - 1))
            * (normalized_x ** (a - 1))
        )

        return transformed_distances, derivative

    def _identity_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identity transformation ("pass-through") function

        Returns the input distances unchanged along with derivatives of 1 (no scaling on gradients during optimization)
        """
        return x, torch.ones_like(x)

    def _compute_distance_matrix(
        self, X: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logger.info(f"Computing distance matrix with metric='{self.metric}'")

        match self.metric:
            case "euclidean":
                if weights is not None:
                    logger.info("Applying weighted Euclidean distance")

                    weighted_X = X * torch.sqrt(weights.unsqueeze(1))

                    return torch.cdist(weighted_X, weighted_X)

                return torch.cdist(X, X)

            case "dot":
                return -torch.matmul(X, X.T)

            case _:
                raise ValueError(f"Unknown metric: {self.metric}")

    def _classical_mds(self, distance_matrix: torch.Tensor):
        """
        Classical multidimensional scaling (MDS) on a distance matrix.
        Returns low-dimensional embedding (shape [n, low_dim])
        """
        distance_matrix = distance_matrix.cpu()

        num_points = distance_matrix.shape[0]

        # centering matrix: subtracts the mean from each row/column
        H = torch.eye(num_points) - torch.ones((num_points, num_points)) / num_points

        # doubme-centering the squared distance matrix
        # This converts distances to a gram matrix
        B = -0.5 * H @ (distance_matrix**2) @ H

        # eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(B)

        # sort eigenvalues by absolute magnitude and take top "low_dim"
        id = torch.argsort(eigenvalues.abs(), descending=True)[: self.low_dim]

        # principal coordinates
        # return eigenvectors[:, id] * torch.sqrt(eigenvalues[id].abs())
        return eigenvectors[:, id]

    def _calculate_stress(
        self,
        low_dim_embedding: torch.Tensor,
        high_dim_distances: torch.Tensor,
        weights: torch.Tensor,
        mixing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Calculate the stress between high-dim and low-dim distances

        It retutrn a combined stress value to minimize:
        1. direct distance difference (D - d)
        2. transformed distance difference (f(D) - f(d))
        """

        # pairwise distances in lowd
        low_dim_distances = torch.cdist(low_dim_embedding, low_dim_embedding)

        # sigmoid transforms
        transformed_high_dim, _ = self.high_dim_transform(high_dim_distances)
        transformed_low_dim, _ = self.low_dim_transform(low_dim_distances)

        # caclulate both components of the stress function
        direct_stress = torch.sum(
            weights * (high_dim_distances - low_dim_distances) ** 2
        )
        transformed_stress = torch.sum(
            weights * (transformed_high_dim - transformed_low_dim) ** 2
        )

        # combine stresses using the mixing ratio
        # when mixing_ratio = 1 : use only direct distances
        # when mixing_ratio = 0 : use only transformed distances
        combined_stress = (
            mixing_ratio * direct_stress + (1 - mixing_ratio) * transformed_stress
        )

        if self.verbose:
            logger.info(
                f"Stress components - direct: {direct_stress.item():.4f}, "
                f"transformed: {transformed_stress.item():.4f}, "
                f"total: {combined_stress.item():.4f}"
            )

        return combined_stress

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

        N, d = init.shape
        Y = init.clone().to(DEVICE).detach().requires_grad_(True)

        if weights is None:
            weights = torch.ones(N, device=DEVICE)

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
                loss = self._calculate_stress(Y, D, weights, imix)
                loss.backward()
                if self.verbose:
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                return loss

            optimizer.step(closure)
            if self.verbose:
                pbar.close()
                with torch.no_grad():
                    final_loss = self._calculate_stress(Y, D, weights, imix).item()
                    print(f"Final pre-opt loss: {final_loss:.6f}")

        Y = Y.detach().requires_grad_(False)

        # ===== Step 2: Global Grid Search =====
        if gopt_steps > 0:
            eye_d = torch.eye(d, device=DEVICE)
            last_loss = float("inf")
            stagnation_counter = 0

            #  initial grid width
            with torch.no_grad():
                current_radius = torch.max(torch.norm(Y, dim=1)).item()
                grid_width = current_radius * 1.2 if auto_grid else self.grid_width
                if self.verbose:
                    print(f"\n--- Global Optimization Phase ---")
                    print(f"Initial grid width: {grid_width:.4f}")
                    pbar = tqdm(total=gopt_steps, desc="Global optimization")

            test_Y_coarse = torch.empty((self.coarse_points, N, d), device=DEVICE)
            test_Y_fine = torch.empty((self.fine_points, N, d), device=DEVICE)

            for global_step in range(gopt_steps):
                if stagnation_counter >= 5:
                    if self.verbose:
                        print(
                            f"\nEarly stopping at step {global_step} due to stagnation"
                        )
                    break

                current_grid_width = grid_width * (1.0 - 0.5 * global_step / gopt_steps)

                # per-point errors
                with torch.no_grad():
                    point_errors = torch.zeros(N, device=DEVICE)
                    for i in range(N):
                        point_errors[i] = self._calculate_stress(
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
                        # coarse grid search
                        coarse_vals = torch.linspace(
                            -current_grid_width,
                            current_grid_width,
                            self.coarse_points,
                            device=DEVICE,
                        )
                        offsets = coarse_vals.view(-1, 1) * eye_d[dim : dim + 1]

                        test_Y_coarse[:] = Y
                        test_Y_coarse[:, point_idx] += offsets

                        with torch.no_grad():
                            losses = torch.stack(
                                [
                                    self._calculate_stress(
                                        test_Y_coarse[i], D, weights, imix
                                    )
                                    for i in range(self.coarse_points)
                                ]
                            )

                        best_idx = losses.argmin()
                        best_offset = coarse_vals[best_idx]

                        # fine grid search
                        fine_start = max(
                            best_offset - current_grid_width / 10, -current_grid_width
                        )
                        fine_end = min(
                            best_offset + current_grid_width / 10, current_grid_width
                        )
                        fine_vals = torch.linspace(
                            fine_start, fine_end, self.fine_points, device=DEVICE
                        )
                        offsets = fine_vals.view(-1, 1) * eye_d[dim : dim + 1]

                        test_Y_fine[:] = Y
                        test_Y_fine[:, point_idx] += offsets

                        with torch.no_grad():
                            losses = torch.stack(
                                [
                                    self._calculate_stress(
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

                # final polishing
                if global_step == gopt_steps - 1 or not updated:
                    if self.verbose:
                        pbar.write(f"Step {global_step}: final polishing with LBFGS")

                    Y = Y.detach().requires_grad_(True)
                    optimizer = torch.optim.LBFGS(
                        [Y],
                        lr=learning_rate * 0.1,
                        max_iter=50,
                        line_search_fn="strong_wolfe",
                    )

                    def polish_closure():
                        optimizer.zero_grad(set_to_none=True)
                        loss = self._calculate_stress(Y, D, weights, imix)
                        loss.backward()
                        return loss

                    optimizer.step(polish_closure)
                    Y = Y.detach().requires_grad_(False)

                # stagnation check
                with torch.no_grad():
                    current_loss = self._calculate_stress(Y, D, weights, imix).item()
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

        X = _to_tensor(X)
        if weights is not None:
            weights = _to_tensor(weights)

        if self.center:
            logger.info("Centering the data")
            X = X - X.mean(dim=0, keepdim=True)

        # compute distance matrix
        distance_matrix = (
            X
            if (self.metric == "dot" and X.shape[0] == X.shape[1])
            else self._compute_distance_matrix(X, weights)
        )

        # init lowdim embedding
        if init is None:
            logger.info("Computing initial coordinates using classical MDS")
            init = self._classical_mds(distance_matrix)
        else:
            init = _to_tensor(init)

        logger.info("Beginning optimization")

        result = self._optimize_embedding(
            distance_matrix,
            init,
            weights,
            preopt_steps,
            gopt_steps,
            imix,
            learning_rate,
            auto_grid,
        )

        logger.info("Finished fit process")

        return result.cpu().numpy()


def _to_tensor(data, dtype=torch.float32):

    if isinstance(data, torch.Tensor):
        return data.to(device=DEVICE, dtype=dtype)

    if isinstance(data, np.ndarray) or isinstance(data, list):
        return torch.tensor(data, device=DEVICE, dtype=dtype)

    raise TypeError(f"Cannot convert {type(data)} to torch.Tensor !")
