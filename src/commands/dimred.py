from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch

from tqdm import tqdm

from src.commands.distance import DistanceCalculator
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

        self._first_stress_call = True

        logger.info(
            f"Initialized DimRed with high_dim={high_dim}, low_dim={low_dim}, "
            f"metric={metric}, period={period}, center={center}, verbose={verbose}, device={DEVICE}"
        )

        self.dist_calculator = DistanceCalculator(metric, period)

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
        # Cast the scalar parameters to tensors on the same device / dtype as `x`
        sigma_t = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        a_t = torch.as_tensor(a, dtype=x.dtype, device=x.device)
        b_t = torch.as_tensor(b, dtype=x.dtype, device=x.device)

        # ---- common sub-expressions ------------------------------------------------
        # Scaling factor:  2^{a/b} − 1
        scaling_factor = torch.pow(2.0, a_t / b_t) - 1.0  # S

        # Normalised input:  u = x / σ
        u = x / sigma_t  # u

        # Inner term:  T = 1 + S * u^{a}
        T = 1.0 + scaling_factor * torch.pow(u, a_t)  # T

        # Power used in the outer exponent
        power = -b_t / a_t  # −b/a

        # ---- function value --------------------------------------------------------
        S_val = 1.0 - torch.pow(T, power)  # S_{σ,a,b}(x)

        # ---- derivative ------------------------------------------------------------
        # dS/dx =  (b * S / σ) * u^{a−1} * T^{−b/a − 1}
        dSdx = (
            (b_t * scaling_factor)
            / sigma_t
            * torch.pow(u, a_t - 1.0)
            * torch.pow(T, power - 1.0)  # power-1 == −b/a − 1
        )

        return S_val, dSdx

    def _identity_transform(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identity transformation ("pass-through") function

        Returns the input distances unchanged along with derivatives of 1 (no scaling on gradients during optimization)
        """
        return x, torch.ones_like(x)

    def _compute_distance_matrix(
        self, points: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logger.info(f"Computing distance matrix with metric='{self.metric}'")

        match self.metric:
            case "euclidean":
                if weights is not None:
                    logger.info("Applying weighted Euclidean distance")

                    weighted_points = points * torch.sqrt(weights.unsqueeze(1))

                    return torch.cdist(weighted_points, weighted_points)

                return torch.cdist(points, points)

            case "dot":
                return -torch.matmul(points, points.T)

            case _:
                raise ValueError(f"Unknown metric: {self.metric}")

    def _classical_mds(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """
        Classical multidimensional scaling (MDS) on a distance matrix.
        Returns low-dimensional embedding (shape [n, low_dim])
        """
        distance_matrix = distance_matrix.cpu()

        num_points = distance_matrix.shape[0]

        # centering matrix: subtracts the mean from each row/column
        identity = torch.eye(num_points)
        ones = torch.ones((num_points, num_points)) / num_points
        centering_matrix = identity - ones

        # double-centering the squared distance matrix
        squared_distances = distance_matrix**2
        gram_matrix = -0.5 * centering_matrix @ squared_distances @ centering_matrix

        eigenvalues, eigenvectors = torch.linalg.eigh(gram_matrix)

        # select top "low_dim" eigenvectors that correspont to the largets eigenvalues
        top_ids = torch.argsort(eigenvalues.abs(), descending=True)[: self.low_dim]

        # principal coordinates
        # return eigenvectors[:, top_ids] * torch.sqrt(eigenvalues[top_ids].abs())
        return eigenvectors[:, top_ids]

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
        low_dim_distances = self.dist_calculator.pairwise_distances(
            low_dim_embedding, low_dim_embedding
        )

        if self.verbose and self._first_stress_call:
            d = low_dim_distances.detach().cpu().numpy()
            D = high_dim_distances.detach().cpu().numpy()

            logger.info(
                f"Low-d distances: min={d.min():.4f}, max={d.max():.4f}, mean={d.mean():.4f}"
            )
            logger.info(
                f"High-d distances: min={D.min():.4f}, max={D.max():.4f}, mean={D.mean():.4f}"
            )

            logger.info("Initial low-dimentional distances (d):")
            logger.info(d)

            logger.info("Initial high-dimentional distances (D):")
            logger.info(D)

            self._first_stress_call = False

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
        high_dim_distances: torch.Tensor,
        initial_embedding: torch.Tensor,
        point_weights: Optional[torch.Tensor],
        num_preopt_steps: int,
        num_global_steps: int,
        mixing_ratio: float,
        learning_rate: float,
        adaptive_grid: bool,
        seed: int = 42,
    ) -> torch.Tensor:
        """
        Optimize low-dimentional embedding through two-phase optimization

        Phase 1: local optimization using L-BFGS
        Phase 2: global grid search with adaptive refinement
        """

        torch.manual_seed(seed)

        num_points, embedding_dim = initial_embedding.shape
        current_embedding = (
            initial_embedding.clone().to(DEVICE).detach().requires_grad_(True)
        )

        if point_weights is None:
            point_weights = torch.ones(num_points, device=DEVICE)

        if self.verbose:
            print("=" * 40 + " Optimization configuration " + "=" * 40)
            print("Points:".ljust(20), num_points)
            print("Dimensions:".ljust(20), embedding_dim)
            print("Local steps:".ljust(20), num_preopt_steps)
            print("Global steps:".ljust(20), num_global_steps)
            print("Learning rate:".ljust(20), "%.2e" % learning_rate)
            print("Mixing ratio:".ljust(20), "%.2f" % mixing_ratio)
            print("=" * 85)

        # local optimization
        if num_preopt_steps > 0:
            current_embedding = self._run_local_optimization(
                current_embedding,
                high_dim_distances,
                point_weights,
                mixing_ratio,
                learning_rate,
                num_preopt_steps,
            )

        # global grid search
        if num_global_steps > 0:
            current_embedding = self._run_global_search(
                current_embedding,
                high_dim_distances,
                point_weights,
                mixing_ratio,
                learning_rate,
                num_global_steps,
                adaptive_grid,
            )

        return current_embedding

    def _run_local_optimization(
        self,
        embedding: torch.Tensor,
        high_dim_distances: torch.Tensor,
        weights: torch.Tensor,
        mixing_ratio: float,
        learning_rate: float,
        num_steps: int,
    ) -> torch.Tensor:

        if self.verbose:
            print("\n--- Local optimization ---")
            progress_bar = tqdm(total=num_steps, desc="LBFGS optimization")

        optimizer = torch.optim.LBFGS(
            [embedding],
            lr=learning_rate,
            max_iter=num_steps,
            line_search_fn="strong_wolfe",
            history_size=100,
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = self._calculate_stress(
                embedding, high_dim_distances, weights, mixing_ratio
            )
            loss.backward()

            if self.verbose:
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.6f}"})

            return loss

        optimizer.step(closure)

        if self.verbose:
            progress_bar.close()

            with torch.no_grad():
                final_loss = self._calculate_stress(
                    embedding, high_dim_distances, weights, mixing_ratio
                ).item()
                print(f"Final local loss: {final_loss:.6f}")

        return embedding.detach().requires_grad_(False)

    def _run_global_search(
        self,
        embedding: torch.Tensor,
        high_dim_distances: torch.Tensor,
        weights: torch.Tensor,
        mixing_ratio: float,
        learning_rate: float,
        num_steps: int,
        adaptive_grid: bool,
    ) -> torch.Tensor:

        num_points, embedding_dim = embedding.shape
        identity_matrix = torch.eye(embedding_dim, device=DEVICE)

        # init grid search params
        with torch.no_grad():
            current_radius = torch.max(torch.norm(embedding, dim=1)).item()
            grid_width = current_radius * 1.2 if adaptive_grid else self.grid_width

            if self.verbose:
                print(f"\n--- Global optimization ---")
                print(f"Init grid width: {grid_width:.4f}")
                progress_bar = tqdm(total=num_steps, desc="Global optimization")

        coarse_grid = torch.empty(
            (self.coarse_points, num_points, embedding_dim), device=DEVICE
        )
        fine_grid = torch.empty(
            (self.fine_points, num_points, embedding_dim), device=DEVICE
        )

        last_loss = float("inf")
        stagnation_count = 0

        for step in range(num_steps):
            if stagnation_count >= 5:
                if self.verbose:
                    print(f"\nEarly stopping at step {step} due to stagnation")
                break

            current_grid_width = grid_width * (1.0 - 0.5 * step / num_steps)
            points_to_update = self._identify_problem_points(
                embedding, high_dim_distances, weights, mixing_ratio
            )

            updated = self._process_grid_searches(
                embedding,
                high_dim_distances,
                weights,
                mixing_ratio,
                points_to_update,
                identity_matrix,
                current_grid_width,
                coarse_grid,
                fine_grid,
            )

            # final polishing if last step or no updates
            if step == num_steps - 1 or not updated:
                embedding = self._polish_embedding(
                    embedding, high_dim_distances, weights, mixing_ratio, learning_rate
                )

            stagnation_count, last_loss = self._check_convergence(
                embedding,
                high_dim_distances,
                weights,
                mixing_ratio,
                last_loss,
                stagnation_count,
                progress_bar,
                current_grid_width,
            )

        if self.verbose:
            progress_bar.close()
            print(f"\nOptimization completed. Final loss: {last_loss:.6f}")

        return embedding

    def _process_grid_searches(
        self,
        embedding: torch.Tensor,
        high_dim_distances: torch.Tensor,
        weights: torch.Tensor,
        mixing_ratio: float,
        point_indices: torch.Tensor,
        basis_vectors: torch.Tensor,
        grid_width: float,
        coarse_grid: torch.Tensor,
        fine_grid: torch.Tensor,
    ) -> bool:
        """
        Perform coarse-to-fine grid searches for selected points.
        Returns whether any points were updated
        """
        updated = False

        for point_idx in point_indices:
            original_point = embedding[point_idx].clone()

            for dim in range(embedding.size(1)):
                # coarse grid search
                coarse_offsets = (
                    torch.linspace(
                        -grid_width, grid_width, self.coarse_points, device=DEVICE
                    ).view(-1, 1)
                    * basis_vectors[dim : dim + 1]
                )

                coarse_grid[:] = embedding
                coarse_grid[:, point_idx] += coarse_offsets

                with torch.no_grad():
                    coarse_losses = torch.stack(
                        [
                            self._calculate_stress(
                                coarse_grid[i],
                                high_dim_distances,
                                weights,
                                mixing_ratio,
                            )
                            for i in range(self.coarse_points)
                        ]
                    )

                best_coarse_id = coarse_losses.argmin()
                best_offset = coarse_offsets[best_coarse_id, 0]

                # fine grid search around best coarse result
                fine_start = max(best_offset - grid_width / 10, -grid_width)
                fine_end = min(best_offset + grid_width / 10, grid_width)
                fine_offsets = (
                    torch.linspace(
                        fine_start, fine_end, self.fine_points, device=DEVICE
                    ).view(-1, 1)
                    * basis_vectors[dim : dim + 1]
                )

                fine_grid[:] = embedding
                fine_grid[:, point_idx] += fine_offsets

                with torch.no_grad():
                    fine_losses = torch.stack(
                        [
                            self._calculate_stress(
                                fine_grid[i], high_dim_distances, weights, mixing_ratio
                            )
                            for i in range(self.fine_points)
                        ]
                    )

                best_fine_idx = fine_losses.argmin()
                new_position = original_point + fine_offsets[best_fine_idx]

                # update if significantly improved
                if not torch.allclose(embedding[point_idx], new_position, rtol=1e-6):
                    embedding[point_idx] = new_position
                    updated = True

        return updated

    def _identify_problem_points(
        self,
        embedding: torch.Tensor,
        high_dim_distances: torch.Tensor,
        weights: torch.Tensor,
        mixing_ratio: float,
        min_points_frac: float = 0.2,
    ) -> torch.Tensor:
        """
        Identify points contributing most to the stress for targeted refinement
        """
        with torch.no_grad():
            # calculate per-point stress contributions
            point_stresses = torch.zeros(embedding.size(0), device=DEVICE)

            for i in range(embedding.size(0)):
                point_stresses[i] = self._calculate_stress(
                    embedding[i : i + 1], high_dim_distances, weights, mixing_ratio
                )

            stress_threshold = point_stresses.mean() + 0.7 * point_stresses.std()
            problem_mask = point_stresses > stress_threshold

            min_points = max(int(min_points_frac * embedding.size(0)), 1)

            if problem_mask.sum() < min_points:
                _, problem_indices = torch.topk(point_stresses, k=min_points)
                return problem_indices

            return problem_mask.nonzero().view(-1)

    def _polish_embedding(
        self,
        embedding: torch.Tensor,
        high_dim_distances: torch.Tensor,
        weights: torch.Tensor,
        mixing_ratio: float,
        learning_rate: float,
    ) -> torch.Tensor:
        """Final refinement using L-BFGS optimization"""

        logger.info("Running final L-BFGS polishing step")

        embed = embedding.detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [embed], lr=learning_rate * 0.1, max_iter=50, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = self._calculate_stress(
                embed, high_dim_distances, weights, mixing_ratio
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        return embed.detach().requires_grad_(False)

    def _check_convergence(
        self,
        embedding: torch.Tensor,
        high_dim_distances: torch.Tensor,
        weights: torch.Tensor,
        mixing_ratio: float,
        previous_loss: float,
        stagnation_count: int,
        progress_bar: tqdm,
        current_grid_width: float,
    ) -> Tuple[int, float]:

        with torch.no_grad():
            current_loss = self._calculate_stress(
                embedding, high_dim_distances, weights, mixing_ratio
            ).item()

            if abs(previous_loss - current_loss) < 1e-5:
                stagnation_count += 1
            else:
                stagnation_count = 0

            if self.verbose:
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "loss": f"{current_loss:.6f}",
                        "grid_width": f"{current_grid_width:.4f}",
                        "stagnation": stagnation_count,
                    }
                )

        return stagnation_count, current_loss

    def fit(
        self,
        data_points: Union[np.ndarray, torch.Tensor],
        weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
        initial_embedding: Optional[Union[np.ndarray, torch.Tensor]] = None,
        preoptimization_steps: int = 100,
        global_optimization_steps: int = 0,
        interpolation_mix: float = 0.0,
        learning_rate: float = 0.001,
        auto_grid: bool = True,
    ) -> np.ndarray:
        """
        Fit the model to the given data points and return the low-dimentional embedding
        """

        logger.info("Starting fit process")

        data_points = _to_tensor(data_points)
        if weights is not None:
            weights = _to_tensor(weights)

        if self.center:
            logger.info("Centering the data")
            data_points = data_points - data_points.mean(dim=0, keepdim=True)

        if self.metric == "dot" and data_points.shape[0] == data_points.shape[1]:
            distance_matrix = data_points
        else:
            distance_matrix = self.dist_calculator.pairwise_distances(
                data_points, data_points
            )

        if initial_embedding is None:
            logger.info("Computing initial coordinates using classical MDS")
            initial_embedding = self._classical_mds(distance_matrix)
        else:
            initial_embedding = _to_tensor(initial_embedding)

        logger.info("Beginning optimization")

        optimized_embedding = self._optimize_embedding(
            high_dim_distances=distance_matrix,
            initial_embedding=initial_embedding,
            point_weights=weights,
            num_preopt_steps=preoptimization_steps,
            num_global_steps=global_optimization_steps,
            mixing_ratio=interpolation_mix,
            learning_rate=learning_rate,
            adaptive_grid=auto_grid,
        )

        logger.info("Finished fit process")

        return optimized_embedding.cpu().numpy()


def _to_tensor(data, dtype=torch.float32):

    if isinstance(data, torch.Tensor):
        return data.to(device=DEVICE, dtype=dtype)

    if isinstance(data, np.ndarray) or isinstance(data, list):
        return torch.tensor(data, device=DEVICE, dtype=dtype)

    raise TypeError(f"Cannot convert {type(data)} to torch.Tensor !")
