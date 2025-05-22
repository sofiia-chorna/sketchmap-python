from typing import Literal, Optional, Tuple

import torch

from sklearn.decomposition import PCA

from tqdm import tqdm

from src.commands.distance import DistanceCalculator
from src.utils.const import DEVICE
from src.utils.logger import logger
from src.utils.tensor import to_tensor


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
    ):
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.metric = metric
        self.period = period
        self.center = center
        self.verbose = verbose

        self.grid_width = grid_width

        self.high_dim_transform = self._identity_transform
        self.low_dim_transform = self._identity_transform

        self._first_stress_call = True

        logger.info(
            f"Initialized DimRed with high_dim={high_dim}, low_dim={low_dim}, "
            f"metric={metric}, period={period}, center={center}, verbose={verbose}, device={DEVICE}"
        )

        self.dist_calculator = DistanceCalculator(metric, period)

    def fit(
        self,
        data_points: torch.Tensor,
        initial_embedding: Optional[torch.Tensor],
        weights: Optional[torch.Tensor] = None,
        preoptimization_steps: int = 100,
        global_optimization_steps: int = 0,
        interpolation_mix: float = 0.0,
        learning_rate: float = 0.001,
        auto_grid: bool = True,
    ) -> torch.Tensor:
        """
        Fit the model to the given data points and return the low-dimentional embedding
        """

        logger.info("Starting fit process")

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
            logger.info("Computing initial coordinates using PCA")
            initial_embedding = self.run_pca(data_points)

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

        return optimized_embedding

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
        sigma_t = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        a_t = torch.as_tensor(a, dtype=x.dtype, device=x.device)
        b_t = torch.as_tensor(b, dtype=x.dtype, device=x.device)

        # Scaling factor:  2^{a/b} − 1
        scaling_factor = torch.pow(2.0, a_t / b_t) - 1.0  # S

        # Normalised input:  u = x / σ
        u = x / sigma_t  # u

        # Inner term:  T = 1 + S * u^{a}
        T = 1.0 + scaling_factor * torch.pow(u, a_t)  # T

        # Power used in the outer exponent
        power = -b_t / a_t  # −b/a

        # function value
        S_val = 1.0 - torch.pow(T, power)  # S_{σ,a,b}(x)

        # derivative dS/dx =  (b * S / σ) * u^{a−1} * T^{−b/a − 1}
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

    def run_pca(self, data_points: torch.Tensor) -> torch.Tensor:
        data_points_np = data_points.cpu().numpy()

        pca = PCA(n_components=self.low_dim)
        pca_embedded = pca.fit_transform(data_points_np)

        return to_tensor(pca_embedded)

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
    ) -> torch.Tensor:
        """
        Optimize low-dimentional embedding through two-phase optimization

        Phase 1: local optimization using L-BFGS
        Phase 2: global optimization with random walk
        """

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

        # global optimization
        if num_global_steps > 0:
            current_embedding = self._run_global_optimization(
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

    def _run_global_optimization(
        self,
        embedding: torch.Tensor,
        high_dim_distances: torch.Tensor,
        weights: torch.Tensor,
        mixing_ratio: float,
        learning_rate: float,
        num_steps: int,
        adaptive_grid: bool,
    ) -> torch.Tensor:
        """
        Optimize embedding using global random walk

        Performs stochastic global optimization by :

        1. Identifying points contributing most to stress
        2. Performing directed random walks for these points
        3. Gradually refining the search area
        4. Tracking the best solution found
        """

        num_points, dim = embedding.shape
        best_embedding = embedding.clone()
        best_loss = float("inf")
        stagnation = 0

        # set init search radius
        with torch.no_grad():
            max_point_radius = torch.norm(embedding, dim=1).max().item()
            search_radius = max_point_radius * 1.2 if adaptive_grid else self.grid_width

        # global optilisation loop
        for step in range(num_steps):

            # early stopping if no improvements
            if stagnation >= 5:
                break

            # reduce search intensity over time (= " annealing schedule ")
            temperature = search_radius * (1 - step / num_steps) ** 2
            lr = learning_rate * (1 - step / num_steps)

            problem_indices = self._identify_problem_points(
                embedding, high_dim_distances, weights, mixing_ratio
            )

            self._directed_random_walk(
                embedding,
                high_dim_distances,
                weights,
                mixing_ratio,
                problem_indices,
                temperature,
                lr,
            )

            with torch.no_grad():
                loss = self._calculate_stress(
                    embedding, high_dim_distances, weights, mixing_ratio
                ).item()

                if loss < best_loss:
                    best_loss = loss
                    best_embedding = embedding.clone()

                if abs(loss - best_loss) < 1e-6:
                    stagnation += 1
                else:
                    stagnation = 0

        # final local refinement
        return self._polish_embedding(
            best_embedding,
            high_dim_distances,
            weights,
            mixing_ratio,
            learning_rate * 0.1,
        )

    def _directed_random_walk(
        self,
        embedding: torch.Tensor,
        high_dim_distances: torch.Tensor,
        weights: torch.Tensor,
        mixing_ratio: float,
        indices: torch.Tensor,
        temperature: float,
        lr: float,
    ) -> bool:
        """
        Move selected points in random directions

        For each specified point this method:
        1. Generates a random direction vector
        2. Calculates a step size based on temperature and learning rate
        3. Evaluates both current and proposed new positions
        4. Accepts moves that either improve the solution or meet probabilistic criteria
        """

        if len(indices) == 0:
            return False

        # generate random exploration directions : create random unit vectors for each point to optimize
        directions = torch.randn(
            len(indices), embedding.shape[1], device=embedding.device
        )

        # normalize to unit length
        directions = directions / directions.norm(dim=1, keepdim=True)

        step_sizes = directions * temperature * lr

        # evaluate current positions
        with torch.no_grad():
            current_losses = torch.stack(
                [
                    self._calculate_stress(
                        embedding[i : i + 1],  # single point's embedding
                        high_dim_distances,
                        weights,
                        mixing_ratio,
                    )
                    for i in indices
                ]
            )

        # calculate proposed new positions
        proposed_positions = embedding[indices] + step_sizes

        # evaluate proposed positions
        with torch.no_grad():
            proposed_losses = torch.stack(
                [
                    self._calculate_stress(
                        # create modified embedding with just this point moved
                        torch.cat(
                            [
                                embedding[:i],  # points before current
                                proposed_positions[j : j + 1],  # proposed new position
                                embedding[i + 1 :],  # points after current
                            ]
                        ),
                        high_dim_distances,
                        weights,
                        mixing_ratio,
                    )
                    for j, i in enumerate(
                        indices
                    )  # j indexes proposals, i indexes original points
                ]
            )

        # decide which moves to accepte : determine which moves improved the solution
        improvements = proposed_losses < current_losses

        # calculate acceptance probability for worse moves (simulated annealing)
        acceptance_prob = torch.exp((current_losses - proposed_losses) / temperature)

        # accept either improvements or some worse moves probabilistically
        accept_move = improvements | (
            torch.rand_like(acceptance_prob) < acceptance_prob
        )

        # apply accepted moves
        if accept_move.any():
            # only update positions for accepted moves
            embedding[indices[accept_move]] = proposed_positions[accept_move]
            return True

        return False

    def _identify_problem_points(
        self,
        embedding: torch.Tensor,
        high_dim_distances: torch.Tensor,
        weights: torch.Tensor,
        mixing_ratio: float,
        min_frac: float = 0.2,
    ) -> torch.Tensor:
        """
        Identifies points contributing most to the stress (= poorly embedded points)

        The method :
        1. Calculates stress for each point
        2. Determines a threshold for "problem points" (mean + 0.7*std)
        3. Ensures at least min_frac of points are always considered
        4. Returns ids of problematic points
        """
        with torch.no_grad():
            # calculate per-point stress contributions
            point_stresses = torch.zeros(embedding.size(0), device=embedding.device)

            for i in range(embedding.size(0)):
                # stress when considering only this point's position
                point_stresses[i] = self._calculate_stress(
                    embedding[i : i + 1], high_dim_distances, weights, mixing_ratio
                )

            # determine automatic threshold (mean + 0.7 * std deviation)
            stress_threshold = point_stresses.mean() + 0.7 * point_stresses.std()

            high_stress_points = point_stresses > stress_threshold

            min_points_to_return = max(int(min_frac * embedding.size(0)), 1)

            # if not enough points meet threshold => take top N worst points
            if high_stress_points.sum() < min_points_to_return:
                _, worst_point_indices = torch.topk(
                    point_stresses, k=min_points_to_return
                )
                return worst_point_indices

            # ids of points exceeding threshold
            return high_stress_points.nonzero().view(-1)

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
