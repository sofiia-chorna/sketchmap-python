from typing import Literal, Optional, Tuple, Callable

import torch

from tqdm import tqdm

from src.commands.distance import DistanceCalculator
from src.commands.init_dimred import run_mds
from src.commands.transform import sigmoid_transform, identity_transform
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
    ):
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.metric = metric
        self.period = period
        self.center = center
        self.verbose = verbose

        self.high_dim_transform = identity_transform
        self.low_dim_transform = identity_transform

        self._first_stress_call = True

        self.dist_calculator = DistanceCalculator(metric, period)

    def fit(
        self,
        data_points: torch.Tensor,
        initial_embedding: Optional[torch.Tensor],
        point_weights: Optional[torch.Tensor] = None,
        local_opt_num_steps: int = 100,
        global_opt_num_steps: int = 0,
        mixing_ratio: float = 0.0,
        learning_rate: float = 1,
    ) -> torch.Tensor:
        """
        Fit the model to the given data points and return the low-dimensional embedding
        """

        logger.info("Starting fit process")

        if self.center:
            logger.info("Centering the data")
            data_points = data_points - data_points.mean(dim=0, keepdim=True)

        distance_matrix = self.dist_calculator.pairwise_distances(
            data_points, data_points
        )

        if initial_embedding is None:
            logger.info("Computing initial coordinates using MDS")
            initial_embedding = run_mds(distance_matrix, self.low_dim)

        logger.info("Beginning optimization")

        num_points, embedding_dim = initial_embedding.shape

        if point_weights is None:
            point_weights = torch.ones((num_points, num_points), device=DEVICE)
            if self.verbose:
                logger.info("Using uniform weights")
        elif self.verbose:
            logger.info(f"Using custom weights. Sample: {point_weights[:10]}")

        if self.verbose:
            logger.info("=" * 20 + " Optimization configuration " + "=" * 20)
            logger.info(f"{'Points:':<20} {num_points}")
            logger.info(f"{'Dimensions:':<20} {embedding_dim}")
            logger.info(f"{'Max LBFGS steps:':<20} {local_opt_num_steps}")
            logger.info(f"{'Max global opt steps:':<20} {global_opt_num_steps}")
            logger.info(f"{'Learning rate:':<20} {learning_rate:.2e}")
            logger.info(f"{'Mixing ratio:':<20} {mixing_ratio:.2f}")
            logger.info("=" * 68)

        def loss_fn(embedding: torch.Tensor) -> torch.Tensor:
            return self._calculate_stress(
                embedding,
                high_dim_distances=distance_matrix,
                weights=point_weights,
                mixing_ratio=mixing_ratio,
            )

        optimized_embedding = self._local_optimize(
            initial_embedding=initial_embedding,
            loss_fn=loss_fn,
            num_steps=local_opt_num_steps,
            learning_rate=learning_rate,
        )

        if global_opt_num_steps > 0:
            logger.info("Starting global optimization")
            optimized_embedding = self._global_optimize(
                initial_embedding=optimized_embedding,
                loss_fn=loss_fn,
                num_steps=global_opt_num_steps,
            )
            final_loss = loss_fn(optimized_embedding).item()
            logger.info(f"Final loss from global optimization: {final_loss:.6f}")

        logger.info("Finished fit process")

        return optimized_embedding

    def _local_optimize(
        self,
        initial_embedding: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        num_steps: int = 100,
        learning_rate: float = 0.001,
    ) -> torch.Tensor:

        embedding = (
            initial_embedding.to(device=DEVICE, dtype=torch.float64)
            .clone()
            .detach()
            .requires_grad_(True)
        )

        optimizer = torch.optim.LBFGS(
            [embedding],
            lr=learning_rate,
            max_iter=num_steps,
            tolerance_grad=1e-6,
            tolerance_change=1e-9,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(embedding)
            loss.backward()

            if self.verbose:
                logger.debug(f"Current loss: {loss.item():.6f}")

            return loss

        logger.info("Starting LBFGS optimization")

        optimizer.step(closure)

        optimized_embedding = embedding.detach()

        if self.verbose:
            final_loss = loss_fn(optimized_embedding)
            logger.info(f"Final loss: {final_loss.item():.6f}")

        return optimized_embedding

    def _global_optimize(
        self,
        initial_embedding: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        num_steps: int = 1000,
        population_size: int = 20,
        learning_rate: float = 1e-2,
    ) -> torch.Tensor:
        """
        Run a global optimization on the embedding using a population-based strategy
        combined with gradient-based refinement with Adam
        """

        # init population with small random perturbations around initial embedding
        noise_scale = 0.1
        population = [
            (
                initial_embedding.detach().clone()
                + noise_scale * torch.randn_like(initial_embedding)
            ).requires_grad_(True)
            for _ in range(population_size)
        ]

        optimizer = torch.optim.Adam(population, lr=learning_rate, betas=(0.95, 0.99))

        best_loss = float("inf")
        patience = 15
        no_improve_steps = 0

        best_embedding = initial_embedding.detach().clone()

        for step in tqdm(range(num_steps)):
            optimizer.zero_grad()

            losses = [loss_fn(individual) for individual in population]

            torch.autograd.backward(losses, [torch.ones_like(l) for l in losses])

            optimizer.step()

            # find best performing member in current population
            current_losses = [l.item() for l in losses]
            min_loss = min(current_losses)
            min_id = current_losses.index(min_loss)

            # update best solution if improvement found
            if min_loss < best_loss:
                best_loss = min_loss
                best_embedding = population[min_id].clone().detach()
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            # early stopping
            if no_improve_steps >= patience:
                if self.verbose:
                    logger.info(f"Early stopping global optimization at step {step}")
                break

            if self.verbose and step % 10 == 0:
                logger.info(
                    f"Global opt step {step}/{num_steps}, best loss: {best_loss:.6f}"
                )

        return best_embedding.detach().clone()

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
                transform_func = lambda x: sigmoid_transform(x, *parameters)

            case "identity":
                transform_func = identity_transform

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

    def _calculate_stress(
        self,
        low_dim_embedding: torch.Tensor,
        high_dim_distances: torch.Tensor,
        weights: torch.Tensor,
        mixing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Calculate the stress between high-dim and low-dim distances

        The stress combines:
        1. direct distance difference (D - d)^2
        2. transformed distance difference (f(D) - f(d))^2

        f() = configured transformation (sigmoid/identity)

        mixing_ratio provides balance between direct and transformed stress (0=all transformed, 1=all direct)
        """
        low_dim_distances = self.dist_calculator.pairwise_distances(
            low_dim_embedding, low_dim_embedding
        )

        transformed_high_dim, _ = self.high_dim_transform(high_dim_distances)
        transformed_low_dim, _ = self.low_dim_transform(low_dim_distances)

        weights_2d = weights.unsqueeze(0) * weights.unsqueeze(1)

        direct_stress = torch.sum(
            weights_2d * (high_dim_distances - low_dim_distances) ** 2
        )
        transformed_stress = torch.sum(
            weights_2d * (transformed_high_dim - transformed_low_dim) ** 2
        )

        combined_stress = (
            mixing_ratio * direct_stress + (1 - mixing_ratio) * transformed_stress
        )

        if self.verbose and self._first_stress_call:
            # log highdim distances
            D = high_dim_distances.detach().cpu().numpy()
            logger.info(f"High-dim distances (sample 3x3):\n{D[:3, :3]}")
            logger.info(
                f"High-dim stats: min={D.min():.4f}, max={D.max():.4f}, mean={D.mean():.4f}"
            )

            # log lowdim distances
            d = low_dim_distances.detach().cpu().numpy()
            logger.info(f"Low-dim distances (sample 3x3):\n{d[:3, :3]}")
            logger.info(
                f"Low-dim stats: min={d.min():.4f}, max={d.max():.4f}, mean={d.mean():.4f}"
            )

            # log transformed distances
            tD = transformed_high_dim.detach().cpu().numpy()
            td = transformed_low_dim.detach().cpu().numpy()
            logger.info(f"Transformed high-dim (sample 3x3):\n{tD[:3, :3]}")
            logger.info(f"Transformed low-dim (sample 3x3):\n{td[:3, :3]}")

            logger.info(
                f"Stress: direct={direct_stress.item():.4f}, "
                f"transformed={transformed_stress.item():.4f},"
                f"combined={combined_stress.item():.4f}, mixing_ratio={mixing_ratio:.2f}"
            )

            self._first_stress_call = False

        normalized_stress = combined_stress / weights_2d.sum()
        # logger.info(f"Normalized stress: {normalized_stress.item():.4f}")

        return normalized_stress
