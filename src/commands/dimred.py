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
        num_steps: int = 100,
        mixing_ratio: float = 0.0,
        learning_rate: float = 0.001,
    ) -> torch.Tensor:
        """
        Fit the model to the given data points and return the low-dimentional embedding
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
            logger.info("Using custom weights")

        if self.verbose:
            logger.info("=" * 20 + " Optimization configuration " + "=" * 20)
            logger.info(f"{'Points:':<20} {num_points}")
            logger.info(f"{'Dimensions:':<20} {embedding_dim}")
            logger.info(f"{'Max LBFGS steps:':<20} {num_steps}")
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

        optimized_embedding = self._optimize(
            initial_embedding=initial_embedding,
            loss_fn=loss_fn,
            num_steps=num_steps,
            learning_rate=learning_rate,
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

        It retutrn a combined stress value to minimize:
        1. direct distance difference (D - d)
        2. transformed distance difference (f(D) - f(d))
        """

        if self.verbose and self._first_stress_call:
            D = high_dim_distances.detach().cpu().numpy()
            logger.info(f"High-dim distances (sample 3x3):\n{D[:3, :3]}")
            logger.info(
                f"High-dim stats: min={D.min():.4f}, max={D.max():.4f}, mean={D.mean():.4f}"
            )

        # low-dim pairwise distances
        low_dim_distances = self.dist_calculator.pairwise_distances(
            low_dim_embedding, low_dim_embedding
        )
        if self.verbose and self._first_stress_call:
            d = low_dim_distances.detach().cpu().numpy()
            logger.info(f"Low-dim distances (sample 3x3):\n{d[:3, :3]}")
            logger.info(
                f"Low-dim stats: min={d.min():.4f}, max={d.max():.4f}, mean={d.mean():.4f}"
            )

        transformed_high_dim, _ = self.high_dim_transform(high_dim_distances)
        transformed_low_dim, _ = self.low_dim_transform(low_dim_distances)

        if self.verbose and self._first_stress_call:
            tD = transformed_high_dim.detach().cpu().numpy()
            td = transformed_low_dim.detach().cpu().numpy()
            logger.info(f"Transformed high-dim (sample 3x3):\n{tD[:3, :3]}")
            logger.info(f"Transformed low-dim (sample 3x3):\n{td[:3, :3]}")

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
                f"Stress: direct={direct_stress.item():.4f}, transformed={transformed_stress.item():.4f}, combined={combined_stress.item():.4f}, mixing_ratio={mixing_ratio:.2f}"
            )
            self._first_stress_call = False

        return combined_stress

    def _optimize(
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
