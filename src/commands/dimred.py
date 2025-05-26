from typing import Literal, Optional, Tuple

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
        grid_width: float = 1.5,
    ):
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.metric = metric
        self.period = period
        self.center = center
        self.verbose = verbose

        self.grid_width = grid_width

        self.high_dim_transform = identity_transform
        self.low_dim_transform = identity_transform

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
            logger.info("Computing initial coordinates using MDS")
            #    initial_embedding = run_pca(data_points, self.low_dim)
            initial_embedding = run_mds(distance_matrix, self.low_dim)

        logger.info("Beginning optimization")

        optimized_embedding = self._optimize_embedding(
            high_dim_distances=distance_matrix,
            initial_embedding=initial_embedding,
            point_weights=weights,
            num_preopt_steps=preoptimization_steps,
            mixing_ratio=interpolation_mix,
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

        # Low-dim pairwise distances
        low_dim_distances = self.dist_calculator.pairwise_distances(
            low_dim_embedding, low_dim_embedding
        )
        if self.verbose and self._first_stress_call:
            d = low_dim_distances.detach().cpu().numpy()
            logger.info(f"Low-dim distances (sample 3x3):\n{d[:3, :3]}")
            logger.info(
                f"Low-dim stats: min={d.min():.4f}, max={d.max():.4f}, mean={d.mean():.4f}"
            )

        # Log distance transforms
        transformed_high_dim, _ = self.high_dim_transform(high_dim_distances)
        transformed_low_dim, _ = self.low_dim_transform(low_dim_distances)
        if self.verbose and self._first_stress_call:
            tD = transformed_high_dim.detach().cpu().numpy()
            td = transformed_low_dim.detach().cpu().numpy()
            logger.info(f"Transformed high-dim (sample 3x3):\n{tD[:3, :3]}")
            logger.info(f"Transformed low-dim (sample 3x3):\n{td[:3, :3]}")

        # Stress calculations
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
                f"Stress components:\n"
                f"  Direct: {direct_stress.item():.4f}\n"
                f"  Transformed: {transformed_stress.item():.4f}\n"
                f"  Combined (λ={mixing_ratio:.2f}): {combined_stress.item():.4f}"
            )
            self._first_stress_call = False

        return combined_stress

    def _optimize_embedding(
        self,
        high_dim_distances: torch.Tensor,
        initial_embedding: torch.Tensor,
        point_weights: Optional[torch.Tensor],
        num_preopt_steps: int,
        mixing_ratio: float,
        learning_rate: float,
    ) -> torch.Tensor:
        num_points, embedding_dim = initial_embedding.shape

        if point_weights is None:
            point_weights = torch.ones(
                (num_points, num_points), device=DEVICE, dtype=torch.float64
            )
            if self.verbose:
                logger.info(
                    f"Initialized default weights (all 1.0, shape: {point_weights.shape})"
                )
        elif self.verbose:
            logger.info(f"Using custom weights (shape: {point_weights.shape})")

        if self.verbose:
            logger.info("=" * 40 + " Optimization Configuration " + "=" * 40)
            logger.info(f"{'Points:':<20} {num_points}")
            logger.info(f"{'Dimensions:':<20} {embedding_dim}")
            logger.info(f"{'Max LBFGS steps:':<20} {num_preopt_steps}")
            logger.info(f"{'Learning rate:':<20} {learning_rate:.2e}")
            logger.info(f"{'Mixing ratio (λ):':<20} {mixing_ratio:.2f}")
            logger.info("=" * 85)

        embedding = (
            initial_embedding.to(device=DEVICE, dtype=torch.float64)
            .clone()
            .detach()
            .requires_grad_(True)
        )

        optimizer = torch.optim.LBFGS(
            [embedding],
            lr=learning_rate,
            max_iter=num_preopt_steps,
            tolerance_grad=1e-6,
            tolerance_change=1e-9,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = self._calculate_stress(
                embedding, high_dim_distances, point_weights, mixing_ratio
            )
            loss.backward()
            if self.verbose:
                logger.debug(f"Current loss: {loss.item():.6f}")
            return loss

        if self.verbose:
            logger.info("Starting LBFGS optimization")

        optimizer.step(closure)

        optimized_embedding = embedding.detach()

        if self.verbose:
            logger.info(
                f"Optimization finished. Final embedding (sample):\n{optimized_embedding[:3].cpu().numpy()}"
            )
            final_loss = self._calculate_stress(
                optimized_embedding, high_dim_distances, point_weights, mixing_ratio
            ).item()
            logger.info(f"Final loss: {final_loss:.6f}")

        return optimized_embedding
