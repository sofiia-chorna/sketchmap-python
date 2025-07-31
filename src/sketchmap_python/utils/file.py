import os
from typing import List, Optional

import numpy as np
import torch
from sketchmap_python.utils.const import DEVICE
from sketchmap_python.utils.logger import logger


def _adjust_dim(points: List[float], dim: Optional[int]):
    data_dim = points.shape[1]

    if not dim:
        return data_dim
    elif data_dim < dim:
        raise ValueError(f"Requested dim: {dim}, but actual: {data_dim}")
    return dim


def _validate_nan(values: List[float]):
    if torch.isnan(values).any():
        logger.warning("Warning: NaN values found! Replacing them with zero")
        values = torch.nan_to_num(values)

    return values


def read_file(
    filepath: str, dim: Optional[int], weighted=False
) -> tuple[torch.Tensor, torch.Tensor]:
    _, extention = os.path.splitext(filepath)

    if extention == ".pt":
        data = torch.load(filepath, weights_only=False, map_location=DEVICE)

        if isinstance(data, (tuple, list)):
            points = data[0]
            weights = (
                data[1] if len(data) > 1 else torch.ones(len(points), device=DEVICE)
            )
        else:
            points = data
            weights = torch.ones(len(points), device=DEVICE)

        print(type(points))
        print(type(weights))

        points = _validate_nan(points)
        weights = _validate_nan(weights)

        dim = _adjust_dim(points, dim)
        logger.info(f"Used highdim-data dimension: {dim}")

        return points[:, :dim], weights

    else:
        data = np.loadtxt(filepath)
        logger.info(f"Used highdim-data dimension: {dim}")

        points = data[:, :dim]
        weights = (
            data[:, dim] if weighted and dim < data.shape[1] else np.ones(len(data))
        )

        points = torch.tensor(points, device=DEVICE)
        weights = torch.tensor(weights, device=DEVICE)

        points = _validate_nan(points)
        weights = _validate_nan(weights)

        return points, weights
