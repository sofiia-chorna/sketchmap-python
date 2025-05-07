import os
from typing import List, Optional

import numpy as np
import torch

from src.utils.const import DEVICE


def _adjust_dim(points: List[float], dim: Optional[int]):
    data_dim = points.shape[1]

    if not dim:
        return data_dim
    elif data_dim < dim:
        raise ValueError(f"Requested dim: {dim}, but actual: {data_dim}")
    return dim


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

        if torch.isnan(points).any():
            print("Warning: NaN values found in points!")
            points = torch.nan_to_num(points)

        if torch.isnan(weights).any():
            print("Warning: NaN values found in weights!")
            weights = torch.nan_to_num(weights)

        dim = _adjust_dim(points, dim)
        print(f"Used highdim-data dimension: {dim}")

        return points[:, :dim], weights
    else:
        data = np.loadtxt(filepath)
        print(f"Used highdim-data dimension: {dim}")

        points = data[:, :dim]

        if np.isnan(points).any():
            print("Warning: NaN values found in points!")
            points = np.nan_to_num(points)

        weights = (
            data[:, dim] if weighted and dim < data.shape[1] else np.ones(len(data))
        )

        if np.isnan(weights).any():
            print("Warning: NaN values found in weights!")
            weights = np.nan_to_num(weights)

        points = torch.tensor(points, device=DEVICE)
        weights = torch.tensor(weights, device=DEVICE)

        return points, weights
