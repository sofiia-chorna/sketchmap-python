import numpy as np
import torch
from sketchmap_python.utils.const import DEVICE


def to_tensor(data, dtype=torch.float32):
    if isinstance(data, torch.Tensor):
        return data.to(device=DEVICE, dtype=dtype)

    if isinstance(data, np.ndarray) or isinstance(data, list):
        return torch.tensor(data, device=DEVICE, dtype=dtype)

    raise TypeError(f"Cannot convert {type(data)} to torch.Tensor !")
