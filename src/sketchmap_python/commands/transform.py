from typing import Tuple

import torch


def sigmoid_transform(
    x: torch.Tensor, sigma: float, a: float, b: float
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


def identity_transform(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Identity transformation ("pass-through") function

    Returns the input distances unchanged along with derivatives of 1 (no scaling on gradients during optimization)
    """
    return x, torch.ones_like(x)
