import math

import torch


def generate_concentric_circles(
    samples_per_class: int = 100,
    inner_radius: float = 0.6,
    outer_radius: float = 1.2,
    noise_std: float = 0.06,
    shuffle: bool = True,
):
    """
    Generate a two-class concentric-circles dataset.

    Returns:
        X: Tensor of shape (2 * samples_per_class, 2)
        Y: Tensor of shape (2 * samples_per_class,)
           labels are 0 for inner circle and 1 for outer circle
    """
    angles_inner = torch.rand(samples_per_class) * (2.0 * math.pi)
    angles_outer = torch.rand(samples_per_class) * (2.0 * math.pi)

    inner = torch.stack(
        [
            inner_radius * torch.cos(angles_inner),
            inner_radius * torch.sin(angles_inner),
        ],
        dim=1,
    )
    outer = torch.stack(
        [
            outer_radius * torch.cos(angles_outer),
            outer_radius * torch.sin(angles_outer),
        ],
        dim=1,
    )

    X = torch.cat([inner, outer], dim=0)
    if noise_std > 0:
        X = X + noise_std * torch.randn_like(X)

    Y = torch.cat(
        [
            torch.zeros(samples_per_class, dtype=torch.long),
            torch.ones(samples_per_class, dtype=torch.long),
        ],
        dim=0,
    )

    if shuffle:
        perm = torch.randperm(X.shape[0])
        X = X[perm]
        Y = Y[perm]

    return X, Y
