import math

import torch


def generate_two_outer_circles_with_middle(
    samples_per_class: int = 100,
    center_value: float = 5.0,
    radius: float = 2.0,
    noise_std: float = 0.35,
    middle_std: float = 2.0,
    shuffle: bool = True,
):
    """
    Generate a 2-class 2D dataset:
    - Class 1: union of two noisy circles centered at (-center_value, -center_value)
      and (center_value, center_value)
    - Class 0: points in-between (Gaussian around origin)

    Returns:
        X: Tensor of shape (2 * samples_per_class, 2)
        Y: Tensor of shape (2 * samples_per_class,)
    """
    n_left = samples_per_class // 2
    n_right = samples_per_class - n_left

    theta_left = torch.rand(n_left) * (2.0 * math.pi)
    theta_right = torch.rand(n_right) * (2.0 * math.pi)

    r_left = radius + noise_std * torch.randn(n_left)
    r_right = radius + noise_std * torch.randn(n_right)

    left_circle = torch.stack(
        [
            -center_value + r_left * torch.cos(theta_left),
            -center_value + r_left * torch.sin(theta_left),
        ],
        dim=1,
    )
    right_circle = torch.stack(
        [
            center_value + r_right * torch.cos(theta_right),
            center_value + r_right * torch.sin(theta_right),
        ],
        dim=1,
    )

    class1_outer = torch.cat([left_circle, right_circle], dim=0)
    class0_middle = middle_std * torch.randn(samples_per_class, 2)

    X = torch.cat([class1_outer, class0_middle], dim=0)
    Y = torch.cat(
        [
            torch.ones(samples_per_class, dtype=torch.long),  # outer circles
            torch.zeros(samples_per_class, dtype=torch.long),  # middle region
        ],
        dim=0,
    )

    if shuffle:
        perm = torch.randperm(X.shape[0])
        X = X[perm]
        Y = Y[perm]

    return X, Y
