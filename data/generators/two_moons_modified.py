import math

import torch


def generate_two_moons(
    samples_per_class: int = 100,
    noise_std: float = 0.08,
    radius: float = 1.0,
    shuffle: bool = True,
):
    """
    Generate a classic two-moons dataset.

    Returns:
        X: Tensor of shape (2 * samples_per_class, 2)
        Y: Tensor of shape (2 * samples_per_class,)
           labels are 0 for first moon and 1 for second moon
    """
    theta = torch.rand(samples_per_class) * math.pi

    # First (upper) moon centered around origin.
    moon_a_x = radius * torch.cos(theta)
    moon_a_y = radius * torch.sin(theta)
    moon_a = torch.stack([moon_a_x, moon_a_y], dim=1)

    # Second (lower) moon shifted right and down (sklearn-style layout).
    moon_b_x = radius * (1.0 - torch.cos(theta))
    moon_b_y = radius * (1.0 - torch.sin(theta) - 0.5)
    moon_b = torch.stack([moon_b_x, moon_b_y], dim=1)

    X = torch.cat([moon_a, moon_b], dim=0)
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X, Y = generate_two_moons(samples_per_class=100, noise_std=0.08)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="viridis")
    plt.title("Two Moons Dataset")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()