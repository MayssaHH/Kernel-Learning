import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    out_dir = Path(__file__).resolve().parent
    x_path = out_dir / "X_5_2.txt"
    y_path = out_dir / "Y_5_2.txt"
    plot_path = out_dir / "plot_5_2.png"

    per_class = 70
    values = np.linspace(-5.0, 5.0, 401)
    rng = np.random.default_rng(0)

    # Choose threshold inside range
    thr = 2.0

    xx, yy = np.meshgrid(values, values, indexing="xy")
    X_all = np.column_stack([xx.ravel(), yy.ravel()])
    y_all = ((X_all[:, 0] > thr) & (X_all[:, 1] > thr)).astype(np.int64)

    idx0 = np.where(y_all == 0)[0]
    idx1 = np.where(y_all == 1)[0]

    pick0 = rng.choice(idx0, size=per_class, replace=False)
    pick1 = rng.choice(idx1, size=per_class, replace=False)

    pick = np.concatenate([pick0, pick1])
    rng.shuffle(pick)

    X = X_all[pick]
    y = y_all[pick]

    np.savetxt(x_path, X, fmt="%.18e", delimiter="\t")
    np.savetxt(y_path, y, fmt="%d")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    mask1 = y == 1

    ax.scatter(X[~mask1, 0], X[~mask1, 1], s=18, label="class 0")
    ax.scatter(X[mask1, 0], X[mask1, 1], s=18, label="class 1")

    ax.set_title("X_5_2: colored by class")
    ax.set_xlabel("X_1")
    ax.set_ylabel("X_2")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()

    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot to {plot_path}")
    print(f"Class counts: 0={(y==0).sum()}, 1={(y==1).sum()}")


if __name__ == "__main__":
    main()
