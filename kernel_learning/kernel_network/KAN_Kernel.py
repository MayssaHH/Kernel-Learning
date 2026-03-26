from typing import Optional

import torch
import torch.nn as nn

from kan import KAN  # pykan: https://github.com/KindXiaoming/pykan


# ============================================================================
# KAN-BASED KERNEL NETWORK (Approach 2)
# ============================================================================

class KAN_Kernel_NX(nn.Module):
    """
    Kernel network based on the closure property K(u,v) = f(u) * f(v).

    Architecture:
        1. Map each input vector u ∈ R^p through a KAN: ψ(u) ∈ R
           ψ(u) = sum_{i=1}^{2p+1} Φ_i( sum_{j=1}^p φ_{i,j}(u^(j)) )
        2. Compute the kernel matrix as the outer product of ψ values:
           K(u, v) = ψ(u) · ψ(v)

    This guarantees the kernel matrix is symmetric and PSD by construction,
    since K = ψ(X) ψ(X)^T is a Gram matrix.
    """

    def __init__(
        self,
        p: int,
        hidden_dims: Optional[list] = None,
        grid: int = 5,
        k: int = 3,
        psd_epsilon: float = 1e-6,
    ):
        """
        Args:
            p: Number of input features.
            hidden_dims: List of hidden layer widths. Defaults to [2p+1] (one hidden layer,
                         following the Kolmogorov-Arnold theorem). Example: [16, 8] gives
                         width=[p, 16, 8, 1].
            grid: Number of grid intervals for the B-spline activations in the KAN.
            k: Spline order for the B-spline activations in the KAN.
            psd_epsilon: Small value added to the diagonal for numerical stability.
        """
        super().__init__()

        self.p = p
        self.grid = grid
        self.k = k
        self.psd_epsilon = psd_epsilon

        if hidden_dims is None:
            hidden_dims = [2 * p + 1]

        # KAN maps raw features u ∈ R^p -> scalar ψ(u) ∈ R
        width = [p] + hidden_dims + [1]
        self.kan = KAN(
            width=width,
            grid=self.grid,
            k=self.k,
        )

    def _psi(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the KAN to each row of X.

        Args:
            X: Input of shape (n, p)

        Returns:
            psi: Tensor of shape (n,) — one scalar per input vector
        """
        return self.kan(X).squeeze(-1)  # (n,)

    def forward(self, X: torch.Tensor, Y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute kernel matrix K(X, Y) = ψ(X) ψ(Y)^T.

        Args:
            X: Input data of shape (n1, p)
            Y: Optional second input of shape (n2, p).
               If None, computes self-kernel K(X, X) — symmetric PSD by construction.
               If provided, computes cross-kernel K(X, Y).

        Returns:
            K: Kernel matrix of shape (n1, n1) or (n1, n2)
        """
        n1, p = X.shape
        assert p == self.p, f"Expected {self.p} features, got {p}"

        psi_X = self._psi(X)  # (n1,)

        if Y is None:
            # Self-kernel: K = ψ(X) ψ(X)^T — PSD by construction
            K = torch.outer(psi_X, psi_X)  # (n1, n1)
            # Small diagonal shift for numerical stability
            K = K + self.psd_epsilon * torch.eye(n1, device=K.device, dtype=K.dtype)
        else:
            assert Y.shape[1] == self.p, f"Expected {self.p} features in Y, got {Y.shape[1]}"
            psi_Y = self._psi(Y)  # (n2,)
            K = torch.outer(psi_X, psi_Y)  # (n1, n2)

        return K

    @torch.no_grad()
    def inference_kernel(self, X, Y=None):
        """
        Compute kernel matrix without tracking gradients (for inference only).

        Compatible with sklearn's custom kernel interface, which calls
        kernel(X, Y) where:
            - During fit: X=X_train, Y=X_train
            - During predict: X=X_test, Y=X_train

        Args:
            X: Input data of shape (n1, p)
            Y: Optional second input of shape (n2, p).

        Returns:
            K: Kernel matrix of shape (n1, n1) or (n1, n2), as numpy array
        """
        import numpy as np
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(Y, np.ndarray):
            Y = torch.from_numpy(Y).float()
        K = self.forward(X, Y)
        return K.numpy()
