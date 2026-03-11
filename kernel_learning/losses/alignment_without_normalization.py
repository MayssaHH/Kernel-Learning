import torch

from .base import BaseLoss

class AlignmentLossWithoutNormalization(BaseLoss):
    """
    Kernel alignment loss:
    A(theta) = sum_{i,j} K_pred[i,j] * y[i] * y[j]
    """

    def forward(self, K_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # y: shape (n,)
        # K_pred: shape (n, n)

        # Ensure y is float for multiplication
        y = y.float()

        # Outer product y y^T
        yyT = torch.outer(y, y)  # shape (n, n)

        # Alignment score
        alignment = torch.sum(K_pred * yyT)

        # We maximize alignment, so return negative for minimization
        return -alignment
