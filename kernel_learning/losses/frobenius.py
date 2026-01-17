import torch

from .base import BaseLoss

class FrobeniusLoss(BaseLoss):
    """
    Frobenius norm loss: ||K_pred - K_true||_F^2
    """
    def forward(self, K_pred: torch.Tensor, K_true: torch.Tensor, **kwargs) -> torch.Tensor:
        diff = K_pred - K_true
        return torch.sum(diff ** 2)
