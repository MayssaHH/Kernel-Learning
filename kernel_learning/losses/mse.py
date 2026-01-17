import torch

from .base import BaseLoss

class MSELoss(BaseLoss):
    """
    Mean squared error: mean((K_pred - K_true)^2)
    """
    def forward(self, K_pred: torch.Tensor, K_true: torch.Tensor, **kwargs) -> torch.Tensor:
        diff = K_pred - K_true
        return torch.mean(diff ** 2)
