import torch
from typing import Optional
from .base import BaseSubKernel


class LinearSubKernel(BaseSubKernel):
    """
    Linear kernel: K[i,j] = x[i] * y[j]
    
    Self-kernel:  K = outer(x, x)  -> (n, n)
    Cross-kernel: K = outer(x, y)  -> (n1, n2)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_col: torch.Tensor, y_col: Optional[torch.Tensor] = None) -> torch.Tensor:
        if y_col is None:
            y_col = x_col
        return torch.outer(x_col, y_col)