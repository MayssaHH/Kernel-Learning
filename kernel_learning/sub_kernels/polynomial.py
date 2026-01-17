import torch
import torch.nn as nn

from .base import BaseSubKernel

class PolynomialSubKernel(BaseSubKernel):
    """
    Polynomial kernel: K[i,j] = (sigma^2 * x[i] * x[j] + c)^d
    """
    def __init__(self, degree: int = 2, initial_sigma: float = 1.0, c: float = 1.0):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(initial_sigma)) #NOTE: the nn.Parameter makes sigma a learnable parameter
        self.c = c
        self.degree = degree
    
    def forward(self, x_col: torch.Tensor) -> torch.Tensor:
        # Vectorized: outer product raised to power
        K = (self.sigma ** 2 * torch.outer(x_col, x_col) + self.c) ** self.degree
        return K
