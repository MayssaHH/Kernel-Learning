import torch
import torch.nn as nn

from .base import BaseSubKernel

class PolynomialSubKernel(BaseSubKernel):
    """
    Polynomial kernel: K[i,j] = (x[i] * x[j] + c)^d
    #TODO: make c and d learnable parameters later? 
    """
    def __init__(self, degree: int = 2,c: float = 1.0):
        super().__init__()
        self.c = c
        self.degree = degree
    
    def forward(self, x_col: torch.Tensor) -> torch.Tensor:
        # Vectorized: outer product raised to power
        K = ( torch.outer(x_col, x_col) + self.c) ** self.degree
        return K
