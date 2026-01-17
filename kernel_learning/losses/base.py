import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# ============================================================================
# LOSSES
# ============================================================================

class BaseLoss(nn.Module, ABC):
    """
    Base class for loss functions.
    Losses can have different signatures(params, so for example some might need K_true, others might need y, etc.) depending on what they need. That is why I use **kwargs to allow flexibility. But the first argument is always K_pred, the predicted kernel matrix from the model.
    """
    @abstractmethod # Every subclass of BaseLoss must implement forward with this signature.
    def forward(self, K_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            K_pred: Predicted kernel matrix (n, n)
            **kwargs: Additional arguments (K_true, y, X, etc.) as needed by specific loss
            
        Returns:
            loss: Scalar loss value
        """
        pass
