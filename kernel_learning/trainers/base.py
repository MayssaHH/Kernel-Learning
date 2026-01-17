from abc import ABC, abstractmethod

import torch

from kernel_learning.kernel_network import KernelNetwork
from kernel_learning.losses import BaseLoss

# ============================================================================
# TRAINERS
# ============================================================================

class BaseTrainer(ABC):
    """
    Base class for training strategies.
    """
    @abstractmethod
    def train_epoch(
        self,
        model: KernelNetwork,
        X: torch.Tensor,
        criterion: BaseLoss,
        lambda_lasso: float = 0.0,
        **loss_kwargs # you can pass anything here depending on the loss function
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            model: KernelNetwork to train
            X: Input data (n, p)
            criterion: Loss function
            lambda_lasso: Lasso regularization strength Keep 0.0 to disable
            **loss_kwargs: Arguments to pass to criterion (e.g., K_true=K_true)
            
        Returns:
            loss: Loss value for this epoch
        """
        pass
