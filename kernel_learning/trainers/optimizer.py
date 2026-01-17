import torch

from .base import BaseTrainer
from kernel_learning.kernel_network import KernelNetwork
from kernel_learning.losses import BaseLoss

class OptimizerTrainer(BaseTrainer):
    """
    Trainer using PyTorch optimizers (Adam, SGD, etc.).
    """
    def __init__(
        self,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam, # default optimizer class
        lr: float = 0.001, # learning rate
        **optimizer_kwargs # Additional keyword arguments passed directly to the optimizer
                            # (e.g. weight_decay, momentum, betas, etc., depending on the optimizer).
                            # NOTE: These are passed as normal keyword arguments (not as a dict) to make
                            # the API more user-friendly. For example:
                            # OptimizerTrainer(optimizer_class=torch.optim.SGD, lr=0.01, momentum=0.9)
                            # instead of passing a separate optimizer_kwargs dict.
    ):
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.optimizer_kwargs = optimizer_kwargs # store optimizer kwargs as a dictionary
        self.optimizer = None # Will be initialized on first call
    
    def train_epoch(
        self,
        model: KernelNetwork,
        X: torch.Tensor,
        criterion: BaseLoss,
        lambda_lasso: float = 0.0,
        **loss_kwargs
    ) -> float:
        # Initialize optimizer on first call
        if self.optimizer is None:
            self.optimizer = self.optimizer_class(
                model.parameters(),
                lr=self.lr,
                **self.optimizer_kwargs
            )
        
        # Zero gradients
        self.optimizer.zero_grad()  # Because PyTorch accumulates gradients by default. This is conceptually the same as param.grad.zero_() in the manual gradient trainer.
        
        # Forward pass
        K_pred = model(X)
        
        # Compute loss
        loss = criterion(K_pred, **loss_kwargs)
        
        # Add Lasso penalty
        if lambda_lasso > 0:
            loss = loss + model.get_lasso_penalty(lambda_lasso)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.optimizer.step() # Update parameters based on computed gradients
        
        return loss.item()
