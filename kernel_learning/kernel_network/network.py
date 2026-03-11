import warnings
from typing import List, Optional, Union, Dict, Any, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernel_learning.sub_kernels import BaseSubKernel, LinearSubKernel, RBFSubKernel, PolynomialSubKernel

# ============================================================================
# KERNEL NETWORK
# ============================================================================

class KernelNetwork(nn.Module):
    """
    Main kernel network that combines multiple sub-kernels.
    
    K_total = Σ_k α_k * SubKernel_k(X[:, k])
    
    Features:
    - Mix-and-match different sub-kernels #NOTE: meaning each feature can have its own kernel type
    - Learnable alpha weights with various constraints #NOTE: these alphas are the weighted sums of sub-kernels
        -- Optional normalization (sum to 1)
        -- Lasso regularization on alphas
    - Automatic symmetry enforcement #NOTE: Meaning the output kernel matrix K is guaranteed to be symmetric, even if numerical issues arise during computation, by averaging K with its transpose.
    """
    def __init__(
        self,
        sub_kernels: List[BaseSubKernel],
        alpha_constraint: str = 'square',  # 'none', 'exp', 'softplus', 'square', 'abs'
        normalize_alphas: bool = False,
        alpha_init: Union[str, torch.Tensor] = 'random',  # 'random', 'ones', or your own tensor of shape (p,)
        symmetry_tolerance: float = 1e-6 
    ):
        """
        Args:
            - sub_kernels: List of SubKernel instances (length p) #NOTE: you have different ways to create this list, either explicitly(i.e. you pass a list of p sub-kernels) or using the class methods provided below

            - alpha_constraint: How to constrain alphas to be non-negative #NOTE: meaning we have the raw weights but we want to ensure the actual alphas used in the weighted sum are non-negative, so we apply a transformation to the raw weights. Options:
                'none'    : no constraint
                'square'  : α_k = raw_alpha_k^2
                'exp'     : α_k = exp(raw_alpha_k)
                'softplus': α_k = log(1 + exp(raw_alpha_k))
                'abs'     : α_k = |raw_alpha_k|
            - normalize_alphas: If True, alphas will sum to 1 #NOTE: this is done after applying the constraint
            - alpha_init: How to initialize raw alpha parameters
            - symmetry_tolerance: Warn if ||K - K.T||_F / ||K||_F > tolerance #NOTE: this will only give you a warning if the asymmetry is above this tolerance, but it will always enforce symmetry, in other words, the resulting kernel matrix K will always be symmetric, but while averaging K with K.T if for an item the asymmetry is above this tolerance, a warning will be issued to alert the user of potential numerical instability. (Might be used for debugging or monitoring purposes.)
        """
        super().__init__()
        
        self.sub_kernels = nn.ModuleList(sub_kernels) #NOTE: this to make sure the sub-kernels are registered as sub-modules of the main module. Without ModuleList, PyTorch would not see the params to tweak for each sub-kernel.
        self.p = len(sub_kernels)
        self.alpha_constraint = alpha_constraint
        self.normalize_alphas = normalize_alphas
        self.symmetry_tolerance = symmetry_tolerance
        
        # Initialize raw alpha parameters
        if isinstance(alpha_init, str):
            if alpha_init == 'random':
                raw_init = torch.randn(self.p) * 0.1
            elif alpha_init == 'ones':
                raw_init = torch.ones(self.p)
            else:
                raise ValueError(f"Unknown alpha_init: {alpha_init}")
        else:
            raw_init = alpha_init
        
        # For constraints like 'square', we want sqrt of desired value as raw param
        if alpha_constraint == 'square' and isinstance(alpha_init, str) and alpha_init != 'random':
            raw_init = torch.sqrt(raw_init) # the raw_init are if you want the base values of alphas before the transformation
        elif alpha_constraint == 'exp' and isinstance(alpha_init, str) and alpha_init != 'random':
            raw_init = torch.log(raw_init + 1e-8)
        
        self.raw_alphas = nn.Parameter(raw_init) # learnable raw alpha parameters, #NOTE: later you will see we will apply the constraint to get the actual alphas used in the weighted sum. So in the example of 'square', the actual alpha will be raw_alpha^2. (Check the _get_alphas method below)
    
    def _get_alphas(self) -> torch.Tensor:
        """
        Apply constraint and optional normalization to get actual alpha values.
        
        Returns:
            alphas: Constrained and optionally normalized alpha weights (p,)
        """
        # Step 1: Apply constraint
        if self.alpha_constraint == 'none':
            alphas = self.raw_alphas
        elif self.alpha_constraint == 'square':
            alphas = self.raw_alphas ** 2
        elif self.alpha_constraint == 'exp':
            alphas = torch.exp(self.raw_alphas)
        elif self.alpha_constraint == 'softplus':
            alphas = F.softplus(self.raw_alphas)
        elif self.alpha_constraint == 'abs':
            alphas = torch.abs(self.raw_alphas)
        else:
            raise ValueError(f"Unknown alpha_constraint: {self.alpha_constraint}")
        
        # Step 2: Optionally normalize (after constraint)
        if self.normalize_alphas:
            alphas = alphas / (alphas.sum() + 1e-8)
        
        return alphas
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel matrix K from input data X.
        
        Args:
            X: Input data of shape (n, p) where n is number of samples,
               p is number of features
               
        Returns:
            K: (Guaranteed) Symmetric kernel matrix of shape (n, n)
        """
        n, p = X.shape
        assert p == self.p, f"Expected {self.p} features, got {p}"
        
        # Compute all sub-kernel matrices (vectorized)
        K_list = [] # list where each element is (n,n) kernel matrix for a feature, and the list length is p
        for k, sub_kernel in enumerate(self.sub_kernels):
            K_k = sub_kernel(X[:, k])  # (n, n), NOTE: here X[:, k] is the k-th feature column, shape (n,)
            K_list.append(K_k)
        
        K_stack = torch.stack(K_list, dim=0)  # (p, n, n), like we were discussing before list of p (n,n) matrices becomes a tensor of shape (p,n,n) because of stacking along a new dimension 0, as an example if I stacked along dim=1, the shape would be (n,p,n) which is not what we want.
        
        # Get constrained alphas (after applying constraints like square, exp, etc. and normalization if any)
        alphas = self._get_alphas()  # (p,)
        
        # Weighted sum (vectorized using einsum) 
        # NOTE: this is just a fancy way(for vectorization efficiency) of doing:

        # K_{i,j} = \sum_{p=1}^{P} \alpha_p K_{i,j}^{(p)}

        K = torch.einsum('p,pij->ij', alphas, K_stack)  # (n, n), this will do sum over p dimension, multiplying each (n,n) matrix by its corresponding alpha weight. #NOTE: check the einsum documentation: https://pytorch.org/docs/stable/generated/torch.einsum.html
        
        # Check symmetry before forcing it(for debugging/monitoring like we discussed above)
        asymmetry = torch.norm(K - K.T, p='fro') / (torch.norm(K, p='fro') + 1e-8)
        if asymmetry > self.symmetry_tolerance:
            warnings.warn(
                f"Kernel matrix asymmetry detected: {asymmetry.item():.2e}. "
                f"This may indicate numerical instability."
            )
        
        # Force perfect symmetry(as mentioned above, this is done by averaging K with its transpose) even if asymmetry is below tolerance
        K = 0.5 * (K + K.T)
        
        return K
    
    def get_lasso_penalty(self, lambda_val: float = 0.0) -> torch.Tensor:
        """
        Compute L1 penalty on alpha weights (the constrained and optionally normalized alphas) for feature selection.
        
        Args:
            lambda_val: Lasso regularization strength, keep 0.0 to disable
            
        Returns:
            penalty: λ * Σ|α_k|
        """
        if lambda_val == 0.0:
            return torch.tensor(0.0, device=self.raw_alphas.device)
        
        alphas = self._get_alphas() #TODO: I am worried that we are applying another time the transformation, and what would happen to the gradients? will pytorch autograd handle this correctly.
        return lambda_val * torch.sum(torch.abs(alphas)) # the regular L1 penalty on the alphas #NOTE: whenever I say alphas here, I mean the constrained and optionally normalized alphas, not the raw ones. The raw ones are just the parameters we optimize over, but the actual alphas used in the weighted sum are the constrained ones. 
    
    ## How to initialize the KernelNetwork in different ways:
    #0. Just pass an explicit list of sub-kernels instances

    #1. If for example you want to create a KernelNetwork with all same sub-kernels, you can use the from_uniform_kernels class method:
    @classmethod    
    def from_uniform_kernels(
        cls, # the cls here refers to the KernelNetwork class itself 
        kernel_class: Type[BaseSubKernel],
        num_features: int, 
        kernel_params: Optional[Dict[str, Any]] = None, # these are for example initial_sigma for LinearSubKernel, or initial_gamma for RBFSubKernel, we pass them as so: {'initial_sigma': 1.0} or {'initial_gamma': 0.5}
        **network_kwargs  # Arguments for KernelNetwork.__init__ e.g. alpha_constraint, normalize_alphas, alpha_init, symmetry_tolerance
    ) -> 'KernelNetwork':
        """
        Create a KernelNetwork where all features use the same type of sub-kernel.
        
        Args:
            kernel_class: SubKernel class (e.g., LinearSubKernel)
            num_features: Number of features (p)
            kernel_params: Parameters to pass to each kernel instance
            **network_kwargs: Arguments for KernelNetwork.__init__
            
        Returns:
            model: KernelNetwork instance
        """
        kernel_params = kernel_params or {}
        sub_kernels = [kernel_class(**kernel_params) for _ in range(num_features)] #NOTE: the ** is to unpack the dictionary into keyword arguments, make sure the keys in kernel_params match the parameter names in the kernel_class constructor
        return cls(sub_kernels, **network_kwargs)  # instantiate the KernelNetwork with the list of sub-kernels and any additional kwargs
    
    #2. If you want to create a KernelNetwork with a repeating pattern of sub-kernels, you can use the from_pattern class method: #NOTE: you could also use this method to create a uniform kernel network by passing a pattern of length 1. But I included both methods for clarity and convenience.
    @classmethod
    def from_pattern(
        cls, # the cls here refers to the KernelNetwork class itself
        pattern: List[Union[str, Type[BaseSubKernel]]], 
        num_features: int,
        kernel_params: Optional[Dict[str, Dict[str, Any]]] = None,
        **network_kwargs
    ) -> 'KernelNetwork':
        """
        Create a KernelNetwork with a repeating pattern of sub-kernels.
        
        Args:
            pattern: List of kernel classes or names (e.g., ['linear', 'rbf', LinearSubKernel]) # List of kernel classes or names (e.g., ['linear', 'rbf', LinearSubKernel]) #NOTE: for passing string names, I am currently suppporting only 3 types: 'linear', 'rbf', 'polynomial'. You can easily extend this mapping in the kernel_map dictionary below if you have more sub-kernel types, but I suggest if you have many types, you might want to just pass the classes directly instead of strings.

            num_features: Total number of features (will cycle through pattern) e.g. if pattern=['linear', 'rbf'] and num_features=5, the resulting sub-kernels will be [LinearSubKernel, RBFSubKernel, LinearSubKernel, RBFSubKernel, LinearSubKernel]

            kernel_params: Dict mapping kernel names to their params #NOTE: here you must be careful to match the keys in this dictionary to the kernel names used in the pattern. For example, if your pattern includes 'linear' and 'rbf', your kernel_params should look like: {'linear': {'initial_sigma': 1.0}, 'rbf': {'initial_gamma': 0.5}}. If you pass a kernel class directly in the pattern, the name will be derived from the class name (e.g., LinearSubKernel -> 'linear'), so make sure to use that as the key in kernel_params if you want to specify parameters for that kernel (this is shown in this line of code kernel_name = kernel_class.__name__.lower().replace('subkernel', '')) make sure to match the keys accordingly. #NOTE: if you are not careful here this won't fail but it will give you warnings #TODO: one design issue here is I won't be able to pass different params for the same kernel type, only if I create another name for the type, this must be fixed.

            **network_kwargs: Arguments for KernelNetwork.__init__ #NOTE: these are the same as in the KernelNetwork constructor, like alpha_constraint, normalize_alphas, alpha_init, symmetry_tolerance
            
        Returns:
            model: KernelNetwork instance
            
        Example:
            model = KernelNetwork.from_pattern(
                pattern=['linear', 'rbf'],
                num_features=10,  # Will alternate: L, R, L, R, L, R, L, R, L, R
                kernel_params={'linear': {'initial_sigma': 1.0}, 'rbf': {'initial_gamma': 0.5}}
            )
        """
        kernel_params = kernel_params or {}
        
        # Map string names to classes
        kernel_map = {
            'linear': LinearSubKernel,
            'rbf': RBFSubKernel,
            'polynomial': PolynomialSubKernel,
        }
        
        sub_kernels = []
        for i in range(num_features):
            pattern_idx = i % len(pattern)
            kernel_spec = pattern[pattern_idx]
            
            # Get kernel class
            if isinstance(kernel_spec, str):
                kernel_class = kernel_map[kernel_spec.lower()]
                kernel_name = kernel_spec.lower()
            else:
                kernel_class = kernel_spec
                kernel_name = kernel_class.__name__.lower().replace('subkernel', '')
            
            # Get params for this kernel type
            params = kernel_params.get(kernel_name, {})
            if kernel_name not in kernel_params and kernel_params:
                warnings.warn(f"No parameters provided for kernel '{kernel_name}'. Using default parameters.")

            sub_kernels.append(kernel_class(**params)) # instantiate the sub-kernel with its parameters
        
        return cls(sub_kernels, **network_kwargs)
