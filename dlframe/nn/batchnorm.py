"""Batch Normalization layer for 2D convolutional networks."""

import numpy as np
from dlframe.nn.module import Module
from dlframe.parameter import Parameter
from dlframe.tensor import Tensor


class BatchNorm2d(Module):
    """Batch Normalization for 2D feature maps (N, C, H, W).
    
    Normalizes each channel independently across the batch dimension,
    applying learnable affine transformation (scale and shift).
    
    Args:
        num_features: Number of channels (C in input shape N, C, H, W).
        eps: Small constant for numerical stability (default: 1e-5).
        momentum: Momentum for running mean/var estimates (default: 0.1).
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable scale (gamma) and shift (beta)
        self.weight = Parameter(np.ones(num_features, dtype=np.float64))  # gamma
        self.bias = Parameter(np.zeros(num_features, dtype=np.float64))   # beta
        
        # Running estimates (not Parameters, not updated by optimizer)
        self.running_mean = np.zeros(num_features, dtype=np.float64)
        self.running_var = np.ones(num_features, dtype=np.float64)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with batch normalization.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
        
        Returns:
            Normalized and scaled tensor of shape (N, C, H, W).
        """
        from dlframe.autograd import BatchNorm2dFunc
        
        if self._training:
            # Training mode: compute batch statistics
            return BatchNorm2dFunc.apply(
                x,
                self.weight,
                self.bias,
                running_mean=self.running_mean,
                running_var=self.running_var,
                eps=self.eps,
                momentum=self.momentum,
                training=True,
            )
        else:
            # Evaluation mode: use running statistics
            return BatchNorm2dFunc.apply(
                x,
                self.weight,
                self.bias,
                running_mean=self.running_mean,
                running_var=self.running_var,
                eps=self.eps,
                momentum=self.momentum,
                training=False,
            )
    
    def __repr__(self):
        return (f"BatchNorm2d(num_features={self.num_features}, "
                f"eps={self.eps}, momentum={self.momentum})")
