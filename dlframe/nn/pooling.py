"""Pooling layers."""

import numpy as np
from dlframe.nn.module import Module
from dlframe.tensor import Tensor


class MaxPool2d(Module):
    """Max pooling layer for 2D feature maps.
    
    Args:
        kernel_size: Size of the pooling window (int or tuple).
        stride: Stride of the pooling (default: same as kernel_size).
        padding: Padding applied to input (default: 0).
    """
    
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        
        self.stride = stride if stride is not None else self.kernel_size
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with max pooling.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
        
        Returns:
            Pooled output tensor.
        """
        from dlframe.autograd import MaxPool2dFunc
        
        return MaxPool2dFunc.apply(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
    
    def __repr__(self):
        return (f"MaxPool2d(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")


class AvgPool2d(Module):
    """Average pooling layer for 2D feature maps.
    
    Args:
        kernel_size: Size of the pooling window (int or tuple).
        stride: Stride of the pooling (default: same as kernel_size).
        padding: Padding applied to input (default: 0).
    """
    
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        
        self.stride = stride if stride is not None else self.kernel_size
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with average pooling.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
        
        Returns:
            Pooled output tensor.
        """
        from dlframe.autograd import AvgPool2dFunc
        
        return AvgPool2dFunc.apply(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
    
    def __repr__(self):
        return (f"AvgPool2d(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")
