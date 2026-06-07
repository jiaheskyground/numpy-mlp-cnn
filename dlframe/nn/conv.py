"""Convolutional layer for 2D feature maps."""

import numpy as np
from dlframe.nn.module import Module
from dlframe.parameter import Parameter
from dlframe.tensor import Tensor


class Conv2d(Module):
    """2D Convolutional layer.
    
    Applies a 2D convolution with learnable kernels and bias.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Size of the convolutional kernel (int or tuple).
        stride: Stride of the convolution (default: 1).
        padding: Padding added to input (default: 0).
        bias: Whether to add a learnable bias (default: True).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        
        self.stride = stride
        self.padding = padding
        
        # Weight: (out_channels, in_channels, kH, kW)
        kh, kw = self.kernel_size
        fan_in = in_channels * kh * kw
        std = np.sqrt(2.0 / fan_in)  # He initialization
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, kh, kw).astype(np.float64) * std
        )
        
        if bias:
            self.bias = Parameter(np.zeros(out_channels, dtype=np.float64))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using im2col convolution.
        
        Args:
            x: Input tensor of shape (N, C_in, H, W).
        
        Returns:
            Output tensor of shape (N, C_out, H_out, W_out).
        """
        from dlframe.autograd import Conv2dFunc
        
        return Conv2dFunc.apply(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )
    
    def __repr__(self):
        return (f"Conv2d(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding})")
