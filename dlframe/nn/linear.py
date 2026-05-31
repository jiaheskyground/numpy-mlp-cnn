"""Linear (fully-connected) layer."""

import numpy as np
from dlframe.nn.module import Module
from dlframe.parameter import Parameter
from dlframe.autograd import MatMul, Add


class Linear(Module):
    """Fully-connected layer: y = x @ W.T + b.

    Args:
        in_features: Dimensionality of each input sample.
        out_features: Dimensionality of each output sample.
        bias: If True, adds a learnable bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight shape: (out_features, in_features) — consistent with PyTorch
        self.weight = Parameter(
            np.random.randn(out_features, in_features).astype(np.float64) * np.sqrt(2.0 / in_features)
        )

        if bias:
            self.bias = Parameter(np.zeros(out_features, dtype=np.float64))
        else:
            self.bias = None

    def forward(self, x):
        # x shape: (batch, in_features)
        # W shape: (out_features, in_features)
        # y = x @ W.T  → (batch, out_features)
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
