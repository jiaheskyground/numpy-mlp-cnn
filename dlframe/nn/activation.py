"""Activation function layers (ReLU, Sigmoid, Tanh, Softmax)."""

from dlframe.nn.module import Module


class ReLU(Module):
    """ReLU activation: max(0, x). No learnable parameters."""

    def forward(self, x):
        return x.relu()

    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation: 1 / (1 + exp(-x)). No learnable parameters."""

    def forward(self, x):
        return x.sigmoid()

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Tanh activation: tanh(x). No learnable parameters."""

    def forward(self, x):
        return x.tanh()

    def __repr__(self):
        return "Tanh()"


class Softmax(Module):
    """Softmax activation along a given dimension. No learnable parameters.

    Args:
        dim: The dimension along which to compute softmax (default: -1).
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        from dlframe.autograd import SoftmaxFunc
        return SoftmaxFunc.apply(x, axis=self.dim)

    def __repr__(self):
        return f"Softmax(dim={self.dim})"
