"""SGD optimizer with optional momentum and weight decay."""

import numpy as np
from dlframe.parameter import Parameter


class SGD:
    """Stochastic Gradient Descent with optional momentum and weight decay.

    Update rule (with momentum and weight_decay):
        v = momentum * v + grad + weight_decay * param
        param = param - lr * v

    Args:
        params: List of parameters to optimize.
        lr: Learning rate.
        momentum: Momentum factor (0 = vanilla SGD).
        weight_decay: L2 regularization strength (0 = no weight decay).
    """

    def __init__(
        self,
        params: list,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        self.params = [p for p in params if isinstance(p, Parameter)]
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._v = [np.zeros_like(p.data) for p in self.params] if momentum > 0 else None

    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.params:
            p.grad = None

    def step(self):
        """Perform a single optimization step."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad

            # Weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            # Momentum
            if self.momentum > 0:
                self._v[i] = self.momentum * self._v[i] + grad
                p.data -= self.lr * self._v[i]
            else:
                p.data -= self.lr * grad

    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum}, weight_decay={self.weight_decay})"
