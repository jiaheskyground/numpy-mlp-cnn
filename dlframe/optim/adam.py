"""Adam optimizer."""

import numpy as np
from dlframe.parameter import Parameter


class Adam:
    """Adam optimizer with optional weight decay.

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        params: List of parameters to optimize.
        lr: Learning rate (default: 0.001).
        betas: Coefficients for running averages (default: (0.9, 0.999)).
        eps: Numerical stability term (default: 1e-8).
        weight_decay: L2 regularization strength (default: 0).
    """

    def __init__(
        self,
        params: list,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = [p for p in params if isinstance(p, Parameter)]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.params:
            p.grad = None

    def step(self):
        """Perform a single optimization step."""
        self._t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad

            # Weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            # Update biased moment estimates
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * grad * grad

            # Bias correction
            m_hat = self._m[i] / (1 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1 - self.beta2 ** self._t)

            # Update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self):
        return f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), eps={self.eps})"
