"""Dropout layer — randomly zeros inputs during training for regularization."""

import numpy as np
from dlframe.nn.module import Module
from dlframe.tensor import Tensor


class Dropout(Module):
    """Dropout regularization layer.

    During training, each element is zeroed with probability `p` and the
    remaining elements are scaled by 1/(1-p) (inverted dropout). During
    evaluation, the input is passed through unchanged.

    Args:
        p: Probability of an element to be zeroed (default: 0.5).
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"Dropout p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self._training or self.p == 0:
            return x
        mask = np.random.binomial(1, 1 - self.p, x.data.shape).astype(np.float64)
        mask /= (1 - self.p)
        return x * mask

    def __repr__(self):
        return f"Dropout(p={self.p})"
