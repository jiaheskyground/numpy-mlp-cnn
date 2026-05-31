"""Loss functions: MSELoss and CrossEntropyLoss."""

import numpy as np
from dlframe.nn.module import Module
from dlframe.tensor import Tensor


class MSELoss(Module):
    """Mean Squared Error loss.

    L = reduce((pred - target)^2)

    Args:
        reduction: 'mean' (default) or 'sum'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target) -> Tensor:
        if not isinstance(target, Tensor):
            target = Tensor(np.asarray(target, dtype=np.float64))
        diff = pred - target
        sq = diff * diff
        if self.reduction == "mean":
            return sq.sum() * (1.0 / sq.data.size)
        else:
            return sq.sum()

    def __repr__(self):
        return f"MSELoss(reduction='{self.reduction}')"


class CrossEntropyLoss(Module):
    """Cross-entropy loss with built-in log-softmax for numerical stability.

    L = -mean(log(softmax(pred)[target])) where target contains class indices.

    The backward pass uses the simplified gradient:
        (softmax(pred) - one_hot(target)) / N

    Args:
        reduction: 'mean' (default) or 'sum'.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            pred: Raw logits, shape (N, C).
            target: Class indices, shape (N,) — integer labels.

        Returns:
            Scalar loss Tensor.
        """
        from dlframe.autograd import Function, Context

        if isinstance(target, Tensor):
            target_data = target.data
        else:
            target_data = np.asarray(target)

        # Use the combined softmax + CE backward for efficiency
        ctx = Context()

        # Numerically stable log-softmax
        x = pred.data
        x_max = np.max(x, axis=1, keepdims=True)
        shifted = x - x_max
        exp_x = np.exp(shifted)
        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        log_softmax = shifted - np.log(np.sum(exp_x, axis=1, keepdims=True))

        # Loss computation
        n = x.shape[0]
        target_int = target_data.astype(np.int64)
        loss_val = -log_softmax[np.arange(n), target_int]

        if self.reduction == "mean":
            loss_val = np.mean(loss_val)
            scale = 1.0 / n
        else:
            loss_val = np.sum(loss_val)
            scale = 1.0

        # Save for backward
        ctx.save_for_backward(softmax, target_data, scale)

        result = Tensor(loss_val, requires_grad=pred.requires_grad)
        if pred.requires_grad:
            result._ctx = ctx
            result._grad_fn = CrossEntropyLoss._backward
            result._parents = (pred,)

        return result

    @staticmethod
    def _backward(ctx, grad_output):
        softmax, target_data, scale = ctx.get_saved()
        n = softmax.shape[0]
        # Create one-hot gradient
        grad = softmax.copy()
        grad[np.arange(n), target_data.astype(np.int64)] -= 1.0
        grad *= scale * grad_output.item() if isinstance(grad_output, np.ndarray) and grad_output.size == 1 else scale
        return (grad,)

    def __repr__(self):
        return f"CrossEntropyLoss(reduction='{self.reduction}')"
