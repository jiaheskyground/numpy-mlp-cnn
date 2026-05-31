"""Tensor class — the core data structure wrapping NumPy arrays with autograd support."""

import numpy as np
from typing import Optional, Callable


class Context:
    """Stores intermediate values during forward pass for use in backward pass."""

    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def get_saved(self):
        return self.saved_tensors


class Tensor:
    """A multi-dimensional array with automatic differentiation support.

    Attributes:
        data: The underlying NumPy array.
        grad: Gradient of the loss with respect to this tensor.
        requires_grad: Whether this tensor participates in gradient computation.
    """

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _grad_fn: Optional[Callable] = None,
        _ctx: Optional[Context] = None,
        _parents: tuple = (),
    ):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self._grad_fn = _grad_fn
        self._ctx = _ctx
        self._parents = _parents

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    # ── backward traversal ────────────────────────────────────────────────

    def backward(self, grad: Optional[np.ndarray] = None):
        """Execute reverse-mode automatic differentiation starting from this tensor.

        Performs topological sort of the computation graph, then computes
        gradients in reverse order, accumulating them to leaf tensors.
        """
        # Build topological order via DFS
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build_topo(node: Tensor):
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node._parents:
                    build_topo(parent)
                topo.append(node)

        build_topo(self)

        # Seed gradient
        if grad is None:
            self.grad = np.ones_like(self.data, dtype=np.float64)
        else:
            self.grad = np.asarray(grad, dtype=np.float64)

        # Process in reverse topological order
        for node in reversed(topo):
            if node._grad_fn is not None:
                grads = node._grad_fn(node._ctx, node.grad)
                for parent, g in zip(node._parents, grads):
                    if parent.requires_grad:
                        self._accumulate_grad(parent, g)

    @staticmethod
    def _accumulate_grad(tensor: "Tensor", grad: np.ndarray):
        if tensor.grad is None:
            tensor.grad = np.asarray(grad, dtype=np.float64)
        else:
            tensor.grad += grad

    # ── helper: broadcast-aware gradient sum ──────────────────────────────

    @staticmethod
    def _broadcast_grad(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Sum gradient along broadcast axes to match target_shape."""
        gshape = grad.shape
        tshape = target_shape
        if gshape == tshape:
            return grad
        # Pad dimensions on the left
        ndim_diff = len(gshape) - len(tshape)
        if ndim_diff > 0:
            # Sum leading extra dims
            axes = tuple(range(ndim_diff))
            grad = grad.sum(axis=axes)
            gshape = grad.shape
        # Sum axes where target is 1 (broadcast axes)
        axis_to_sum = []
        for i, (gs, ts) in enumerate(zip(gshape, tshape)):
            if ts == 1 and gs > 1:
                axis_to_sum.append(i)
        if axis_to_sum:
            grad = grad.sum(axis=tuple(axis_to_sum), keepdims=True)
        # Reshape to target
        return grad.reshape(target_shape)

    # ── operator overloading ──────────────────────────────────────────────

    def __add__(self, other):
        from dlframe.autograd import Add
        if isinstance(other, (int, float)):
            other = np.asarray(other, dtype=np.float64)
        return Add.apply(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __matmul__(self, other):
        from dlframe.autograd import MatMul
        return MatMul.apply(self, other)

    def __rmatmul__(self, other):
        from dlframe.autograd import MatMul
        return MatMul.apply(other, self)

    def __mul__(self, other):
        from dlframe.autograd import Mul
        if isinstance(other, (int, float)):
            other = np.asarray(other, dtype=np.float64)
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        from dlframe.autograd import Neg
        return Neg.apply(self)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = np.asarray(other, dtype=np.float64)
        return self * (other ** -1)

    def __pow__(self, exponent):
        from dlframe.autograd import Pow
        return Pow.apply(self, exponent)

    # ── activation / reduction helpers ────────────────────────────────────

    def relu(self):
        from dlframe.autograd import ReLUFunc
        return ReLUFunc.apply(self)

    def sigmoid(self):
        from dlframe.autograd import SigmoidFunc
        return SigmoidFunc.apply(self)

    def tanh(self):
        from dlframe.autograd import TanhFunc
        return TanhFunc.apply(self)

    def sum(self, axis=None, keepdims=False):
        from dlframe.autograd import Sum
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            n = self.data.size
        elif isinstance(axis, int):
            n = self.data.shape[axis]
        else:
            n = 1
            for ax in axis:
                n *= self.data.shape[ax]
        return s * (1.0 / n)

    def reshape(self, *shape):
        from dlframe.autograd import Reshape
        return Reshape.apply(self, shape=shape)

    def transpose(self, axes=None):
        from dlframe.autograd import Transpose
        return Transpose.apply(self, axes=axes)

    @property
    def T(self):
        return self.transpose()

    # ── utility ───────────────────────────────────────────────────────────

    def item(self):
        """Return a Python scalar (only valid for 0-d / 1-element tensors)."""
        return self.data.item()

    def astype(self, dtype):
        """Return a new Tensor cast to dtype (detached, no grad)."""
        return Tensor(self.data.astype(dtype))

    def __repr__(self):
        req = "True" if self.requires_grad else "False"
        g = f" grad={self.grad.shape}" if self.grad is not None else ""
        return f"Tensor({self.data}, requires_grad={req}{g})"

    def __array__(self):
        """Allow np.asarray(tensor) to extract underlying data directly."""
        return self.data
