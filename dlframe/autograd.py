"""Function base class and all differentiable operation subclasses."""

import numpy as np
from dlframe.tensor import Tensor, Context


# ═══════════════════════════════════════════════════════════════════════════
# Function 基类
# ═══════════════════════════════════════════════════════════════════════════

class Function:
    """Base class for all differentiable operations.

    Subclasses must implement:
      - forward(ctx, *inputs)        : compute output from numpy arrays,
                                       save intermediate values to ctx.
      - backward(ctx, grad_output)   : compute gradients for each input,
                                       return a tuple matching forward's inputs.
    """

    @staticmethod
    def forward(ctx: Context, *inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output: np.ndarray) -> tuple:
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs, **kwargs):
        """Execute forward, wrap result as Tensor, and wire up the computation graph."""
        ctx = Context()
        raw = []
        tensors = []
        for inp in inputs:
            if isinstance(inp, Tensor):
                raw.append(inp.data)
                tensors.append(inp)
            elif inp is None:
                # None values pass through (e.g., default axes for transpose)
                raw.append(None)
                tensors.append(None)
            else:
                raw.append(np.asarray(inp, dtype=np.float64))
                tensors.append(Tensor(np.asarray(inp, dtype=np.float64)))

        result_data = cls.forward(ctx, *raw, **kwargs)
        requires_grad = any(t.requires_grad for t in tensors if isinstance(t, Tensor))
        result = Tensor(
            result_data if isinstance(result_data, np.ndarray)
            else np.asarray(result_data, dtype=np.float64),
            requires_grad=requires_grad,
        )

        if requires_grad:
            result._ctx = ctx
            result._grad_fn = cls.backward
            result._parents = tuple(t for t in tensors if isinstance(t, Tensor))

        return result


# ═══════════════════════════════════════════════════════════════════════════
# 算术操作
# ═══════════════════════════════════════════════════════════════════════════

class Add(Function):
    """Element-wise addition: y = a + b (with broadcasting)."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.shape, b.shape)
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.get_saved()
        grad_a = Tensor._broadcast_grad(grad_output, a_shape)
        grad_b = Tensor._broadcast_grad(grad_output, b_shape)
        return grad_a, grad_b


class Sub(Function):
    """Element-wise subtraction: y = a - b."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.shape, b.shape)
        return a - b

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.get_saved()
        grad_a = Tensor._broadcast_grad(grad_output, a_shape)
        grad_b = Tensor._broadcast_grad(-grad_output, b_shape)
        return grad_a, grad_b


class Mul(Function):
    """Element-wise multiplication: y = a * b (with broadcasting)."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b, a.shape, b.shape)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b, a_shape, b_shape = ctx.get_saved()
        grad_a = Tensor._broadcast_grad(grad_output * b, a_shape)
        grad_b = Tensor._broadcast_grad(grad_output * a, b_shape)
        return grad_a, grad_b


class Neg(Function):
    """Negation: y = -x."""

    @staticmethod
    def forward(ctx, x):
        return -x

    @staticmethod
    def backward(ctx, grad_output):
        return (-grad_output,)


class Pow(Function):
    """Power: y = x ** exponent  (exponent must be a scalar)."""

    @staticmethod
    def forward(ctx, x, exponent):
        ctx.save_for_backward(x, exponent)
        return x ** exponent

    @staticmethod
    def backward(ctx, grad_output):
        x, exponent = ctx.get_saved()
        grad_x = grad_output * exponent * (x ** (exponent - 1))
        return (grad_x,)


# ═══════════════════════════════════════════════════════════════════════════
# 矩阵操作
# ═══════════════════════════════════════════════════════════════════════════

class MatMul(Function):
    """Matrix multiplication: y = a @ b."""

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.get_saved()
        # Ensure at least 2D for matrix operations
        grad_output = np.atleast_2d(grad_output)
        a_2d = np.atleast_2d(a)
        b_2d = np.atleast_2d(b)

        grad_a = grad_output @ b_2d.swapaxes(-1, -2)
        grad_b = a_2d.swapaxes(-1, -2) @ grad_output

        # Squeeze to match original input shapes
        if a.ndim < a_2d.ndim:
            grad_a = grad_a.reshape(a.shape)
        if b.ndim < b_2d.ndim:
            grad_b = grad_b.reshape(b.shape)

        return grad_a, grad_b


class Sum(Function):
    """Sum over axes: y = x.sum(axis, keepdims)."""

    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        ctx.save_for_backward(x.shape, axis, keepdims)
        return x.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        original_shape, axis, keepdims = ctx.get_saved()
        # Broadcast grad_output back to original shape
        if not keepdims and axis is not None:
            # Insert summed dimensions back
            if isinstance(axis, int):
                axis = (axis,)
            grad = np.expand_dims(grad_output, axis=axis)
            # np.expand_dims shifts axes; we need to place at original positions
            grad = np.broadcast_to(grad, original_shape).copy()
        else:
            grad = np.broadcast_to(grad_output, original_shape).copy()
        return (grad,)


class Reshape(Function):
    """Reshape: y = x.reshape(shape)."""

    @staticmethod
    def forward(ctx, x, shape):
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        (original_shape,) = ctx.get_saved()
        return (grad_output.reshape(original_shape),)


class Transpose(Function):
    """Transpose: y = x.transpose(axes)."""

    @staticmethod
    def forward(ctx, x, axes=None):
        ctx.save_for_backward(axes)
        return np.transpose(x, axes)

    @staticmethod
    def backward(ctx, grad_output):
        (axes,) = ctx.get_saved()
        if axes is None:
            return (np.transpose(grad_output),)
        # Inverse permutation
        inv_axes = np.argsort(axes)
        return (np.transpose(grad_output, inv_axes),)


# ═══════════════════════════════════════════════════════════════════════════
# 激活函数
# ═══════════════════════════════════════════════════════════════════════════

class ReLUFunc(Function):
    """Rectified Linear Unit: y = max(0, x)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.maximum(0, x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.get_saved()
        return (grad_output * (x > 0),)


class SigmoidFunc(Function):
    """Sigmoid: y = 1 / (1 + exp(-x))."""

    @staticmethod
    def forward(ctx, x):
        # Numerically stable sigmoid
        y = np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.get_saved()
        return (grad_output * y * (1 - y),)


class TanhFunc(Function):
    """Hyperbolic tangent: y = tanh(x)."""

    @staticmethod
    def forward(ctx, x):
        y = np.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.get_saved()
        return (grad_output * (1 - y * y),)


class SoftmaxFunc(Function):
    """Softmax along a given axis (numerically stable)."""

    @staticmethod
    def forward(ctx, x, axis=-1):
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(shifted)
        y = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # Softmax Jacobian: diag(y) - y y^T applied to grad_output
        # => y * (grad_output - sum(grad_output * y, axis=-1, keepdims=True))
        (y,) = ctx.get_saved()
        s = np.sum(grad_output * y, axis=-1, keepdims=True)
        return (y * (grad_output - s),)
