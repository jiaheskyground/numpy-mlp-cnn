"""Function base class and all differentiable operation subclasses."""

import numpy as np
from dlframe.tensor import Tensor, Context


# ═════════════════════════════════════════════════════════════════
# Function 基类
# ═════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# 算术操作
# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# 矩阵操作
# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# 激活函数
# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# 卷积操作 (Conv2d)
# ════════════════════════════════════════════════════════════════

class Conv2dFunc(Function):
    """2D Convolution using im2col method."""

    @staticmethod
    def forward(ctx, x, w, b, stride=1, padding=0):
        """
        Args:
            x: Input (N, C_in, H, W)
            w: Weight (C_out, C_in, kH, kW)
            b: Bias (C_out,) or None
            stride: Convolution stride
            padding: Padding size
        
        Returns:
            Output (N, C_out, H_out, W_out)
        """
        N, C_in, H, W = x.shape
        C_out, _, kH, kW = w.shape
        
        # Pad input
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        else:
            x_padded = x
        
        # Compute output dimensions
        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1
        
        # Im2col: convert to (N*H_out*W_out, C_in*kH*kW)
        col = np.zeros((N * H_out * W_out, C_in * kH * kW), dtype=x.dtype)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                patch = x_padded[:, :, h_start:h_start+kH, w_start:w_start+kW]
                col[idx:idx+N] = patch.reshape(N, -1)
                idx += N
        
        # Reshape weight: (C_out, C_in*kH*kW)
        w_mat = w.reshape(C_out, -1)  # (C_out, C_in*kH*kW)
        
        # Convolution: col @ w.T
        out = col @ w_mat.T  # (N*H_out*W_out, C_out)
        
        # Add bias if present
        if b is not None:
            out = out + b[np.newaxis, :]
        
        out = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)  # (N, C_out, H_out, W_out)
        
        # Save for backward
        ctx.save_for_backward(x, w, b, stride, padding, H, W, kH, kW, H_out, W_out, C_in, C_out)
        
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (N, C_out, H_out, W_out)
        """
        x, w, b, stride, padding, H, W, kH, kW, H_out, W_out, C_in, C_out = ctx.get_saved()
        N = x.shape[0]
        
        # Pad input
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        else:
            x_padded = x
        
        # Reshape grad_output: (N, C_out, H_out, W_out) -> (N*H_out*W_out, C_out)
        grad_out = grad_output.transpose(0, 2, 3, 1).reshape(-1, C_out)
        
        # Gradient w.r.t. weight
        col = np.zeros((N * H_out * W_out, C_in * kH * kW), dtype=x.dtype)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                patch = x_padded[:, :, h_start:h_start+kH, w_start:w_start+kW]
                col[idx:idx+N] = patch.reshape(N, -1)
                idx += N
        
        grad_w = col.T @ grad_out  # (C_in*kH*kW, C_out)
        grad_w = grad_w.T.reshape(C_out, C_in, kH, kW)
        
        # Gradient w.r.t. bias
        grad_b = None
        if b is not None:
            grad_b = np.sum(grad_out, axis=0)  # (C_out,)
        
        # Gradient w.r.t. input (requires col2im)
        w_mat = w.reshape(C_out, -1).T  # (C_in*kH*kW, C_out)
        grad_col = grad_out @ w_mat.T  # (N*H_out*W_out, C_in*kH*kW)
        
        grad_x_padded = np.zeros_like(x_padded)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                grad_patch = grad_col[idx:idx+N].reshape(N, C_in, kH, kW)
                grad_x_padded[:, :, h_start:h_start+kH, w_start:w_start+kW] += grad_patch
                idx += N
        
        # Remove padding
        if padding > 0:
            grad_x = grad_x_padded[:, :, padding:-padding, padding:-padding]
        else:
            grad_x = grad_x_padded
        
        return (grad_x, grad_w, grad_b)


# ════════════════════════════════════════════════════════════════
# Batch Normalization (BatchNorm2d)
# ════════════════════════════════════════════════════════════════

class BatchNorm2dFunc(Function):
    """Batch Normalization for 2D feature maps."""

    @staticmethod
    def forward(ctx, x, w, b, running_mean=None, running_var=None, eps=1e-5, momentum=0.1, training=True):
        """
        Args:
            x: Input (N, C, H, W)
            w: Scale/gamma (C,)
            b: Shift/beta (C,)
            running_mean: Running mean estimate (C,)
            running_var: Running variance estimate (C,)
            eps: Numerical stability constant
            momentum: Momentum for running statistics
            training: Whether in training mode
        """
        N, C, H, W = x.shape
        
        if training:
            # Compute batch statistics
            x_reshaped = x.reshape(N, C, -1)  # (N, C, H*W)
            mean = np.mean(x_reshaped, axis=(0, 2), keepdims=True)  # (1, C, 1)
            var = np.var(x_reshaped, axis=(0, 2), keepdims=True)    # (1, C, 1)
            
            # Update running statistics
            mean_scalar = mean.squeeze()
            var_scalar = var.squeeze()
            if running_mean is not None:
                running_mean[:] = momentum * mean_scalar + (1 - momentum) * running_mean
            if running_var is not None:
                running_var[:] = momentum * var_scalar + (1 - momentum) * running_var
            
            # Normalize
            x_norm = (x - mean) / np.sqrt(var + eps)
            
            # Scale and shift
            w_reshaped = w.reshape(1, C, 1, 1)
            b_reshaped = b.reshape(1, C, 1, 1)
            y = w_reshaped * x_norm + b_reshaped
            
            ctx.save_for_backward(x, mean, var, w, x_norm, eps, C)
        
        else:
            # Use running statistics
            mean = running_mean.reshape(1, C, 1, 1)
            var = running_var.reshape(1, C, 1, 1)
            
            x_norm = (x - mean) / np.sqrt(var + eps)
            
            w_reshaped = w.reshape(1, C, 1, 1)
            b_reshaped = b.reshape(1, C, 1, 1)
            y = w_reshaped * x_norm + b_reshaped
            
            ctx.save_for_backward(x, None, None, w, None, eps, C)
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, mean, var, w, x_norm, eps, C = ctx.get_saved()
        
        N, C, H, W = grad_output.shape
        
        w_reshaped = w.reshape(1, C, 1, 1)
        
        # Gradient w.r.t. scale and shift
        grad_w = np.sum(grad_output * x_norm, axis=(0, 2, 3))
        grad_b = np.sum(grad_output, axis=(0, 2, 3))
        
        # Gradient w.r.t. normalized input
        grad_x_norm = grad_output * w_reshaped
        
        # Gradient w.r.t. input (batch normalization backprop)
        x_reshaped = x.reshape(N, C, -1)
        grad_x_reshaped = grad_x_norm.reshape(N, C, -1)
        
        HW = H * W
        std = np.sqrt(var + eps)
        
        # Backprop through normalization
        dvar = np.sum(grad_x_reshaped * (x_reshaped - mean) * (-0.5) * (std ** -3), axis=(0, 2), keepdims=True)
        dmean = np.sum(grad_x_reshaped * (-1.0 / std), axis=(0, 2), keepdims=True)
        dmean += dvar * np.sum(-2.0 * (x_reshaped - mean), axis=(0, 2), keepdims=True) / (N * HW)
        
        grad_x = grad_x_reshaped / std + dvar * 2.0 * (x_reshaped - mean) / (N * HW) + dmean / (N * HW)
        
        grad_x = grad_x.reshape(N, C, H, W)
        
        return (grad_x, grad_w, grad_b)


# ════════════════════════════════════════════════════════════════
# Pooling (MaxPool2d, AvgPool2d)
# ════════════════════════════════════════════════════════════════

class MaxPool2dFunc(Function):
    """Max pooling for 2D feature maps."""

    @staticmethod
    def forward(ctx, x, kernel_size=2, stride=None, padding=0):
        """
        Args:
            x: Input (N, C, H, W)
            kernel_size: Pooling kernel size (int or tuple)
            stride: Pooling stride (default: kernel_size)
            padding: Padding size
        """
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        
        if stride is None:
            stride = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        
        N, C, H, W = x.shape
        
        # Pad input
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=-np.inf)
        else:
            x_padded = x
        
        H_padded = H + 2 * padding
        W_padded = W + 2 * padding
        
        H_out = (H_padded - kH) // stride + 1
        W_out = (W_padded - kW) // stride + 1
        
        # Perform max pooling with argmax tracking
        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        max_indices = np.zeros((N, C, H_out, W_out, 2), dtype=np.int32)
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                patch = x_padded[:, :, h_start:h_start+kH, w_start:w_start+kW]  # (N, C, kH, kW)
                
                # Max pooling
                patch_reshaped = patch.reshape(N, C, -1)
                max_vals = np.max(patch_reshaped, axis=2)
                max_idx = np.argmax(patch_reshaped, axis=2)
                
                out[:, :, i, j] = max_vals
                max_indices[:, :, i, j, 0] = max_idx // kW
                max_indices[:, :, i, j, 1] = max_idx % kW
        
        ctx.save_for_backward(x.shape, kernel_size, stride, padding, max_indices)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_shape, kernel_size, stride, padding, max_indices = ctx.get_saved()
        N, C, H, W = x_shape
        
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        
        if stride is None:
            stride = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        
        # Pad input
        if padding > 0:
            grad_x_padded = np.zeros((N, C, H + 2*padding, W + 2*padding), dtype=grad_output.dtype)
        else:
            grad_x_padded = np.zeros((N, C, H, W), dtype=grad_output.dtype)
        
        H_padded = H + 2 * padding
        W_padded = W + 2 * padding
        H_out = grad_output.shape[2]
        W_out = grad_output.shape[3]
        
        # Backpropagate gradients to max positions
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                
                for n in range(N):
                    for c in range(C):
                        max_h = max_indices[n, c, i, j, 0]
                        max_w = max_indices[n, c, i, j, 1]
                        grad_x_padded[n, c, h_start + max_h, w_start + max_w] += grad_output[n, c, i, j]
        
        # Remove padding
        if padding > 0:
            grad_x = grad_x_padded[:, :, padding:-padding, padding:-padding]
        else:
            grad_x = grad_x_padded
        
        return (grad_x,)


class AvgPool2dFunc(Function):
    """Average pooling for 2D feature maps."""

    @staticmethod
    def forward(ctx, x, kernel_size=2, stride=None, padding=0):
        """
        Args:
            x: Input (N, C, H, W)
            kernel_size: Pooling kernel size (int or tuple)
            stride: Pooling stride (default: kernel_size)
            padding: Padding size
        """
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        
        if stride is None:
            stride = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        
        N, C, H, W = x.shape
        
        # Pad input with zeros
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        else:
            x_padded = x
        
        H_padded = H + 2 * padding
        W_padded = W + 2 * padding
        
        H_out = (H_padded - kH) // stride + 1
        W_out = (W_padded - kW) // stride + 1
        
        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                patch = x_padded[:, :, h_start:h_start+kH, w_start:w_start+kW]
                out[:, :, i, j] = np.mean(patch, axis=(2, 3))
        
        ctx.save_for_backward(x.shape, kernel_size, stride, padding)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_shape, kernel_size, stride, padding = ctx.get_saved()
        N, C, H, W = x_shape
        
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        
        if stride is None:
            stride = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        
        # Pad gradient
        if padding > 0:
            grad_x_padded = np.zeros((N, C, H + 2*padding, W + 2*padding), dtype=grad_output.dtype)
        else:
            grad_x_padded = np.zeros((N, C, H, W), dtype=grad_output.dtype)
        
        H_out = grad_output.shape[2]
        W_out = grad_output.shape[3]
        
        pool_size = kH * kW
        grad_val = grad_output / pool_size
        
        # Distribute gradient to all positions in the pooling window
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                grad_x_padded[:, :, h_start:h_start+kH, w_start:w_start+kW] += grad_val[:, :, i:i+1, j:j+1]
        
        # Remove padding
        if padding > 0:
            grad_x = grad_x_padded[:, :, padding:-padding, padding:-padding]
        else:
            grad_x = grad_x_padded
        
        return (grad_x,)
