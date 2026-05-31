"""Gradient check tests for the autograd engine."""

import numpy as np
import sys
sys.path.insert(0, ".")

from dlframe.tensor import Tensor
from dlframe.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from dlframe.nn import init


def numerical_gradient(f, x, eps=1e-5):
    """Compute numerical gradient of f at x using central finite differences."""
    grad = np.zeros_like(x.data)
    flat_x = x.data.ravel()
    flat_grad = np.zeros_like(flat_x)
    for i in range(len(flat_x)):
        old = flat_x[i]
        flat_x[i] = old + eps
        x.data = flat_x.reshape(x.data.shape)
        y1 = f(x)

        flat_x[i] = old - eps
        x.data = flat_x.reshape(x.data.shape)
        y2 = f(x)

        flat_grad[i] = (y1.item() - y2.item()) / (2 * eps)
        flat_x[i] = old

    x.data = flat_x.reshape(x.data.shape)
    return flat_grad.reshape(x.data.shape)


def relative_error(analytical, numerical):
    """Compute relative error between two gradient arrays."""
    denom = np.abs(analytical) + np.abs(numerical) + 1e-8
    return np.max(np.abs(analytical - numerical) / denom)


def test_add():
    print("Testing Add backward...")
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    y = (a + b).sum()
    y.backward()

    def fn(x):
        return (x + b).sum()
    num_grad = numerical_gradient(fn, a)
    err = relative_error(a.grad, num_grad)
    print(f"  a.grad: {a.grad}, num: {num_grad}, rel_err: {err:.2e}")
    assert err < 1e-5, f"Add: gradient error too large: {err}"
    print("  PASSED")


def test_mul():
    print("Testing Mul backward...")
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    y = (a * b).sum()
    y.backward()

    def fn(x):
        return (x * b).sum()
    num_grad = numerical_gradient(fn, a)
    err = relative_error(a.grad, num_grad)
    print(f"  a.grad: {a.grad}, num: {num_grad}, rel_err: {err:.2e}")
    assert err < 1e-5, f"Mul: gradient error too large: {err}"
    print("  PASSED")


def test_matmul():
    print("Testing MatMul backward...")
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.randn(4, 2), requires_grad=True)
    y = (a @ b).sum()
    y.backward()

    def fn_a(x):
        return (x @ b).sum()
    num_a = numerical_gradient(fn_a, a)
    err_a = relative_error(a.grad, num_a)
    print(f"  a.grad rel_err: {err_a:.2e}")

    def fn_b(x):
        return (a @ x).sum()
    num_b = numerical_gradient(fn_b, b)
    err_b = relative_error(b.grad, num_b)
    print(f"  b.grad rel_err: {err_b:.2e}")
    assert err_a < 1e-5 and err_b < 1e-5, f"MatMul: gradient error too large"
    print("  PASSED")


def test_relu():
    print("Testing ReLU backward...")
    x = Tensor([1.0, -2.0, 3.0, -4.0], requires_grad=True)
    y = x.relu().sum()
    y.backward()

    expected = np.array([1.0, 0.0, 1.0, 0.0])
    assert np.allclose(x.grad, expected), f"ReLU: expected {expected}, got {x.grad}"
    print(f"  x.grad: {x.grad}, expected: {expected}")
    print("  PASSED")


def test_linear_layer():
    print("Testing Linear layer backward...")
    batch = Tensor(np.random.randn(2, 4), requires_grad=True)
    linear = Linear(4, 3)
    # Use he_normal init for reproducibility test
    init.he_normal_(linear.weight)

    out = linear(batch)
    loss = out.sum()
    loss.backward()

    # Check weight gradient via numerical method
    def fn_w(flat_w):
        linear.weight.data = flat_w.reshape(3, 4)
        o = linear(batch)
        return o.sum().item()

    num_w = numerical_gradient_wrapper(fn_w, linear.weight)
    err_w = relative_error(linear.weight.grad, num_w)
    print(f"  weight.grad rel_err: {err_w:.2e}")
    assert err_w < 1e-5, f"Linear: weight gradient error too large"

    # Check bias gradient
    def fn_b(flat_b):
        linear.bias.data = flat_b.reshape(3)
        o = linear(batch)
        return o.sum().item()

    num_b = numerical_gradient_wrapper(fn_b, linear.bias)
    err_b = relative_error(linear.bias.grad, num_b)
    print(f"  bias.grad rel_err: {err_b:.2e}")
    assert err_b < 1e-5, f"Linear: bias gradient error too large"
    print("  PASSED")


def numerical_gradient_wrapper(f, param):
    """Numerical gradient for a parameter."""
    grad = np.zeros_like(param.data)
    flat = param.data.ravel()
    flat_g = np.zeros_like(flat)
    for i in range(len(flat)):
        old = flat[i]
        flat[i] = old + 1e-5
        y1 = f(flat)
        flat[i] = old - 1e-5
        y2 = f(flat)
        flat_g[i] = (y1 - y2) / (2e-5)
        flat[i] = old
    return flat_g.reshape(param.data.shape)


def test_mlp_forward_backward():
    print("Testing MLP forward/backward (no crash + gradient flow)...")
    model = Sequential(
        Linear(10, 8),
        ReLU(),
        Linear(8, 3),
    )

    x = Tensor(np.random.randn(4, 10), requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Verify all parameters have gradients
    for p in model.parameters():
        assert p.grad is not None, f"Parameter {p.shape} has no gradient!"
        assert p.grad.shape == p.data.shape, f"Gradient shape mismatch: {p.grad.shape} vs {p.data.shape}"

    # Verify gradient is not all zeros
    weight_grad_norm = np.linalg.norm(model._modules["0"].weight.grad)
    assert weight_grad_norm > 0, "Weight gradient is zero!"
    print(f"  weight grad norm: {weight_grad_norm:.4f}")
    print("  PASSED")


def test_cross_entropy():
    print("Testing CrossEntropyLoss backward...")
    logits = Tensor(np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]), requires_grad=True)
    target = np.array([0, 1])  # class 0, class 1
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, target)
    loss.backward()

    # Compute numerical gradient
    def fn(x):
        l = loss_fn(x, target)
        return l  # Return Tensor

    num_grad = numerical_gradient(fn, logits)
    err = relative_error(logits.grad, num_grad)
    print(f"  loss: {loss.item():.4f}")
    print(f"  analytical grad:\n{logits.grad}")
    print(f"  numerical grad:\n{num_grad}")
    print(f"  rel_err: {err:.2e}")
    assert err < 1e-5, f"CrossEntropyLoss: gradient error too large: {err}"
    print("  PASSED")


def test_optimizer_sgd():
    print("Testing SGD optimizer...")
    p = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    from dlframe.parameter import Parameter
    from dlframe.optim import SGD

    param = Parameter(np.array([1.0, 2.0, 3.0]))
    param.grad = np.array([0.1, 0.2, 0.3])
    opt = SGD([param], lr=0.1)
    opt.step()

    expected = np.array([1.0, 2.0, 3.0]) - 0.1 * np.array([0.1, 0.2, 0.3])
    assert np.allclose(param.data, expected), f"SGD: {param.data} != {expected}"
    print(f"  updated: {param.data}, expected: {expected}")
    print("  PASSED")


def test_dataloader():
    print("Testing DataLoader...")
    from dlframe.data import DataLoader

    X = np.random.randn(100, 5)
    y = np.arange(100)
    loader = DataLoader(X, y, batch_size=32, shuffle=False)

    batches = list(loader)
    assert len(batches) == 4, f"Expected 4 batches, got {len(batches)}"
    assert batches[0][0].shape == (32, 5)
    assert batches[-1][0].shape == (4, 5)  # last batch: 100 - 3*32 = 4
    print(f"  n_batches: {len(batches)}, last_batch_shape: {batches[-1][0].shape}")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Gradient Check Tests")
    print("=" * 60)

    test_add()
    test_mul()
    test_matmul()
    test_relu()
    test_linear_layer()
    test_mlp_forward_backward()
    test_cross_entropy()
    test_optimizer_sgd()
    test_dataloader()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
