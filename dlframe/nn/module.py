"""Module base class — the foundation of all neural network layers."""

from collections import OrderedDict
from dlframe.parameter import Parameter
from dlframe.tensor import Tensor


class Module:
    """Base class for all neural network modules.

    Mimics PyTorch's nn.Module: tracks sub-modules and parameters,
    provides train/eval mode switching, and recursive parameter collection.

    Subclasses must implement forward().
    """

    def __init__(self):
        self._modules: OrderedDict[str, "Module"] = OrderedDict()
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._training: bool = True

    def __setattr__(self, name, value):
        # Auto-register Module and Parameter on assignment
        if isinstance(value, Module):
            self._modules[name] = value
            # Also register this module's parameters under its name prefix
        if isinstance(value, Parameter):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> list:
        """Recursively collect all Parameters, including those in sub-modules."""
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def zero_grad(self):
        """Set the gradient of all parameters to None."""
        for p in self.parameters():
            p.grad = None

    def train(self):
        """Set the module to training mode (recursive)."""
        self._training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self):
        """Set the module to evaluation mode (recursive)."""
        self._training = False
        for m in self._modules.values():
            m.eval()
        return self

    def forward(self, *inputs) -> Tensor:
        """Define the forward computation. Must be overridden by subclasses."""
        raise NotImplementedError

    def __call__(self, *inputs) -> Tensor:
        return self.forward(*inputs)

    def _flatten_modules(self) -> list:
        """Return all sub-modules in DFS order (for Sequential)."""
        result = []
        for m in self._modules.values():
            result.append(m)
            result.extend(m._flatten_modules())
        return result
