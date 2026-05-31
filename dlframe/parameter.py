"""Parameter — a Tensor subclass marking trainable parameters."""

from dlframe.tensor import Tensor


class Parameter(Tensor):
    """A Tensor that is automatically registered as a module parameter.

    When assigned as an attribute of a Module, it is added to the module's
    _parameters dictionary. Its requires_grad is always True.
    """

    def __init__(self, data, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)

    def __repr__(self):
        return f"Parameter({self.data}, shape={self.shape})"
