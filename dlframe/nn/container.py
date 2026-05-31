"""Sequential container — chains modules in order."""

from dlframe.nn.module import Module


class Sequential(Module):
    """A sequential container that chains modules in order.

    Args:
        *modules: Variable number of Module instances.
    """

    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self._modules[str(i)] = m

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x

    def __repr__(self):
        items = "\n  ".join(f"({k}): {repr(v)}" for k, v in self._modules.items())
        return f"Sequential(\n  {items}\n)"

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)
