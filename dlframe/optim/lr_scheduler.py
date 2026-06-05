"""Learning rate schedulers — StepLR and ExponentialLR."""


class StepLR:
    """Decays the learning rate by gamma every step_size epochs.

    Args:
        optimizer: Wrapped optimizer whose lr will be adjusted.
        step_size: Number of epochs between lr decays.
        gamma: Multiplicative factor of lr decay (default: 0.1).
    """

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.lr
        self._epoch = 0

    def step(self):
        """Call after each epoch to decay lr if needed."""
        self._epoch += 1
        if self._epoch % self.step_size == 0:
            self.optimizer.lr = self.base_lr * (self.gamma ** (self._epoch // self.step_size))

    def get_lr(self) -> float:
        return self.optimizer.lr

    def __repr__(self):
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma})"


class ExponentialLR:
    """Decays the learning rate by gamma every epoch: lr = base_lr * gamma^epoch.

    Args:
        optimizer: Wrapped optimizer whose lr will be adjusted.
        gamma: Multiplicative factor of lr decay per epoch (default: 0.95).
    """

    def __init__(self, optimizer, gamma: float = 0.95):
        self.optimizer = optimizer
        self.gamma = gamma
        self.base_lr = optimizer.lr
        self._epoch = 0

    def step(self):
        """Call after each epoch to decay lr."""
        self._epoch += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** self._epoch)

    def get_lr(self) -> float:
        return self.optimizer.lr

    def __repr__(self):
        return f"ExponentialLR(gamma={self.gamma})"
