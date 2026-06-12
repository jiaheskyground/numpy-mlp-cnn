"""DataLoader utility for batch iteration."""

import numpy as np


class DataLoader:
    """Simple data loader for batch iteration.
    
    Args:
        X: Input data array (N, ...)
        y: Target array (N,)
        batch_size: Size of each batch.
        shuffle: Whether to shuffle data.
    """
    
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
    
    def __iter__(self):
        """Iterate over batches."""
        indices = np.arange(self.n_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]
    
    def __len__(self):
        """Return number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size
