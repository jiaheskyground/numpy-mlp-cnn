"""MNIST MLP training script skeleton for Week 1.

Downloads MNIST data and trains a simple MLP to verify the framework works end-to-end.
"""

import numpy as np
import urllib.request
import gzip
import os
import struct
import sys
sys.path.insert(0, ".")

from dlframe.tensor import Tensor
from dlframe.nn import Linear, ReLU, Sequential, init
from dlframe.nn.loss import CrossEntropyLoss
from dlframe.optim import SGD, Adam
from dlframe.data import DataLoader


# ── MNIST Data Loading ────────────────────────────────────────────────────

def download_mnist(data_dir="./data"):
    """Download MNIST dataset if not present."""
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for fn in files:
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            print(f"Downloading {fn}...")
            urllib.request.urlretrieve(base_url + fn, path)
    return data_dir


def load_mnist_images(path):
    """Load MNIST images from gzipped IDX file."""
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return data.astype(np.float64) / 255.0


def load_mnist_labels(path):
    """Load MNIST labels from gzipped IDX file."""
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


# ── Accuracy ──────────────────────────────────────────────────────────────

def compute_accuracy(model, X, y, batch_size=512):
    """Compute classification accuracy."""
    loader = DataLoader(X, y, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        logits = model(Tensor(X_batch))
        preds = np.argmax(logits.data, axis=1)
        correct += np.sum(preds == y_batch)
        total += len(y_batch)
    return correct / total


# ── Training Loop ─────────────────────────────────────────────────────────

def train_epoch(model, loader, loss_fn, optimizer):
    """Train for one epoch, return average loss."""
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        # Forward
        x = Tensor(X_batch)
        logits = model(x)
        loss = loss_fn(logits, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Step
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading MNIST data...")
    data_dir = download_mnist()
    X_train = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Test:  {X_test.shape}, {y_test.shape}")

    # Build model: 784 → 128 → 64 → 10
    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
    )

    # Initialize weights with He initialization (good for ReLU)
    for m in model._modules.values():
        if isinstance(m, Linear):
            init.he_normal_(m.weight)
            if m.bias is not None:
                m.bias.data = np.zeros_like(m.bias.data)

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(X_train, y_train, batch_size=64, shuffle=True)

    print("\nStarting training...")
    n_epochs = 5
    for epoch in range(n_epochs):
        avg_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        train_acc = compute_accuracy(model, X_train, y_train)
        test_acc = compute_accuracy(model, X_test, y_test)
        print(f"  Epoch {epoch+1}/{n_epochs} | loss: {avg_loss:.4f} | "
              f"train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")

    print(f"\nFinal test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
