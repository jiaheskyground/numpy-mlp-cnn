"""MNIST MLP training — multi-architecture comparison with Dropout & LR scheduling."""

import numpy as np
import urllib.request
import gzip
import os
import struct
import sys
sys.path.insert(0, ".")

from dlframe.tensor import Tensor
from dlframe.nn import Linear, ReLU, Sequential, init, Dropout
from dlframe.nn.loss import CrossEntropyLoss
from dlframe.optim import Adam, StepLR
from dlframe.data import DataLoader


# ── MNIST Data Loading ────────────────────────────────────────────────────

def download_mnist(data_dir="./data"):
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
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return data.astype(np.float64) / 255.0


def load_mnist_labels(path):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


# ── Data preprocessing ────────────────────────────────────────────────────

def train_val_split(X, y, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ── Accuracy ──────────────────────────────────────────────────────────────

def compute_accuracy(model, X, y, batch_size=512):
    was_training = model._training
    model.eval()
    loader = DataLoader(X, y, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        logits = model(Tensor(X_batch))
        preds = np.argmax(logits.data, axis=1)
        correct += np.sum(preds == y_batch)
        total += len(y_batch)
    if was_training:
        model.train()
    return correct / total


# ── Model building ────────────────────────────────────────────────────────

def build_mlp(hidden_sizes, dropout=0.0):
    """Build an MLP Sequential model with optional Dropout.

    Args:
        hidden_sizes: list of hidden layer widths, e.g. [128, 64].
        dropout: Dropout probability (0 = no dropout).
    """
    layers = []
    in_dim = 784
    for h_dim in hidden_sizes:
        layers.append(Linear(in_dim, h_dim))
        layers.append(ReLU())
        if dropout > 0:
            layers.append(Dropout(p=dropout))
        in_dim = h_dim
    layers.append(Linear(in_dim, 10))

    model = Sequential(*layers)

    # He initialization for all Linear layers
    for m in model._modules.values():
        if isinstance(m, Linear):
            init.he_normal_(m.weight)
            if m.bias is not None:
                m.bias.data = np.zeros_like(m.bias.data)

    return model


# ── Training loop ─────────────────────────────────────────────────────────

def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        x = Tensor(X_batch)
        logits = model(x)
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def train_model(model, train_loader, val_X, val_y, loss_fn, optimizer,
                scheduler, n_epochs, model_name="MLP"):
    """Train one model, return training history."""
    history = {"epoch": [], "train_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(n_epochs):
        avg_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        val_acc = compute_accuracy(model, val_X, val_y)
        scheduler.step()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        lr = scheduler.get_lr()
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | loss: {avg_loss:.4f} | "
              f"val acc: {val_acc:.4f} | lr: {lr:.2e}")

    return history, best_val_acc


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading MNIST data...")
    data_dir = download_mnist()
    X_train_full = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train_full = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    print(f"  Train: {X_train_full.shape}, {y_train_full.shape}")
    print(f"  Test:  {X_test.shape}, {y_test.shape}")

    # Split train into train/val
    X_train, y_train, X_val, y_val = train_val_split(X_train_full, y_train_full, val_ratio=0.1)
    print(f"  After split — train: {X_train.shape}, val: {X_val.shape}")

    # ── Experiment configurations ──────────────────────────────────────────

    configs = [
        {
            "name": "MLP-128-64 (baseline, no dropout)",
            "hidden": [128, 64],
            "dropout": 0.0,
        },
        {
            "name": "MLP-128-64 + Dropout 0.25",
            "hidden": [128, 64],
            "dropout": 0.25,
        },
        {
            "name": "MLP-256-128 + Dropout 0.25",
            "hidden": [256, 128],
            "dropout": 0.25,
        },
        {
            "name": "MLP-256-128-64 + Dropout 0.25",
            "hidden": [256, 128, 64],
            "dropout": 0.25,
        },
    ]

    n_epochs = 20
    batch_size = 128
    results = {}

    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Training: {cfg['name']}")
        print(f"{'='*60}")

        model = build_mlp(cfg["hidden"], dropout=cfg["dropout"])
        loss_fn = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=8, gamma=0.5)

        train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)

        history, best_val = train_model(
            model, train_loader, X_val, y_val,
            loss_fn, optimizer, scheduler, n_epochs,
            model_name=cfg["name"],
        )

        # Final test accuracy
        test_acc = compute_accuracy(model, X_test, y_test)
        print(f"  Best val acc: {best_val:.4f} | Test acc: {test_acc:.4f}")

        results[cfg["name"]] = {
            "best_val_acc": best_val,
            "test_acc": test_acc,
            "history": history,
        }

    # ── Results summary ────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"{'Model':<38} {'Best Val':>10} {'Test Acc':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<38} {r['best_val_acc']:.4f}     {r['test_acc']:.4f}")


if __name__ == "__main__":
    main()
