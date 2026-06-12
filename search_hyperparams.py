"""Hyperparameter random search for optimal MLP on MNIST.

Strategy (Bergstra & Bengio 2012): random search is more efficient than grid
search when only a few hyperparameters dominate performance. We sample 24
configurations, evaluate each with 15-epoch quick training, then run full
30-epoch training on the top-3 candidates.
"""

import numpy as np
import urllib.request
import gzip
import os
import struct
import sys
import json
from itertools import product
sys.path.insert(0, ".")

from dlframe.tensor import Tensor
from dlframe.nn import Linear, ReLU, Sequential, init, Dropout
from dlframe.nn.loss import CrossEntropyLoss
from dlframe.optim import Adam, SGD, StepLR, ExponentialLR
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


def train_val_split(X, y, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ── Metrics ───────────────────────────────────────────────────────────────

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
    for m in model._modules.values():
        if isinstance(m, Linear):
            init.he_normal_(m.weight)
            if m.bias is not None:
                m.bias.data = np.zeros_like(m.bias.data)
    return model


# ── Training ──────────────────────────────────────────────────────────────

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


def evaluate_config(config, X_train, y_train, X_val, y_val, n_epochs=15):
    """Quick evaluation of a single hyperparameter configuration."""
    model = build_mlp(config["hidden"], dropout=config["dropout"])
    loss_fn = CrossEntropyLoss()

    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=config["lr"],
                         weight_decay=config.get("weight_decay", 0))
    else:
        optimizer = SGD(model.parameters(), lr=config["lr"],
                        momentum=config.get("momentum", 0.9),
                        weight_decay=config.get("weight_decay", 0))

    if config["scheduler"] == "steplr":
        scheduler = StepLR(optimizer, step_size=config["step_size"],
                          gamma=config["gamma"])
    else:
        scheduler = ExponentialLR(optimizer, gamma=config["gamma"])

    train_loader = DataLoader(X_train, y_train, batch_size=config["batch_size"], shuffle=True)

    best_val_acc = 0.0
    best_epoch = 0
    val_accs = []

    for epoch in range(n_epochs):
        avg_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        val_acc = compute_accuracy(model, X_val, y_val, batch_size=512)
        scheduler.step()
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_val_acc": val_acc,
        "val_accs": val_accs,
    }


# ── Search Space ──────────────────────────────────────────────────────────

ARCHITECTURES = [
    [128, 64],
    [256, 128],
    [512, 256],
    [256, 128, 64],
    [512, 256, 128],
    [512, 256, 128, 64],
    [1024, 512],
    [1024, 512, 256],
]

DROPOUT_RATES = [0.0, 0.1, 0.2, 0.25, 0.3, 0.5]
LEARNING_RATES = [0.0005, 0.001, 0.0015, 0.002, 0.003]
BATCH_SIZES = [64, 128, 256]
OPTIMIZERS = ["adam", "sgd"]
SCHEDULERS = ["steplr", "exponential"]
STEP_SIZES = [5, 8, 10]
GAMMAS = [0.3, 0.5, 0.7]
WEIGHT_DECAYS = [0.0, 1e-4, 1e-3]


def sample_random_configs(n=24, seed=42):
    """Sample n random configurations from the search space."""
    rng = np.random.RandomState(seed)
    configs = []

    for i in range(n):
        hidden = ARCHITECTURES[rng.randint(0, len(ARCHITECTURES))]
        dropout = DROPOUT_RATES[rng.randint(0, len(DROPOUT_RATES))]
        lr = LEARNING_RATES[rng.randint(0, len(LEARNING_RATES))]
        batch_size = BATCH_SIZES[rng.randint(0, len(BATCH_SIZES))]
        opt = OPTIMIZERS[rng.randint(0, len(OPTIMIZERS))]
        sched = SCHEDULERS[rng.randint(0, len(SCHEDULERS))]
        step_size = STEP_SIZES[rng.randint(0, len(STEP_SIZES))]
        gamma = GAMMAS[rng.randint(0, len(GAMMAS))]
        wd = WEIGHT_DECAYS[rng.randint(0, len(WEIGHT_DECAYS))]

        config = {
            "id": i,
            "hidden": hidden,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": opt,
            "scheduler": sched,
            "step_size": step_size,
            "gamma": gamma,
            "weight_decay": wd,
            "momentum": 0.9 if opt == "sgd" else 0,
        }
        configs.append(config)

    return configs


# ── Full Training for Top Configs ─────────────────────────────────────────

def train_full(model, X_train, y_train, X_val, y_val, X_test, y_test,
               loss_fn, optimizer, scheduler, n_epochs, batch_size):
    """Full training with detailed logging."""
    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_acc": [], "lr": []}

    for epoch in range(n_epochs):
        avg_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        val_acc = compute_accuracy(model, X_val, y_val)
        scheduler.step()

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(scheduler.get_lr())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

        print(f"  Epoch {epoch+1:3d}/{n_epochs} | loss: {avg_loss:.4f} | "
              f"val acc: {val_acc:.4f} | lr: {scheduler.get_lr():.2e}")

    test_acc = compute_accuracy(model, X_test, y_test)
    return best_val_acc, best_epoch, test_acc, history


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading MNIST data...")
    data_dir = download_mnist()
    X_train_full = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train_full = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    X_train, y_train, X_val, y_val = train_val_split(X_train_full, y_train_full)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ── Phase 1: Random Search (24 configs × 15 epochs) ────────────────────

    N_SEARCH = 24
    QUICK_EPOCHS = 15

    print(f"\n{'='*60}")
    print(f"Phase 1: Random Search — {N_SEARCH} configs × {QUICK_EPOCHS} epochs")
    print(f"{'='*60}")

    configs = sample_random_configs(N_SEARCH, seed=42)
    results = []

    for cfg in configs:
        name = (f"#{cfg['id']:02d} {cfg['optimizer'].upper()} "
                f"hidden={cfg['hidden']} dropout={cfg['dropout']} "
                f"lr={cfg['lr']:.0e} batch={cfg['batch_size']} "
                f"{cfg['scheduler']}(step={cfg['step_size']},γ={cfg['gamma']})")
        print(f"\n[{name}]")

        result = evaluate_config(cfg, X_train, y_train, X_val, y_val, n_epochs=QUICK_EPOCHS)
        result["config"] = cfg
        result["name"] = name
        results.append(result)

        print(f"  Best val: {result['best_val_acc']:.4f} @ epoch {result['best_epoch']}")

    # Sort by best validation accuracy
    results.sort(key=lambda r: r["best_val_acc"], reverse=True)

    print(f"\n{'='*60}")
    print("Phase 1 Results (sorted by best val acc)")
    print(f"{'='*60}")
    print(f"{'Rank':<5} {'Best Val':>10} {'Config'}")
    print("-" * 80)
    for rank, r in enumerate(results):
        cfg = r["config"]
        desc = (f"{cfg['optimizer']} hidden={cfg['hidden']} "
                f"dropout={cfg['dropout']} lr={cfg['lr']:.0e} "
                f"batch={cfg['batch_size']} {cfg['scheduler']}"
                f"(step={cfg['step_size']},γ={cfg['gamma']}) wd={cfg['weight_decay']}")
        print(f"{rank+1:<5} {r['best_val_acc']:.4f}     {desc}")

    # ── Phase 2: Full training on top-3 configs ─────────────────────────────

    TOP_K = 3
    FULL_EPOCHS = 30

    print(f"\n{'='*60}")
    print(f"Phase 2: Full Training — Top {TOP_K} configs × {FULL_EPOCHS} epochs")
    print(f"{'='*60}")

    top_results = []

    for rank in range(TOP_K):
        cfg = results[rank]["config"]
        name = (f"Top-{rank+1} {cfg['optimizer'].upper()} "
                f"hidden={cfg['hidden']} dropout={cfg['dropout']} "
                f"lr={cfg['lr']:.0e} batch={cfg['batch_size']}")
        print(f"\n[{name}]")

        model = build_mlp(cfg["hidden"], dropout=cfg["dropout"])
        loss_fn = CrossEntropyLoss()

        if cfg["optimizer"] == "adam":
            optimizer = Adam(model.parameters(), lr=cfg["lr"],
                             weight_decay=cfg.get("weight_decay", 0))
        else:
            optimizer = SGD(model.parameters(), lr=cfg["lr"],
                            momentum=cfg.get("momentum", 0.9),
                            weight_decay=cfg.get("weight_decay", 0))

        if cfg["scheduler"] == "steplr":
            scheduler = StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"])
        else:
            scheduler = ExponentialLR(optimizer, gamma=cfg["gamma"])

        best_val, best_epoch, test_acc, history = train_full(
            model, X_train, y_train, X_val, y_val, X_test, y_test,
            loss_fn, optimizer, scheduler, FULL_EPOCHS, cfg["batch_size"],
        )

        top_results.append({
            "config": cfg,
            "name": name,
            "best_val_acc": best_val,
            "best_epoch": best_epoch,
            "test_acc": test_acc,
            "history": history,
        })

        print(f"  Best val: {best_val:.4f} @ epoch {best_epoch} | Test: {test_acc:.4f}")

    # ── Final Summary ──────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("Final Results — Top Configurations after Full Training")
    print(f"{'='*60}")
    print(f"{'Model':<55} {'Best Val':>10} {'Test Acc':>10}")
    print("-" * 78)
    for r in top_results:
        print(f"{r['name']:<55} {r['best_val_acc']:.4f}     {r['test_acc']:.4f}")

    # Identify overall best
    best = max(top_results, key=lambda r: r["test_acc"])
    print(f"\nOptimal configuration: {best['name']}")
    print(f"  Test accuracy: {best['test_acc']:.4f}")

    # Save results to JSON for report
    output = {
        "search_results": [
            {
                "rank": i + 1,
                "config": {k: v for k, v in r["config"].items() if k != "id"},
                "best_val_acc_15ep": float(r["best_val_acc"]),
                "best_epoch_15ep": r["best_epoch"],
            }
            for i, r in enumerate(results)
        ],
        "top_results": [
            {
                "name": r["name"],
                "config": {k: v for k, v in r["config"].items() if k != "id"},
                "best_val_acc": float(r["best_val_acc"]),
                "best_epoch": r["best_epoch"],
                "test_acc": float(r["test_acc"]),
            }
            for r in top_results
        ],
        "optimal_test_acc": float(best["test_acc"]),
    }

    with open("search_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nResults saved to search_results.json")


if __name__ == "__main__":
    main()
