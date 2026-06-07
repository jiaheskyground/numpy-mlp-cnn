"""CIFAR-10 dataset loader."""

import numpy as np
import os
import urllib.request
import tarfile
import pickle


def download_cifar10(data_dir="./data/cifar10"):
    """Download CIFAR-10 dataset if not present."""
    os.makedirs(data_dir, exist_ok=True)
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    
    if not os.path.exists(tar_path):
        print(f"Downloading CIFAR-10 to {tar_path}...")
        urllib.request.urlretrieve(url, tar_path)
        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
    
    return os.path.join(data_dir, "cifar-10-batches-py")


def load_cifar10_batch(filepath):
    """Load a single CIFAR-10 batch file."""
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch[b'data'], np.array(batch[b'labels'])


def load_cifar10(data_dir="./data/cifar10", split='train'):
    """Load CIFAR-10 dataset.
    
    Args:
        data_dir: Directory containing CIFAR-10 data.
        split: 'train' or 'test'.
    
    Returns:
        (X, y) where X is shape (N, 3, 32, 32) and y is shape (N,).
    """
    cifar_dir = download_cifar10(data_dir)
    
    if split == 'train':
        X_list = []
        y_list = []
        for i in range(1, 6):
            batch_path = os.path.join(cifar_dir, f'data_batch_{i}')
            X, y = load_cifar10_batch(batch_path)
            X_list.append(X)
            y_list.append(y)
        
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
    
    else:  # test
        test_path = os.path.join(cifar_dir, 'test_batch')
        X, y = load_cifar10_batch(test_path)
    
    # Reshape from (N, 3072) to (N, 3, 32, 32)
    X = X.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = y.astype(np.int64)
    
    return X, y


def normalize_cifar10(X_train, X_test=None):
    """Normalize CIFAR-10 using training set statistics.
    
    Args:
        X_train: Training images (N, 3, 32, 32).
        X_test: Test images (optional).
    
    Returns:
        (X_train_norm, X_test_norm) or just X_train_norm.
    """
    # Compute per-channel mean and std from training set
    mean = X_train.mean(axis=(0, 2, 3), keepdims=True)  # (1, 3, 1, 1)
    std = X_train.std(axis=(0, 2, 3), keepdims=True)
    
    X_train_norm = (X_train - mean) / (std + 1e-5)
    
    if X_test is not None:
        X_test_norm = (X_test - mean) / (std + 1e-5)
        return X_train_norm, X_test_norm
    
    return X_train_norm
