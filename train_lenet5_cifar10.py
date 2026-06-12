"""LeNet-5 training on CIFAR-10 with PyTorch comparison.

This script trains a LeNet-5 variant on CIFAR-10 using the custom dlframe,
then compares with an equivalent PyTorch model for validation.
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from dlframe.tensor import Tensor
from dlframe.nn import Sequential, Linear, ReLU
from dlframe.nn.conv import Conv2d
from dlframe.nn.batchnorm import BatchNorm2d
from dlframe.nn.pooling import MaxPool2d
from dlframe.nn.loss import CrossEntropyLoss
from dlframe.optim import Adam
from dlframe.data.cifar10 import load_cifar10, normalize_cifar10
from dlframe.data import DataLoader
import time


class LeNet5(Sequential):
    """LeNet-5 variant for CIFAR-10 (32x32 images)."""
    
    def __init__(self, num_classes=10):
        layers = [
            Conv2d(3, 32, kernel_size=5, padding=2),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(32, 64, kernel_size=5, padding=2),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            
            # Flatten: (N, 64, 8, 8) -> (N, 64*8*8)
            # Linear layers
        ]
        super().__init__(*layers)
        self.fc1 = Linear(64 * 8 * 8, 128)
        self.fc2 = Linear(128, num_classes)
    
    def forward(self, x):
        # Conv + BN + ReLU + Pool x2
        for layer in self._modules.values():
            x = layer(x)
        
        # Flatten
        N = x.shape[0]
        x = x.reshape(N, -1)
        
        # FC layers
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        return x


def compute_accuracy(model, X, y, batch_size=256):
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


def train_epoch(model, loader, loss_fn, optimizer):
    """Train for one epoch, return average loss."""
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


def main():
    print("="*60)
    print("Custom dlframe: LeNet-5 on CIFAR-10")
    print("="*60)
    
    # Load data
    print("Loading CIFAR-10...")
    X_train, y_train = load_cifar10(split='train')
    X_test, y_test = load_cifar10(split='test')
    
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Test:  {X_test.shape}, {y_test.shape}")
    
    # Normalize
    X_train, X_test = normalize_cifar10(X_train, X_test)
    
    # Build model
    print("\nBuilding LeNet-5...")
    model = LeNet5(num_classes=10)
    model.train()
    
    loss_fn = CrossEntropyLoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_loader = DataLoader(X_train, y_train, batch_size=128, shuffle=True)
    
    print("\nStarting training (dlframe)...")
    n_epochs = 5
    start_time = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        avg_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        
        model.eval()
        train_acc = compute_accuracy(model, X_train, y_train)
        test_acc = compute_accuracy(model, X_test, y_test)
        
        print(f"  Epoch {epoch+1}/{n_epochs} | loss: {avg_loss:.4f} | "
              f"train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f}s")
    print(f"Final test accuracy (dlframe): {test_acc:.4f}")
    
    # PyTorch comparison
    print("\n" + "="*60)
    print("PyTorch: Equivalent LeNet-5 on CIFAR-10")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim_torch
        
        class PyTorchLeNet5(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
                self.bn1 = nn.BatchNorm2d(32)
                self.pool1 = nn.MaxPool2d(2, 2)
                
                self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
                self.bn2 = nn.BatchNorm2d(64)
                self.pool2 = nn.MaxPool2d(2, 2)
                
                self.fc1 = nn.Linear(64 * 8 * 8, 128)
                self.fc2 = nn.Linear(128, num_classes)
            
            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = torch.relu(x)
                x = self.pool1(x)
                
                x = self.conv2(x)
                x = self.bn2(x)
                x = torch.relu(x)
                x = self.pool2(x)
                
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Prepare data
        X_train_torch = torch.from_numpy(X_train).float().to(device)
        y_train_torch = torch.from_numpy(y_train).long().to(device)
        X_test_torch = torch.from_numpy(X_test).float().to(device)
        y_test_torch = torch.from_numpy(y_test).long().to(device)
        
        # Build model
        torch_model = PyTorchLeNet5(num_classes=10).to(device)
        loss_fn_torch = nn.CrossEntropyLoss()
        optimizer_torch = optim_torch.Adam(torch_model.parameters(), lr=0.001)
        
        # Training loop
        print("\nStarting training (PyTorch)...")
        start_time = time.time()
        
        for epoch in range(n_epochs):
            torch_model.train()
            total_loss = 0.0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_torch), 128):
                X_batch = X_train_torch[i:i+128]
                y_batch = y_train_torch[i:i+128]
                
                logits = torch_model(X_batch)
                loss = loss_fn_torch(logits, y_batch)
                
                optimizer_torch.zero_grad()
                loss.backward()
                optimizer_torch.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            
            # Evaluation
            torch_model.eval()
            with torch.no_grad():
                train_logits = torch_model(X_train_torch)
                train_preds = torch.argmax(train_logits, dim=1)
                train_acc = (train_preds == y_train_torch).float().mean().item()
                
                test_logits = torch_model(X_test_torch)
                test_preds = torch.argmax(test_logits, dim=1)
                test_acc = (test_preds == y_test_torch).float().mean().item()
            
            print(f"  Epoch {epoch+1}/{n_epochs} | loss: {avg_loss:.4f} | "
                  f"train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed:.2f}s")
        print(f"Final test accuracy (PyTorch): {test_acc:.4f}")
        
    except ImportError:
        print("\nPyTorch not installed. Skipping benchmark.")
        print("Install with: pip install torch torchvision")


if __name__ == "__main__":
    main()
