
"""Train a simple fully connected FashionMNIST model."""

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 3
DATA_DIR = "./data/FashionMNIST"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(batch_size: int):
    """Load FashionMNIST dataset with train/test split.

    Args:
        batch_size: Number of samples per batch

    Returns:
        Tuple of (train_loader, test_loader)
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    transform = ToTensor()
    train_data = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class FashionMNISTModel(nn.Module):
    """A fully connected neural network for FashionMNIST image classification."""

    def __init__(self, input_size: int, hidden_units: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct / len(y_true) * 100.0

def train_step(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        predictions = model(X)
        loss = loss_fn(predictions, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy_fn(y, predictions.argmax(dim=1))
    return total_loss / len(loader), total_acc / len(loader)

def test_step(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            loss = loss_fn(predictions, y)
            total_loss += loss.item()
            total_acc += accuracy_fn(y, predictions.argmax(dim=1))
    return total_loss / len(loader), total_acc / len(loader)

def main() -> None:
    """Main training and evaluation function for FashionMNIST classification."""
    torch.manual_seed(RANDOM_SEED)

    # Validate configuration
    if EPOCHS <= 0:
        raise ValueError("EPOCHS must be positive")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive")

    train_loader, test_loader = load_data(BATCH_SIZE)
    model = FashionMNISTModel(input_size=28 * 28, hidden_units=10, output_size=10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print(f"Using device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    start_time = time.perf_counter()
    print("\nStarting training...")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer)
        test_loss, test_acc = test_step(model, test_loader, loss_fn)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_loss:.5f}, Train acc: {train_acc:.2f}% | "
            f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%"
        )
    duration = time.perf_counter() - start_time
    print(f"\nTraining completed in {duration:.2f} seconds")

if __name__ == "__main__":
    main()
