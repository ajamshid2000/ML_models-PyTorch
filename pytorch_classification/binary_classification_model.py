
"""Train a binary classification model on synthetic circle data."""

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_SAMPLES = 1000
EPOCHS = 1000

def create_circle_dataset(n_samples: int, test_size: float, random_state: int):
    """Create and split the synthetic circle dataset.

    Args:
        n_samples: Total number of samples to generate
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")

    X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=random_state)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

class CircleClassifier(nn.Module):
    """A simple neural network for binary classification of circular data patterns."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze()

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.eq(y_true, y_pred).sum().item() / len(y_true) * 100.0

def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, steps=200),
        torch.linspace(y_min, y_max, steps=200),
        indexing="xy",
    )
    grid = torch.stack([xx.ravel(), yy.ravel()], dim=1)
    model.eval()
    with torch.inference_mode():
        logits = model(grid)
        predictions = torch.round(torch.sigmoid(logits)).reshape(xx.shape).numpy()
    plt.contourf(xx.numpy(), yy.numpy(), predictions, cmap=plt.cm.RdYlBu, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor="k", s=35)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

def main() -> None:
    """Main training and evaluation function for binary classification."""
    torch.manual_seed(RANDOM_SEED)

    # Validate configuration
    if EPOCHS <= 0:
        raise ValueError("EPOCHS must be positive")
    if N_SAMPLES <= 0:
        raise ValueError("N_SAMPLES must be positive")

    X_train, X_test, y_train, y_test = create_circle_dataset(N_SAMPLES, TEST_SIZE, RANDOM_SEED)

    print(f"Dataset created with {N_SAMPLES} samples")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {X_train.shape[1]}, Classes: {len(torch.unique(y_train))}")

    model = CircleClassifier()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 1:
            with torch.inference_mode():
                train_preds = torch.round(torch.sigmoid(model(X_train)))
                test_preds = torch.round(torch.sigmoid(model(X_test)))
                train_acc = accuracy_fn(y_true=y_train, y_pred=train_preds)
                test_acc = accuracy_fn(y_true=y_test, y_pred=test_preds)
                test_loss = loss_fn(model(X_test), y_test)
                print(
                    f"Epoch {epoch:4d} | Train loss: {loss.item():.5f}, "
                    f"Train acc: {train_acc:.2f}% | Test loss: {test_loss.item():.5f}, "
                    f"Test acc: {test_acc:.2f}%"
                )

    print("\nTraining completed. Generating decision boundary plots...")

    plt.figure(figsize=(6, 5))
    plt.title("Decision Boundary on Training Data")
    plot_decision_boundary(model, X_train, y_train)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.title("Decision Boundary on Test Data")
    plot_decision_boundary(model, X_test, y_test)
    plt.show()

if __name__ == "__main__":
    main()
