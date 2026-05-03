
"""Train a multiclass classifier on synthetic blobs."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn

RANDOM_SEED = 42
NUM_CLASSES = 4
NUM_FEATURES = 2
N_SAMPLES = 1000
TEST_SIZE = 0.2
EPOCHS = 100

def create_blob_dataset():
    """Create and split the synthetic blob dataset for multiclass classification.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X, y = make_blobs(
        n_samples=N_SAMPLES,
        n_features=NUM_FEATURES,
        centers=NUM_CLASSES,
        cluster_std=1.5,
        random_state=RANDOM_SEED,
    )
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

class BlobClassifier(nn.Module):
    """A neural network for multiclass classification of clustered data points."""

    def __init__(self, input_features: int, hidden_units: int, output_features: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torch.eq(y_true, y_pred).sum().item() / len(y_true) * 100.0

def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = torch.from_numpy(np.column_stack([xx.ravel(), yy.ravel()])).float()
    model.eval()
    with torch.inference_mode():
        logits = model(grid)
        predictions = logits.softmax(dim=1).argmax(dim=1).numpy().reshape(xx.shape)
    plt.contourf(xx, yy, predictions, cmap=plt.cm.RdYlBu, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y.numpy(), cmap=plt.cm.RdYlBu, edgecolor="k", s=35)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

def main() -> None:
    """Main training and evaluation function for multiclass classification."""
    torch.manual_seed(RANDOM_SEED)

    # Validate configuration
    if EPOCHS <= 0:
        raise ValueError("EPOCHS must be positive")
    if NUM_CLASSES <= 1:
        raise ValueError("NUM_CLASSES must be greater than 1")

    X_train, X_test, y_train, y_test = create_blob_dataset()

    print(f"Dataset created with {N_SAMPLES} samples")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {NUM_FEATURES}, Classes: {NUM_CLASSES}")

    model = BlobClassifier(input_features=NUM_FEATURES, hidden_units=8, output_features=NUM_CLASSES)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            with torch.inference_mode():
                train_preds = logits.softmax(dim=1).argmax(dim=1)
                test_logits = model(X_test)
                test_preds = test_logits.softmax(dim=1).argmax(dim=1)
                train_acc = accuracy_fn(y_train, train_preds)
                test_acc = accuracy_fn(y_test, test_preds)
                test_loss = loss_fn(test_logits, y_test)
                print(
                    f"Epoch {epoch:4d} | Train loss: {loss.item():.5f}, "
                    f"Train acc: {train_acc:.2f}% | Test loss: {test_loss.item():.5f}, "
                    f"Test acc: {test_acc:.2f}%"
                )

    print("\nTraining completed. Generating decision boundary plots...")

    plt.figure(figsize=(6, 5))
    plt.title("Training Data Decision Boundaries")
    plot_decision_boundary(model, X_train, y_train)
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.title("Test Data Decision Boundaries")
    plot_decision_boundary(model, X_test, y_test)
    plt.show()

if __name__ == "__main__":
    main()
