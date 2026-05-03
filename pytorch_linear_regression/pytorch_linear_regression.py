"""Train and evaluate a simple linear regression model using PyTorch."""

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
EPOCHS = 2500
LEARNING_RATE = 0.01
STEP_SIZE = 200
LR_GAMMA = 1 / 1.2


def create_linear_data(weight: float, bias: float, start: float, end: float, step: float):
    """Generate synthetic linear regression data."""
    X = torch.arange(start, end, step, dtype=torch.float32).unsqueeze(dim=1)
    y = weight * X + bias
    return X, y


def split_data(X: torch.Tensor, y: torch.Tensor, train_ratio: float):
    """Split data into training and testing sets.

    Args:
        X: Input features tensor
        y: Target values tensor
        train_ratio: Fraction of data to use for training (0 < train_ratio < 1)

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)

    Raises:
        ValueError: If train_ratio is not between 0 and 1
    """
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1")

    split_index = int(train_ratio * len(X))
    return X[:split_index], y[:split_index], X[split_index:], y[split_index:]


def plot_predictions(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    predictions: torch.Tensor | None = None,
) -> None:
    """Visualize training data, test data, and optional predictions."""
    plt.figure(figsize=(6, 4))
    plt.scatter(train_data, train_labels, color="tab:blue", s=20, label="Training data")
    plt.scatter(test_data, test_labels, color="tab:green", s=20, label="Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, color="tab:red", s=20, label="Predictions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear regression predictions")
    plt.legend()
    plt.show()


class LinearRegressionModel(nn.Module):
    """A simple linear regression model with learnable weight and bias."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


def train_model(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
) -> tuple[list[float], list[float], list[int], torch.Tensor]:
    train_losses = []
    test_losses = []
    epochs_logged = []
    test_predictions = torch.zeros_like(y_test)

    for epoch in range(1, epochs + 1):
        model.train()
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            test_predictions = model(X_test)
            test_loss = loss_fn(test_predictions, y_test)

        if epoch % 10 == 0 or epoch == 1:
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            epochs_logged.append(epoch)
            print(
                f"Epoch {epoch:4d} | Train MAE: {loss.item():.6f} | Test MAE: {test_loss.item():.6f}"
            )

    return train_losses, test_losses, epochs_logged, test_predictions


def main() -> None:
    """Main training and evaluation function."""
    torch.manual_seed(RANDOM_SEED)

    # Validate configuration
    if EPOCHS <= 0:
        raise ValueError("EPOCHS must be positive")
    if LEARNING_RATE <= 0:
        raise ValueError("LEARNING_RATE must be positive")

    X, y = create_linear_data(weight=0.7, bias=0.3, start=0.0, end=1.0, step=0.02)
    X_train, y_train, X_test, y_test = split_data(X, y, TRAIN_RATIO)

    print(f"Dataset size: {len(X)} samples")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    model = LinearRegressionModel()
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_GAMMA)

    print("\nStarting training for linear regression model...")
    train_losses, test_losses, epochs_logged, test_predictions = train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=EPOCHS,
    )

    print("\nLearned parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.item():.6f}")

    print("\nTarget parameters:")
    print("  weight: 0.700000")
    print("  bias: 0.300000")

    plot_predictions(X_train, y_train, X_test, y_test, predictions=test_predictions)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs_logged, train_losses, label="Train loss")
    plt.plot(epochs_logged, test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Training and Test Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
