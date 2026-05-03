
# PyTorch Linear Regression

This module implements a simple linear regression model using PyTorch to predict continuous values based on a linear relationship.

## Overview

The model learns to predict the y-coordinate of points forming a straight line given their x-coordinate. The implementation demonstrates fundamental PyTorch concepts including:

- Synthetic data generation
- Model creation and training
- Loss function optimization
- Learning rate scheduling
- Data visualization

## Mathematical Foundation

The model learns the linear relationship: `y = wx + b`

Where:
- `w` is the weight (slope)
- `b` is the bias (y-intercept)
- Target: `w = 0.7`, `b = 0.3`

## Implementation Details

### Data Generation
- Creates a range of x values from 0.0 to 1.0 with 0.02 step size
- Computes corresponding y values using the target linear equation
- Splits data into 80% training and 20% testing sets

### Model Architecture
- Custom `LinearRegressionModel` class inheriting from `nn.Module`
- Single weight and bias parameters (both `nn.Parameter`)
- Forward pass: `y = weight * x + bias`

### Training Configuration
- **Loss Function**: L1Loss (Mean Absolute Error)
- **Optimizer**: SGD with initial learning rate of 0.01
- **Scheduler**: StepLR reducing LR by factor of 1/1.2 every 200 epochs
- **Epochs**: 2500 total training iterations
- **Logging**: Progress printed every 10 epochs

### Visualization
- Scatter plots showing training data, test data, and final predictions
- Loss curves displaying training and test MAE over epochs
- Clear differentiation between data types with color coding

## Usage

```bash
python pytorch_linear_regression.py
```

## Expected Output

The model should converge to learn parameters close to the target values:
- Learned weight: ≈ 0.7
- Learned bias: ≈ 0.3

Training typically shows decreasing loss over epochs with the learning rate scheduler helping achieve better convergence.

## Key Learning Concepts

1. **Parameter Initialization**: Using `torch.randn()` for weight and bias
2. **Manual Training Loop**: Implementing forward pass, loss calculation, backpropagation, and optimization
3. **Learning Rate Scheduling**: Using `StepLR` to adjust learning rate during training
4. **Evaluation**: Computing loss on both training and test sets
5. **Visualization**: Plotting data distributions and loss curves for analysis

