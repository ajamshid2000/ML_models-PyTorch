
# PyTorch Machine Learning Models Collection

A comprehensive collection of PyTorch implementations for fundamental machine learning tasks, including regression, classification, and computer vision. This repository serves as an educational resource for understanding neural network architectures and training methodologies.

## Project Structure

```
ML_models-PyTorch-/
├── pytorch_linear_regression/
│   ├── pytorch_linear_regression.py    # Linear regression with manual training loop
│   └── README.md                       # Detailed linear regression documentation
├── pytorch_classification/
│   ├── binary_classification_model.py  # Binary classifier on synthetic circle data
│   └── multiclass_classification.py    # Multiclass classifier on synthetic blobs
└── pytorch_computer_vision/
    ├── image_classification_model.py          # Fully connected FashionMNIST classifier
    └── image_classification_model_with_CNN.py # Convolutional FashionMNIST classifier
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- torchvision
- scikit-learn
- matplotlib
- numpy

### Installation

Install the required dependencies using pip:

```bash
pip install torch torchvision scikit-learn matplotlib numpy
```

### Usage

Each model can be run independently. Navigate to the respective directory and execute the Python script:

```bash
# Linear Regression
cd pytorch_linear_regression
python pytorch_linear_regression.py

# Binary Classification
cd ../pytorch_classification
python binary_classification_model.py

# Multiclass Classification
python multiclass_classification.py

# Computer Vision Models
cd ../pytorch_computer_vision
python image_classification_model.py
python image_classification_model_with_CNN.py
```

## Model Descriptions

### Linear Regression
- **File**: `pytorch_linear_regression/pytorch_linear_regression.py`
- **Objective**: Predict continuous values using a simple linear relationship
- **Features**:
  - Synthetic data generation (`y = 0.7x + 0.3`)
  - Manual training loop with learning rate scheduling
  - Loss curve visualization
  - Train/test split evaluation

### Binary Classification
- **File**: `pytorch_classification/binary_classification_model.py`
- **Objective**: Classify points into two categories using concentric circles dataset
- **Architecture**: 3-layer fully connected network
- **Features**:
  - Decision boundary visualization
  - BCEWithLogitsLoss for binary classification
  - Accuracy tracking during training

### Multiclass Classification
- **File**: `pytorch_classification/multiclass_classification.py`
- **Objective**: Classify points into multiple categories using synthetic blob clusters
- **Architecture**: 3-layer fully connected network with ReLU activations
- **Features**:
  - Multi-class decision boundary visualization
  - CrossEntropyLoss for multiclass classification
  - 4-class classification problem

### Computer Vision Models

#### Fully Connected Model
- **File**: `pytorch_computer_vision/image_classification_model.py`
- **Dataset**: FashionMNIST (28x28 grayscale images)
- **Architecture**: Single hidden layer fully connected network
- **Features**:
  - Data loading with torchvision
  - Batch processing with DataLoader
  - GPU acceleration support

#### Convolutional Neural Network
- **File**: `pytorch_computer_vision/image_classification_model_with_CNN.py`
- **Dataset**: FashionMNIST
- **Architecture**: 3-layer CNN with max pooling
- **Features**:
  - Convolutional feature extraction
  - Spatial hierarchy learning
  - Improved performance over fully connected model

## Key Features

- **Modular Design**: Each model is self-contained with clear separation of concerns
- **Educational Focus**: Comprehensive comments and docstrings for learning
- **Visualization**: Rich plotting capabilities for understanding model behavior
- **GPU Support**: Automatic device detection (CUDA/MPS/CPU)
- **Reproducibility**: Fixed random seeds for consistent results
- **Best Practices**: Modern PyTorch conventions and patterns

## Performance Metrics

| Model | Dataset | Accuracy | Training Time |
|-------|---------|----------|---------------|
| Linear Regression | Synthetic | N/A (MAE loss) | ~2 seconds |
| Binary Classification | Circles | ~99% | ~5 seconds |
| Multiclass Classification | Blobs | ~98% | ~2 seconds |
| FC FashionMNIST | FashionMNIST | ~85% | ~30 seconds |
| CNN FashionMNIST | FashionMNIST | ~90% | ~45 seconds |

*Note: Performance may vary based on hardware and random initialization*


## Additional Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

