
# SMO_ES: Support Vector Machine with Early Stopping

## Overview

The `SMO_ES` class provides a flexible interface for training Support Vector Machines (SVMs) with additional support for early stopping. This algorithm enhances the standard Sequential Minimal Optimization (SMO) by introducing early stopping based on specific objectives like accuracy or hinge loss.

## Features

- **Support for Linear and RBF Kernels**: Choose between linear and non-linear kernels to suit your dataset.
- **Early Stopping (ES)**: Early stopping can be enabled based on the specified criteria, such as accuracy or hinge loss, to prevent unnecessary computations and overfitting.
- **Customizable Stopping Criteria**: Define the tolerance for convergence and early stopping thresholds.
- **Time Limitation**: Set a maximum training time to ensure the algorithm doesn't run indefinitely.
- **Logging**: Log intermediate objectives to analyze the training process and convergence.

## Available Parameters

### `c (float)`
- Regularization parameter that controls the trade-off between achieving a low error on the training set and minimizing model complexity. Higher values imply stricter margins, possibly increasing the risk of overfitting.

### `tolerance (float)`
- Convergence tolerance. The optimization process halts when the change between iterations is smaller than this threshold.

### `kernel (str)`
- Specifies the kernel type used in the SVM:
  - `'linear'`: Linear kernel (default).
  - `'rbf'`: Radial Basis Function (Gaussian) kernel for non-linear problems.

### `gamma (float or str)`
- Kernel coefficient for the RBF kernel. This can either be:
  - A float (direct gamma value).
  - `'auto'`: Sets `gamma = 1 / n_features`, where `n_features` is the number of input features.
  - `'scale'`: Default setting, `gamma = 1 / (n_features * X.var())`, where `X.var()` is the feature variance.

### `max_iter (int)`
- Maximum number of iterations for the optimization process. Set to:
  - A positive integer to specify a limit.
  - `-1` to disable the iteration limit and allow the process to run until convergence or early stopping.

### `r (int)`
- Number of iterations after which early stopping objectives are evaluated.

### `patience (int)`
- Number of iterations to wait without improvement before early stopping is triggered. Set to:
  - A positive integer to specify the patience threshold.
  - `-1` to disable early stopping.

### `early_stop (bool)`
- Enables early stopping when set to `True`. Training will halt if the stopping criteria are met.

### `es_tolerance (float)`
- The threshold for early stopping based on improvement. If the change in the objective is smaller than this value, early stopping is triggered.

### `es_objective (str)`
- Specifies the optimization objective for early stopping:
  - `'acc'`: Accuracy on the validation set.
  - `'hinge'`: Hinge loss on the validation set.

### `time_lim (float, optional)`
- Maximum allowed training time in seconds. If the time limit is reached, the training process stops.

### `log_objectives (bool)`
- Set to `True` to log intermediate objective values (accuracy or hinge loss). This is helpful for tracking the training progress.

## Example Usage

```python
from smo_es import SMO_ES

# Define your SVM with early stopping
model = SMO_ES(
    c=1.0,
    tolerance=1e-5,
    kernel='rbf',
    gamma='scale',
    max_iter=1000,
    r=10,
    patience=5,
    early_stop=True,
    es_tolerance=1e-3,
    es_objective='acc',
    log_objectives=True
)

# Fit the model on your data
model.fit(X_train, y_train)

# Access the results
print("Final Accuracy:", model.accuracy)
```

## Installation

To install the `SMO_ES` algorithm, simply clone the repository and install the dependencies:

```bash
git clone https://github.com/your-repo/smo_es.git
cd smo_es
pip install -r requirements.txt
```

## Dependencies

- `numpy`
- `scipy`
- `sklearn`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
