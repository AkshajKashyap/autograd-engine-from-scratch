# Autograd Engine From Scratch

A NumPy-based autograd engine and tiny neural network library implementing reverse-mode differentiation, tensor operations, neural network layers, optimizers, gradient checks, and small training experiments from scratch.

## Goal

Build enough of a deep learning framework to understand how backpropagation, tensors, layers, optimizers, and training loops actually work internally.

## Current Status

The scalar and NumPy-backed tensor autograd engines are complete. Reusable neural network
modules, loss functions, and SGD, momentum, and Adam optimizers are implemented. Small
regression and XOR experiments now exercise the full training pipeline end to end.

## Examples

Run the deterministic experiments from the repository root:

```bash
python examples/train_regression.py
python examples/train_xor.py
```

The regression example demonstrates fitting noisy synthetic data with an MLP and mean squared
error. The XOR example demonstrates learning a nonlinear binary decision function with a
sigmoid output and binary cross-entropy. Both scripts print their losses and save a loss curve
and Markdown report under `reports/`.

Larger datasets and richer visualizations remain future milestones.
