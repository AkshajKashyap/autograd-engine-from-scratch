# Autograd Engine From Scratch

A NumPy-based autograd engine and tiny neural network library implementing reverse-mode differentiation, tensor operations, neural network layers, optimizers, gradient checks, and small training experiments from scratch.

## Goal

Build enough of a deep learning framework to understand how backpropagation, tensors, layers, optimizers, and training loops actually work internally.

## Current Status

The scalar and NumPy-backed tensor autograd engines are complete. Reusable neural network
modules, loss functions, and SGD, momentum, and Adam optimizers are implemented. Small
regression, XOR, and multiclass spiral experiments now exercise the full training pipeline.

## Examples

Run the deterministic experiments from the repository root:

```bash
python examples/train_regression.py
python examples/train_xor.py
python examples/train_spiral.py
```

The regression example demonstrates fitting noisy synthetic data with an MLP and mean squared
error. The XOR example demonstrates learning a nonlinear binary decision function with a
sigmoid output and binary cross-entropy. Both scripts print their losses and save a loss curve
and Markdown report under `reports/`.

The spiral example demonstrates three-class nonlinear classification with softmax
cross-entropy and saves both a loss curve and decision-boundary visualization.

MNIST, larger image datasets, and richer visualizations remain future milestones.
