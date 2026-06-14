# Project Summary

## Overview

This project builds a small automatic differentiation and neural network library from scratch
using Python and NumPy. It starts with scalar computation graphs, extends the same ideas to
multidimensional tensors, and finishes with reusable models, optimizers, loss functions,
gradient checks, and end-to-end training experiments.

The goal is not to compete with production frameworks. The goal is to expose the mechanics
that those frameworks automate.

## Reverse-Mode Autodiff

Every operation creates a result that remembers its parent values and a small local backward
function. Calling `backward()` first orders the computation graph topologically, then walks it
in reverse. Each node combines the gradient arriving from later operations with its own local
derivative and accumulates contributions into its parents.

This is reverse-mode automatic differentiation: one scalar output can efficiently produce
gradients for many parameters, which is exactly the shape of neural network training.

## Tensor Gradients and Broadcasting

Tensor operations add a practical complication: NumPy may broadcast smaller operands across
larger dimensions. For example, adding a bias vector of shape `(features,)` to a batch of shape
`(samples, features)` repeats that bias for every sample.

During backpropagation those repeated contributions must be summed back to the bias's original
shape. The engine uses an `_unbroadcast` helper to reduce expanded axes and singleton
dimensions, preserving correct gradients for scalar operations, bias addition, and other
broadcasted expressions.

## Neural Network Components

The library includes:

- `Parameter` and recursive `Module` parameter discovery
- `Linear`, `ReLU`, `Tanh`, `Sigmoid`, `Sequential`, and configurable `MLP` modules
- `Flatten`, readable NCHW `Conv2D`, and deterministic `MaxPool2D`
- SGD with weight decay, momentum SGD, and Adam
- Mean squared error, binary cross-entropy, softmax, and stable multiclass cross-entropy
- Numerical finite-difference checks for validating analytical gradients
- `TensorDataset`, seeded mini-batch loading, reusable train/evaluation loops, and metric history
- Binary and multiclass accuracy metrics
- Deterministic parameter state dictionaries and non-pickle NumPy `.npz` save/load

All model forward passes are composed from the same public `Tensor` operations used in the
basic tests.

## Training and Persistence

`TensorDataset` and `DataLoader` provide deterministic in-memory batching without external
dataset dependencies. `train_epoch` and `evaluate` centralize the repeated mechanics of
forward passes, loss aggregation, gradient clearing, backpropagation, updates, and optional
accuracy calculation.

Models can be represented as ordered state dictionaries of copied NumPy arrays. Those states
can be restored directly or stored in `.npz` archives without pickle. Loading validates both
parameter keys and shapes, and a same-shaped model reproduces identical predictions after a
round trip.

## Experiments

### Regression

An MLP fits noisy samples from a one-dimensional linear relationship. This verifies that
matrix multiplication, broadcasting, nonlinear layers, MSE, and Adam work together in a
continuous prediction problem.

### XOR

A hidden-layer MLP learns the XOR truth table. A linear model cannot represent XOR, so this
experiment demonstrates that nonlinear activations and backpropagation can learn a genuinely
nonlinear binary decision function.

### Three-Class Spiral

An MLP learns interleaved spiral arms using stable softmax cross-entropy. This exercises
multiclass logits, integer labels, deeper networks, and nonlinear decision boundaries. The
saved visualization makes the learned class regions directly inspectable.

### MNIST MLP

The MNIST experiment downloads and parses the original gzip-compressed IDX files with standard
library tools and NumPy. A fully connected network then trains through the framework's
`TensorDataset`, `DataLoader`, reusable train/evaluation loops, cross-entropy loss, Adam,
accuracy metric, and model serialization.

The default run intentionally uses 5,000 training images, 1,000 test images, one small hidden
layer, and three epochs. This keeps a pure Python/NumPy autodiff demonstration approachable
while proving that the same engine can train on a real multiclass image dataset. It is not
configured or optimized for state-of-the-art accuracy.

### MNIST CNN

A second optional MNIST experiment keeps the image grid intact and trains a tiny convolution,
ReLU, max-pool, and linear classifier. `Conv2D` implements cross-correlation directly over
spatial windows and accumulates gradients for inputs, kernels, and biases. `MaxPool2D` records
the first maximum in each window and routes its backward gradient to that location.

The default CNN run uses only 1,000 training and 300 test images for two epochs. This is an
honest runtime tradeoff: the implementation is designed to make convolution backpropagation
readable, not to match optimized numerical libraries.

## Verification

The test suite covers forward values, analytical gradients, broadcasting, reductions, matrix
multiplication, modules, optimizers, losses, training smoke tests, and numerical gradient
checks. The deterministic examples can also be run together with:

```bash
python examples/run_all.py
```

MNIST remains separate from the quick aggregate runner because it may need a network download
and takes longer:

```bash
python examples/train_mnist_mlp.py
python examples/train_mnist_cnn.py
```

## Current Limitations

- CPU-only NumPy execution
- Convolution uses readable spatial loops rather than vectorized `im2col` or compiled kernels
- No recurrent, normalization, or dropout layers
- No train/evaluation behavior modes for layers such as dropout or batch normalization
- Serialization covers model parameters but not optimizer state or resumable checkpoints
- No graph detachment or no-gradient context
- Educational implementations are prioritized over memory and runtime optimization

## Possible Next Milestones

Useful extensions would include vectorized `im2col` convolution, additional CNN layers,
optimizer checkpointing, and richer image experiments. Profiling and graph-lifetime controls
would also make the engine more practical while preserving its readable design.
