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
- SGD with weight decay, momentum SGD, and Adam
- Mean squared error, binary cross-entropy, softmax, and stable multiclass cross-entropy
- Numerical finite-difference checks for validating analytical gradients

All model forward passes are composed from the same public `Tensor` operations used in the
basic tests.

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

## Verification

The test suite covers forward values, analytical gradients, broadcasting, reductions, matrix
multiplication, modules, optimizers, losses, training smoke tests, and numerical gradient
checks. The deterministic examples can also be run together with:

```bash
python examples/run_all.py
```

## Current Limitations

- CPU-only NumPy execution
- No convolutional, recurrent, normalization, or dropout layers
- No data loader, mini-batch abstraction, or train/evaluation modes
- No model serialization or checkpointing
- No graph detachment or no-gradient context
- Educational implementations are prioritized over memory and runtime optimization

## Possible Next Milestones

Useful extensions would include additional tensor operations, parameter serialization,
mini-batch data utilities, convolutional layers, and a small image-classification experiment
such as MNIST. Profiling and graph-lifetime controls would also make the engine more practical
while preserving its readable design.
