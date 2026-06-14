# Architecture

## Computation Graphs

`Tensor` wraps a NumPy array in `data` and stores a same-shaped gradient array in `grad`.
Every differentiable operation creates a new `Tensor` containing:

- the forward result
- references to its parent tensors
- a short operation label
- a local `_backward` closure

The closure knows only the derivative of that operation. For example, multiplication sends
the upstream gradient to each parent after multiplying by the other operand. Broadcasted
operands pass through `_unbroadcast`, which sums repeated gradient dimensions back to the
parent's original shape.

## Backward Pass

`Tensor.backward()` performs a depth-first topological sort from the output to all reachable
parents. It seeds the output gradient with ones, then visits nodes in reverse topological
order and calls each local backward closure.

Gradients use `+=` rather than assignment. This matters when one tensor feeds several later
operations: every path contributes to the same derivative.

Shape operations invert their forward transformation. `reshape` restores the original shape,
and `transpose` applies the inverse axis permutation. Convolution and pooling create the same
kind of graph node, but use explicit NumPy loops and cached window information in backward.

## Modules and Parameters

`Parameter` is a trainable `Tensor`. `Module.parameters()` recursively walks module attributes,
lists, tuples, and dictionaries, collecting each parameter once in deterministic order.

Layers such as `Linear`, `Conv2D`, and `MLP` are ordinary Python objects whose forward methods
compose tensor operations. `Sequential` passes one module's output into the next. There is no
separate graph compiler or hidden execution path.

## Optimizers

Optimizers receive the list returned by `model.parameters()`. After backpropagation:

1. `SGD` subtracts the scaled gradient.
2. `MomentumSGD` maintains a velocity array for each parameter.
3. `Adam` maintains first and second moments with bias correction.

`optimizer.zero_grad()` clears accumulated parameter gradients before the next backward pass.
Model state serialization uses the same deterministic parameter order and stores copied arrays
in a non-pickle NumPy `.npz` archive.

## Training Flow

`TensorDataset` stores input and target arrays. `DataLoader` creates deterministic optional
shuffles and yields mini-batches as `Tensor` objects.

A typical epoch is:

```text
DataLoader batch
  -> model(inputs)
  -> loss(predictions, targets)
  -> optimizer.zero_grad()
  -> loss.backward()
  -> optimizer.step()
```

`train_epoch` implements that loop and returns sample-weighted average loss. `evaluate` runs
forward passes and aggregates loss plus an optional accuracy metric without updating model
parameters. `History` stores per-epoch values for reports and plots.

## Examples

- Regression verifies continuous prediction and MSE training.
- XOR verifies nonlinear binary classification.
- Spiral verifies multiclass softmax cross-entropy and decision boundaries.
- MNIST MLP verifies real IDX data loading, batching, metrics, and serialization.
- MNIST CNN verifies convolution, pooling, flattening, and image classification.

The quick synthetic examples run through `examples/run_all.py`. MNIST experiments remain
separate because they download data and perform heavier training.
