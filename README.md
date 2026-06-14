# Autograd Engine From Scratch

A NumPy-based reverse-mode autodiff engine and tiny neural network library built to make
backpropagation inspectable from scalar operations through complete training loops.

## Features

- [x] Scalar and NumPy-backed tensor computation graphs
- [x] Broadcasting-aware gradients, reductions, matrix multiplication, and nonlinearities
- [x] Modules, parameters, linear layers, activations, sequential models, and MLPs
- [x] SGD, momentum SGD, and Adam
- [x] Regression, binary classification, and multiclass losses
- [x] Central finite-difference gradient checks
- [x] Deterministic regression, XOR, and spiral experiments

## Verify

```bash
pytest
ruff check .
```

## Examples

```bash
python examples/train_regression.py
python examples/train_xor.py
python examples/train_spiral.py
python examples/run_all.py
```

Regression demonstrates continuous function fitting, XOR demonstrates nonlinear binary
classification, and the spiral experiment demonstrates multiclass decision boundaries. The
scripts save metrics, reports, and figures under `reports/`.

See [reports/project_summary.md](reports/project_summary.md) for the design overview and
experiment discussion.

## Limitations

This is an educational CPU-only engine without GPU support, batching utilities, model
serialization, convolutional layers, or a no-gradient mode. It favors readable implementation
over production performance.

## Future Work

Possible next steps include richer tensor operations, data loaders, model persistence,
convolutional layers, and experiments on MNIST or similarly sized datasets.
