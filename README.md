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
- [x] Seeded mini-batch data loading and reusable train/evaluation loops
- [x] Binary and multiclass accuracy metrics
- [x] Deterministic model state dictionaries and NumPy `.npz` persistence
- [x] Deterministic regression, XOR, and spiral experiments
- [x] MNIST IDX loading and a subset-based MLP image classifier
- [x] Reshape/transpose autograd plus readable `Conv2D`, `MaxPool2D`, and `Flatten`

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

Run the heavier MNIST experiment separately:

```bash
python examples/train_mnist_mlp.py
python examples/train_mnist_cnn.py
```

The MLP uses 5,000/1,000 images by default. The CNN uses a smaller 1,000/300 subset and a
single convolution/pooling stage because its NumPy loops prioritize readable gradient logic
over production speed. Both are framework demonstrations rather than state-of-the-art
benchmarks.

See [reports/project_summary.md](reports/project_summary.md) for the design overview and
experiment discussion.

## Limitations

This is an educational CPU-only engine without GPU support, vectorized production convolution,
data augmentation, automatic graph cleanup, or a no-gradient mode. It favors readable
implementation over production performance.

## Future Work

Possible next steps include vectorized `im2col` convolution, more CNN layers, optimizer
checkpointing, train/evaluation modes, and broader image experiments.
