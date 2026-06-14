# Autograd Engine From Scratch

A compact NumPy-based reverse-mode autodiff engine and neural network library built from
first principles. The project covers scalar and tensor computation graphs, broadcasting-aware
gradients, MLP and CNN layers, optimizers, losses, data loading, serialization, gradient
checking, and deterministic end-to-end experiments.

## Features

- [x] Scalar and NumPy tensor reverse-mode autodiff
- [x] Broadcasting, reductions, matrix multiplication, reshape, and transpose gradients
- [x] `Linear`, activations, `MLP`, `Conv2D`, `MaxPool2D`, and `Flatten`
- [x] SGD, momentum SGD, Adam, regression and classification losses
- [x] DataLoader, train/evaluation loops, metrics, and `.npz` model serialization
- [x] Central finite-difference gradient checks
- [x] Regression, XOR, spiral, MNIST MLP, and MNIST CNN experiments

## Quickstart

```bash
make install
make check
make examples
```

Equivalent direct commands:

```bash
pytest
ruff check .
python examples/run_all.py
```

## Experiments

```bash
python examples/train_regression.py
python examples/train_xor.py
python examples/train_spiral.py
python examples/train_mnist_mlp.py
python examples/train_mnist_cnn.py
```

MNIST downloads the original IDX files when absent. The MLP uses a 5,000/1,000 subset; the
readable loop-based CNN uses 1,000/300 images to keep runtime practical.

## Results

| Experiment | Final loss | Accuracy |
| --- | ---: | ---: |
| Regression | 0.007498 | - |
| XOR | 0.000069 | 100.00% |
| Spiral | 0.015983 | 99.33% |
| MNIST MLP | 0.333938 | 89.40% |
| MNIST CNN | 0.476929 | 85.00% |

Full context and artifacts are listed in
[reports/results_summary.md](reports/results_summary.md).

## Documentation

- [Project summary](reports/project_summary.md)
- [Architecture](docs/architecture.md)
- [Concepts review](docs/concepts_review.md)

## Limitations

This is an educational CPU-only framework. It has no GPU backend, compiled kernels, automatic
graph cleanup, no-gradient context, data augmentation, or production-speed convolution.
Optimizer state is not included in saved checkpoints.

## Future Work

Natural extensions include vectorized `im2col` convolution, more tensor operations, resumable
optimizer checkpoints, train/evaluation modes, and deeper image models.
