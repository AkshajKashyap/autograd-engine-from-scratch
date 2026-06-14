# Results Summary

All results use deterministic seeds and the framework's own tensors, modules, losses,
optimizers, metrics, and training loops.

| Experiment | Task | Model | Dataset / subset | Initial loss | Final loss | Accuracy | Artifacts |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Regression | Regression | `MLP(1, 12, 1)` | 80 noisy samples from `y = 2x - 1` | 3.213180 | 0.007498 | - | [Report](regression_experiment.md), [loss](figures/regression_loss.png) |
| XOR | Binary classification | `MLP(2, 8, 1)` | 4-point XOR truth table | 0.705255 | 0.000069 | 100.00% | [Report](xor_experiment.md), [loss](figures/xor_loss.png) |
| Spiral | Multiclass classification | `MLP(2, 32, 32, 3)` | 300 synthetic spiral points | 1.116882 | 0.015983 | 99.33% | [Report](spiral_experiment.md), [loss](figures/spiral_loss.png), [boundary](figures/spiral_decision_boundary.png) |
| MNIST MLP | Image classification | `MLP(784, 64, 10)` | 5,000 train / 1,000 test | 2.355082 | 0.333938 | 89.40% | [Report](mnist_mlp_experiment.md), [loss](figures/mnist_mlp_loss.png), [accuracy](figures/mnist_mlp_accuracy.png) |
| MNIST CNN | Image classification | Conv-ReLU-Pool-Linear | 1,000 train / 300 test | 2.530988 | 0.476929 | 85.00% | [Report](mnist_cnn_experiment.md), [loss](figures/mnist_cnn_loss.png), [accuracy](figures/mnist_cnn_accuracy.png) |

The MNIST runs intentionally use small subsets and short schedules. They demonstrate that the
framework trains real image classifiers; they are not optimized benchmark submissions.
