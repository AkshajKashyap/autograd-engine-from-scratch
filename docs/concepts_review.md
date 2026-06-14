# Concepts Review

## Reverse-Mode Autodiff

Reverse-mode autodiff computes derivatives of one scalar output with respect to many inputs.
The forward pass records operations. The backward pass applies the chain rule in reverse.
This is well suited to neural networks because a scalar loss may depend on thousands or
millions of parameters.

Key distinction: autodiff evaluates exact derivatives of the recorded program. It is not
symbolic algebra and it is not numerical finite differences.

## Computational Graphs

A computational graph represents values as nodes and operations as edges. Each result stores
its parents and a local derivative rule. Backpropagation requires parents to be processed only
after all downstream gradient contributions are known, so the graph is traversed in reverse
topological order.

## Gradient Accumulation

If `y = x * x + x`, then `x` reaches the output through three paths. Its gradient is the sum
of all path contributions. Backward implementations therefore accumulate with `+=`.

This also explains why gradients must be cleared between optimization steps. Otherwise,
gradients from earlier batches remain in parameter buffers.

## Broadcasting Gradients

Broadcasting repeats an operand conceptually without copying it. If a bias vector is added to
every row in a batch, its backward gradient must sum across the batch dimension.

The general rule is to remove extra leading dimensions and sum axes where the original operand
had size one. The result must match the operand's original shape exactly.

## Matrix Multiplication Gradients

For `Y = A @ B` and upstream gradient `G`:

```text
dL/dA = G @ B.T
dL/dB = A.T @ G
```

Vector and batched cases require temporary dimension promotion and reduction of broadcasted
batch axes, but the same matrix identities apply.

## Softmax Cross-Entropy

Softmax converts logits into normalized class probabilities. Direct exponentiation can
overflow, so subtract the row maximum first. The shift does not change the probabilities.

Cross-entropy selects the log probability assigned to the correct class and averages its
negative value. A stable implementation computes log-softmax using a shifted log-sum-exp
expression instead of taking `log(softmax)` after probabilities may have underflowed.

## Optimizers

**SGD** follows the negative gradient. It is simple and predictable but may oscillate or move
slowly through poorly scaled landscapes.

**Momentum** accumulates a velocity, smoothing noisy updates and carrying progress through
consistent gradient directions.

**Adam** tracks exponential averages of gradients and squared gradients. Bias correction
matters early because both moment estimates start at zero. Adam is often convenient, but its
extra state and adaptive scaling do not guarantee better generalization.

## CNN Basics

Convolutional layers share a small kernel across spatial locations. In most deep learning
libraries the operation called convolution is technically cross-correlation because the
kernel is not flipped.

Backward propagation must:

- accumulate input gradients from every overlapping window
- accumulate kernel gradients across batches and spatial positions
- sum bias gradients over batches and output locations

Max pooling keeps the maximum from each window. Its gradient routes only to the selected
maximum. This project resolves ties deterministically by choosing the first maximum.

Flattening reshapes feature maps into vectors before a linear classifier. Its backward pass
simply restores the original shape.

## Finite-Difference Gradient Checks

Central finite differences approximate one derivative as:

```text
(f(x + eps) - f(x - eps)) / (2 * eps)
```

They are slow because each input element needs extra forward evaluations, but they are valuable
for testing local backward rules. Use them on tiny deterministic inputs and compare with both
absolute and relative tolerances.

Finite differences are a verification tool, not a training algorithm.

## Educational Framework Limitations

A readable NumPy engine makes graph construction and gradients inspectable, but it lacks many
production features:

- compiled kernels, GPU execution, and memory planning
- vectorized high-performance convolution
- automatic graph disposal and no-gradient contexts
- distributed training and mixed precision
- broad operation coverage and extensive numerical hardening

Those limitations are part of the lesson: mature frameworks provide substantial systems
engineering around the core chain rule.
