"""Loss functions built from differentiable Tensor operations."""

import numpy as np

from autograd_engine.tensor import ArrayLike, Tensor


def _as_tensor(value: Tensor | ArrayLike) -> Tensor:
    return value if isinstance(value, Tensor) else Tensor(value)


def mse_loss(prediction: Tensor, target: Tensor | ArrayLike) -> Tensor:
    """Return the mean squared error between predictions and targets."""
    target = _as_tensor(target)
    return ((prediction - target) ** 2).mean()


def binary_cross_entropy(
    prediction: Tensor,
    target: Tensor | ArrayLike,
    eps: float = 1e-7,
) -> Tensor:
    """Return mean binary cross-entropy for probability predictions."""
    if eps <= 0:
        raise ValueError("eps must be positive")

    target = _as_tensor(target)
    positive_term = target * (prediction + eps).log()
    negative_term = (1 - target) * (1 - prediction + eps).log()
    return -(positive_term + negative_term).mean()


def softmax(logits: Tensor, axis: int = -1) -> Tensor:
    """Convert logits into probabilities along an axis."""
    # The maximum is a stop-gradient constant used only to keep exp values bounded.
    shifted_logits = logits - np.max(logits.data, axis=axis, keepdims=True)
    exponentials = shifted_logits.exp()
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def cross_entropy(
    logits: Tensor,
    targets: Tensor | ArrayLike,
    eps: float = 1e-12,
) -> Tensor:
    """Return mean multiclass cross-entropy from logits and labels."""
    if logits.data.ndim != 2:
        raise ValueError("logits must have shape (n_samples, n_classes)")
    if eps <= 0:
        raise ValueError("eps must be positive")

    target_data = targets.data if isinstance(targets, Tensor) else np.asarray(targets)
    n_samples, n_classes = logits.data.shape

    if target_data.shape == (n_samples,):
        if not np.all(np.equal(target_data, target_data.astype(int))):
            raise ValueError("class labels must be integers")
        class_labels = target_data.astype(int)
        if np.any((class_labels < 0) | (class_labels >= n_classes)):
            raise ValueError("class labels must be between 0 and n_classes - 1")
        one_hot = np.eye(n_classes)[class_labels]
    elif target_data.shape == logits.data.shape:
        one_hot = np.asarray(target_data, dtype=float)
    else:
        raise ValueError("targets must be class labels or one-hot labels matching logits")

    # Compute log-softmax directly so very small class probabilities do not
    # underflow before the logarithm.
    shifted_logits = logits - np.max(logits.data, axis=-1, keepdims=True)
    log_normalizer = (shifted_logits.exp().sum(axis=-1, keepdims=True) + eps).log()
    log_probabilities = shifted_logits - log_normalizer
    return -(Tensor(one_hot) * log_probabilities).sum(axis=-1).mean()
