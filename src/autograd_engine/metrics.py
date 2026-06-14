"""Common prediction metrics."""

import numpy as np

from autograd_engine.tensor import ArrayLike, Tensor


def _as_array(value: Tensor | ArrayLike) -> np.ndarray:
    return value.data if isinstance(value, Tensor) else np.asarray(value)


def binary_accuracy(
    prediction: Tensor,
    target: Tensor | ArrayLike,
    threshold: float = 0.5,
) -> float:
    """Return the fraction of correct binary probability predictions."""
    target_data = _as_array(target)
    if prediction.data.shape != target_data.shape:
        raise ValueError("prediction and target shapes must match")

    predicted_labels = prediction.data >= threshold
    target_labels = target_data >= threshold
    return float(np.mean(predicted_labels == target_labels))


def multiclass_accuracy(logits: Tensor, target: Tensor | ArrayLike) -> float:
    """Return accuracy for integer or one-hot multiclass targets."""
    if logits.data.ndim != 2:
        raise ValueError("logits must have shape (n_samples, n_classes)")

    target_data = _as_array(target)
    if target_data.shape == (logits.data.shape[0],):
        target_labels = target_data.astype(int)
    elif target_data.shape == logits.data.shape:
        target_labels = np.argmax(target_data, axis=1)
    else:
        raise ValueError("target must contain integer labels or one-hot rows")

    predictions = np.argmax(logits.data, axis=1)
    return float(np.mean(predictions == target_labels))
