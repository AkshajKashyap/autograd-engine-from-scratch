"""Loss functions built from differentiable Tensor operations."""

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
