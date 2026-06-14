"""Numerical gradient checks for Tensor expressions."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from autograd_engine.tensor import Tensor


@dataclass(frozen=True)
class InputGradcheckResult:
    """Error summary for one input tensor."""

    passed: bool
    max_absolute_error: float
    max_relative_error: float


@dataclass(frozen=True)
class GradcheckResult:
    """Combined analytical-versus-numerical gradient comparison."""

    passed: bool
    max_absolute_error: float
    max_relative_error: float
    inputs: tuple[InputGradcheckResult, ...]


def _validate_inputs(inputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
    tensors = tuple(inputs)
    if not tensors:
        raise ValueError("gradcheck requires at least one input tensor")
    if not all(isinstance(tensor, Tensor) for tensor in tensors):
        raise TypeError("all gradcheck inputs must be Tensor objects")
    return tensors


def _evaluate_scalar(fn: Callable[..., Tensor], inputs: tuple[Tensor, ...]) -> Tensor:
    output = fn(*inputs)
    if not isinstance(output, Tensor):
        raise TypeError("gradcheck function must return a Tensor")
    if output.data.shape != ():
        raise ValueError("gradcheck function must return a scalar Tensor")
    return output


def finite_difference_grad(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
) -> tuple[np.ndarray, ...]:
    """Estimate gradients with central finite differences."""
    if eps <= 0:
        raise ValueError("eps must be positive")

    tensors = _validate_inputs(inputs)
    original_data = [tensor.data.copy() for tensor in tensors]
    numerical_gradients = [np.zeros_like(tensor.data) for tensor in tensors]

    try:
        for tensor, gradient in zip(tensors, numerical_gradients, strict=True):
            for index in np.ndindex(tensor.data.shape):
                original_value = tensor.data[index]

                tensor.data[index] = original_value + eps
                forward_value = _evaluate_scalar(fn, tensors).data.item()

                tensor.data[index] = original_value - eps
                backward_value = _evaluate_scalar(fn, tensors).data.item()

                gradient[index] = (forward_value - backward_value) / (2 * eps)
                tensor.data[index] = original_value
    finally:
        for tensor, data in zip(tensors, original_data, strict=True):
            tensor.data[...] = data

    return tuple(numerical_gradients)


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> GradcheckResult:
    """Compare autograd gradients with central finite differences."""
    tensors = _validate_inputs(inputs)
    for tensor in tensors:
        tensor.grad.fill(0.0)

    output = _evaluate_scalar(fn, tensors)
    output.backward()
    analytical_gradients = tuple(tensor.grad.copy() for tensor in tensors)
    numerical_gradients = finite_difference_grad(fn, tensors, eps=eps)

    input_results: list[InputGradcheckResult] = []
    for analytical, numerical in zip(
        analytical_gradients,
        numerical_gradients,
        strict=True,
    ):
        absolute_error = np.abs(analytical - numerical)
        scale = np.maximum(np.abs(analytical), np.abs(numerical))
        relative_error = absolute_error / np.maximum(scale, np.finfo(float).eps)
        input_results.append(
            InputGradcheckResult(
                passed=bool(np.allclose(analytical, numerical, atol=atol, rtol=rtol)),
                max_absolute_error=float(np.max(absolute_error, initial=0.0)),
                max_relative_error=float(np.max(relative_error, initial=0.0)),
            )
        )

    results = tuple(input_results)
    return GradcheckResult(
        passed=all(result.passed for result in results),
        max_absolute_error=max(result.max_absolute_error for result in results),
        max_relative_error=max(result.max_relative_error for result in results),
        inputs=results,
    )
