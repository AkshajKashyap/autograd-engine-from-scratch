"""NumPy-backed tensor reverse-mode automatic differentiation."""

from collections.abc import Sequence
from numbers import Real
from typing import Callable

import numpy as np

ArrayLike = Real | Sequence[Real] | np.ndarray
Axis = int | tuple[int, ...] | None


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Reduce a broadcasted gradient back to an operand's original shape."""
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(target_shape)


class Tensor:
    """A NumPy array that records the computation used to produce it."""

    __array_priority__ = 1000

    def __init__(
        self,
        data: ArrayLike,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._prev = _children
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, grad={self.grad!r})"

    @staticmethod
    def _coerce(other: "Tensor | ArrayLike") -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other: "Tensor | ArrayLike") -> "Tensor":
        other = self._coerce(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __mul__(self, other: "Tensor | ArrayLike") -> "Tensor":
        other = self._coerce(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self_grad = out.grad * other.data
            other_grad = out.grad * self.data
            self.grad += _unbroadcast(self_grad, self.data.shape)
            other.grad += _unbroadcast(other_grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __neg__(self) -> "Tensor":
        return self * -1

    def __sub__(self, other: "Tensor | ArrayLike") -> "Tensor":
        return self + -self._coerce(other)

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) - self

    def __truediv__(self, other: "Tensor | ArrayLike") -> "Tensor":
        return self * self._coerce(other) ** -1

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) / self

    def __pow__(self, exponent: Real) -> "Tensor":
        if not isinstance(exponent, Real):
            raise TypeError("Tensor only supports powers with a numeric exponent")

        out = Tensor(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor | ArrayLike") -> "Tensor":
        other = self._coerce(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            left = self.data
            right = other.data
            upstream = out.grad

            # Promote vectors to matrices, apply the matrix derivatives, then
            # remove the temporary dimensions and any broadcasted batch axes.
            left_vector = left.ndim == 1
            right_vector = right.ndim == 1
            left_matrix = left[np.newaxis, :] if left_vector else left
            right_matrix = right[:, np.newaxis] if right_vector else right

            if left_vector and right_vector:
                upstream_matrix = upstream.reshape(1, 1)
            elif left_vector:
                upstream_matrix = np.expand_dims(upstream, axis=-2)
            elif right_vector:
                upstream_matrix = np.expand_dims(upstream, axis=-1)
            else:
                upstream_matrix = upstream

            left_grad = upstream_matrix @ np.swapaxes(right_matrix, -1, -2)
            right_grad = np.swapaxes(left_matrix, -1, -2) @ upstream_matrix

            if left_vector:
                left_grad = np.squeeze(left_grad, axis=-2)
            if right_vector:
                right_grad = np.squeeze(right_grad, axis=-1)

            self.grad += _unbroadcast(left_grad, left.shape)
            other.grad += _unbroadcast(right_grad, right.shape)

        out._backward = _backward
        return out

    def __rmatmul__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) @ self

    def reshape(self, *shape: int | tuple[int, ...]) -> "Tensor":
        """Return a view with a new shape while preserving gradient flow."""
        target_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        out = Tensor(self.data.reshape(target_shape), (self,), "reshape")

        def _backward() -> None:
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: int | tuple[int, ...]) -> "Tensor":
        """Permute dimensions and invert that permutation during backward."""
        if len(axes) == 1 and isinstance(axes[0], tuple):
            permutation = axes[0]
        else:
            permutation = axes or tuple(reversed(range(self.data.ndim)))
        if len(permutation) != self.data.ndim:
            raise ValueError("transpose axes must include every tensor dimension")

        out = Tensor(self.data.transpose(permutation), (self,), "transpose")
        inverse_permutation = tuple(np.argsort(permutation))

        def _backward() -> None:
            self.grad += out.grad.transpose(inverse_permutation)

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        """Return the transpose of a two-dimensional tensor."""
        if self.data.ndim != 2:
            raise ValueError("Tensor.T is only defined for two-dimensional tensors")
        return self.transpose(1, 0)

    def sum(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else axis
                normalized_axes = tuple(
                    current_axis if current_axis >= 0 else current_axis + self.data.ndim
                    for current_axis in axes
                )
                grad = np.expand_dims(grad, axis=normalized_axes)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis: Axis = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            divisor = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            divisor = int(np.prod([self.data.shape[current_axis] for current_axis in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / divisor

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        result = np.tanh(self.data)
        out = Tensor(result, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - result**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        result = np.exp(self.data)
        out = Tensor(result, (self,), "exp")

        def _backward() -> None:
            self.grad += result * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def backward(self, grad: ArrayLike | None = None) -> None:
        """Backpropagate from this tensor through its computation graph."""
        if grad is None:
            initial_grad = np.ones_like(self.data)
        else:
            initial_grad = np.asarray(grad, dtype=float)
            if initial_grad.shape != self.data.shape:
                raise ValueError("Initial gradient must have the same shape as the tensor")

        topological_order: list[Tensor] = []
        visited: set[Tensor] = set()

        def build_topological_order(node: Tensor) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                build_topological_order(parent)
            topological_order.append(node)

        build_topological_order(self)

        self.grad = initial_grad.copy()
        for node in reversed(topological_order):
            node._backward()
