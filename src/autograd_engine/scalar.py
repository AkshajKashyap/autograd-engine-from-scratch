"""Scalar reverse-mode automatic differentiation."""

import math
from numbers import Real
from typing import Callable


class Value:
    """A scalar value that records the computation used to produce it."""

    def __init__(
        self,
        data: Real,
        _children: tuple["Value", ...] = (),
        _op: str = "",
    ) -> None:
        if not isinstance(data, Real):
            raise TypeError("Value data must be a real number")

        self.data = float(data)
        self.grad = 0.0
        self._prev = _children
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    @staticmethod
    def _coerce(other: "Value | Real") -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: "Value | Real") -> "Value":
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Real) -> "Value":
        return self + other

    def __mul__(self, other: "Value | Real") -> "Value":
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Real) -> "Value":
        return self * other

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: "Value | Real") -> "Value":
        return self + -self._coerce(other)

    def __rsub__(self, other: Real) -> "Value":
        return self._coerce(other) - self

    def __truediv__(self, other: "Value | Real") -> "Value":
        return self * self._coerce(other) ** -1

    def __rtruediv__(self, other: Real) -> "Value":
        return self._coerce(other) / self

    def __pow__(self, exponent: Real) -> "Value":
        if not isinstance(exponent, Real):
            raise TypeError("Value only supports powers with a numeric exponent")

        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        result = math.tanh(self.data)
        out = Value(result, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - result**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        result = math.exp(self.data)
        out = Value(result, (self,), "exp")

        def _backward() -> None:
            self.grad += result * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        result = math.log(self.data)
        out = Value(result, (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Backpropagate gradients from this value through its computation graph."""
        topological_order: list[Value] = []
        visited: set[Value] = set()

        def build_topological_order(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                build_topological_order(parent)
            topological_order.append(node)

        build_topological_order(self)

        self.grad = 1.0
        for node in reversed(topological_order):
            node._backward()
