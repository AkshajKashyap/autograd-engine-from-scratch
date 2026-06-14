"""Small neural network building blocks backed by the Tensor autograd engine."""

from collections.abc import Sequence

import numpy as np

from autograd_engine.tensor import ArrayLike, Tensor


class Parameter(Tensor):
    """A tensor intended to be updated during optimization."""

    def __init__(self, data: ArrayLike | Tensor) -> None:
        super().__init__(data.data if isinstance(data, Tensor) else data)


class Module:
    """Base class for objects that contain trainable parameters."""

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def parameters(self) -> list[Parameter]:
        parameters: list[Parameter] = []
        seen: set[int] = set()

        def collect(value: object) -> None:
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    parameters.append(value)
                    seen.add(id(value))
            elif isinstance(value, Module):
                for child in vars(value).values():
                    collect(child)
            elif isinstance(value, dict):
                for child in value.values():
                    collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)

        for attribute in vars(self).values():
            collect(attribute)
        return parameters

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad.fill(0.0)


class Linear(Module):
    """An affine transformation: ``inputs @ weight + bias``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")

        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Parameter(np.random.randn(in_features, out_features) * scale)
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs @ self.weight
        return output + self.bias if self.bias is not None else output


class ReLU(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.relu()


class Tanh(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.tanh()


class Sigmoid(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return 1 / (1 + (-inputs).exp())


class Sequential(Module):
    """Apply a sequence of modules in order."""

    def __init__(self, *modules: Module) -> None:
        self.modules = list(modules)

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for module in self.modules:
            output = module(output)
        return output


class MLP(Sequential):
    """A multilayer perceptron built from a sequence of feature sizes."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: type[Module] = ReLU,
        output_activation: type[Module] | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("MLP requires at least an input and output size")

        modules: list[Module] = []
        last_layer_index = len(layer_sizes) - 2
        for index, (in_features, out_features) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], strict=True)
        ):
            modules.append(Linear(in_features, out_features))
            if index < last_layer_index:
                modules.append(activation())
            elif output_activation is not None:
                modules.append(output_activation())

        super().__init__(*modules)
