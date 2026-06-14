"""Small neural network building blocks backed by the Tensor autograd engine."""

from collections.abc import Sequence

import numpy as np

from autograd_engine.tensor import ArrayLike, Tensor


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("expected an integer or a pair of integers")
    return value


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


class Flatten(Module):
    """Flatten every dimension after the batch dimension."""

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.data.ndim < 2:
            raise ValueError("Flatten expects a batch dimension")
        return inputs.reshape(inputs.data.shape[0], -1)


class Conv2D(Module):
    """A readable NCHW two-dimensional cross-correlation layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        bias: bool = True,
    ) -> None:
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")

        kernel_height, kernel_width = _pair(kernel_size)
        stride_height, stride_width = _pair(stride)
        padding_height, padding_width = _pair(padding)
        if kernel_height <= 0 or kernel_width <= 0:
            raise ValueError("kernel_size must be positive")
        if stride_height <= 0 or stride_width <= 0:
            raise ValueError("stride must be positive")
        if padding_height < 0 or padding_width < 0:
            raise ValueError("padding must be non-negative")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_height, kernel_width)
        self.stride = (stride_height, stride_width)
        self.padding = (padding_height, padding_width)

        fan_in = in_channels * kernel_height * kernel_width
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, kernel_height, kernel_width)
            * np.sqrt(2.0 / fan_in)
        )
        self.bias = Parameter(np.zeros(out_channels)) if bias else None

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.data.ndim != 4:
            raise ValueError("Conv2D expects input shape (batch, channels, height, width)")
        if inputs.data.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {inputs.data.shape[1]}"
            )

        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        padding_height, padding_width = self.padding
        batch_size, _, input_height, input_width = inputs.data.shape
        padded_height = input_height + 2 * padding_height
        padded_width = input_width + 2 * padding_width
        if kernel_height > padded_height or kernel_width > padded_width:
            raise ValueError("kernel is larger than the padded input")

        output_height = (padded_height - kernel_height) // stride_height + 1
        output_width = (padded_width - kernel_width) // stride_width + 1
        padded_inputs = np.pad(
            inputs.data,
            (
                (0, 0),
                (0, 0),
                (padding_height, padding_height),
                (padding_width, padding_width),
            ),
        )
        output_data = np.empty(
            (batch_size, self.out_channels, output_height, output_width),
            dtype=float,
        )

        for output_row in range(output_height):
            row_start = output_row * stride_height
            for output_column in range(output_width):
                column_start = output_column * stride_width
                window = padded_inputs[
                    :,
                    :,
                    row_start : row_start + kernel_height,
                    column_start : column_start + kernel_width,
                ]
                output_data[:, :, output_row, output_column] = np.einsum(
                    "bchw,ochw->bo",
                    window,
                    self.weight.data,
                )

        if self.bias is not None:
            output_data += self.bias.data.reshape(1, -1, 1, 1)

        parents = (inputs, self.weight) if self.bias is None else (inputs, self.weight, self.bias)
        out = Tensor(output_data, parents, "conv2d")

        def _backward() -> None:
            padded_input_grad = np.zeros_like(padded_inputs)

            for output_row in range(output_height):
                row_start = output_row * stride_height
                for output_column in range(output_width):
                    column_start = output_column * stride_width
                    upstream = out.grad[:, :, output_row, output_column]
                    window = padded_inputs[
                        :,
                        :,
                        row_start : row_start + kernel_height,
                        column_start : column_start + kernel_width,
                    ]
                    padded_input_grad[
                        :,
                        :,
                        row_start : row_start + kernel_height,
                        column_start : column_start + kernel_width,
                    ] += np.einsum("bo,ochw->bchw", upstream, self.weight.data)
                    self.weight.grad += np.einsum("bo,bchw->ochw", upstream, window)

            if padding_height == 0 and padding_width == 0:
                inputs.grad += padded_input_grad
            else:
                inputs.grad += padded_input_grad[
                    :,
                    :,
                    padding_height : padding_height + input_height,
                    padding_width : padding_width + input_width,
                ]
            if self.bias is not None:
                self.bias.grad += out.grad.sum(axis=(0, 2, 3))

        out._backward = _backward
        return out


class MaxPool2D(Module):
    """Two-dimensional max pooling for NCHW tensors."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 2,
        stride: int | tuple[int, int] | None = None,
    ) -> None:
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(kernel_size if stride is None else stride)
        if any(size <= 0 for size in (*self.kernel_size, *self.stride)):
            raise ValueError("kernel_size and stride must be positive")

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.data.ndim != 4:
            raise ValueError("MaxPool2D expects input shape (batch, channels, height, width)")

        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        batch_size, channels, input_height, input_width = inputs.data.shape
        if kernel_height > input_height or kernel_width > input_width:
            raise ValueError("pooling kernel is larger than the input")

        output_height = (input_height - kernel_height) // stride_height + 1
        output_width = (input_width - kernel_width) // stride_width + 1
        output_data = np.empty((batch_size, channels, output_height, output_width))
        max_indices = np.empty(
            (batch_size, channels, output_height, output_width),
            dtype=int,
        )

        for output_row in range(output_height):
            row_start = output_row * stride_height
            for output_column in range(output_width):
                column_start = output_column * stride_width
                window = inputs.data[
                    :,
                    :,
                    row_start : row_start + kernel_height,
                    column_start : column_start + kernel_width,
                ]
                flattened_window = window.reshape(batch_size, channels, -1)
                # np.argmax chooses the first maximum, making tie handling deterministic.
                indices = np.argmax(flattened_window, axis=-1)
                max_indices[:, :, output_row, output_column] = indices
                output_data[:, :, output_row, output_column] = np.take_along_axis(
                    flattened_window,
                    indices[..., None],
                    axis=-1,
                ).squeeze(-1)

        out = Tensor(output_data, (inputs,), "maxpool2d")

        def _backward() -> None:
            input_grad = np.zeros_like(inputs.data)
            batch_indices = np.arange(batch_size)[:, None]
            channel_indices = np.arange(channels)[None, :]

            for output_row in range(output_height):
                row_start = output_row * stride_height
                for output_column in range(output_width):
                    column_start = output_column * stride_width
                    indices = max_indices[:, :, output_row, output_column]
                    row_offsets = indices // kernel_width
                    column_offsets = indices % kernel_width
                    input_grad[
                        batch_indices,
                        channel_indices,
                        row_start + row_offsets,
                        column_start + column_offsets,
                    ] += out.grad[:, :, output_row, output_column]

            inputs.grad += input_grad

        out._backward = _backward
        return out


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
