"""Optimizers for trainable Tensor parameters."""

from collections.abc import Iterable

import numpy as np

from autograd_engine.nn import Parameter


class Optimizer:
    """Base class for parameter update rules."""

    def __init__(self, parameters: Iterable[Parameter]) -> None:
        self.parameters = list(parameters)

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad.fill(0.0)


class SGD(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        *,
        lr: float | None = None,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = learning_rate if lr is None else lr
        self.weight_decay = weight_decay

    def step(self) -> None:
        for parameter in self.parameters:
            gradient = parameter.grad + self.weight_decay * parameter.data
            parameter.data -= self.learning_rate * gradient


class MomentumSGD(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        *,
        lr: float | None = None,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = learning_rate if lr is None else lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self) -> None:
        for parameter, velocity in zip(self.parameters, self.velocities, strict=True):
            gradient = parameter.grad + self.weight_decay * parameter.data
            velocity *= self.momentum
            velocity += gradient
            parameter.data -= self.learning_rate * velocity


class Adam(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        lr: float | None = None,
    ) -> None:
        super().__init__(parameters)
        self.learning_rate = learning_rate if lr is None else lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.step_count = 0
        self.first_moments = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self.second_moments = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def step(self) -> None:
        self.step_count += 1

        state = zip(
            self.parameters,
            self.first_moments,
            self.second_moments,
            strict=True,
        )
        for parameter, first_moment, second_moment in state:
            gradient = parameter.grad + self.weight_decay * parameter.data

            first_moment *= self.beta1
            first_moment += (1 - self.beta1) * gradient
            second_moment *= self.beta2
            second_moment += (1 - self.beta2) * gradient**2

            corrected_first = first_moment / (1 - self.beta1**self.step_count)
            corrected_second = second_moment / (1 - self.beta2**self.step_count)
            parameter.data -= self.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + self.epsilon
            )
