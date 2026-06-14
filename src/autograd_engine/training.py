"""Reusable datasets, batching, and training loops."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import math

import numpy as np

from autograd_engine.nn import Module
from autograd_engine.optim import Optimizer
from autograd_engine.tensor import ArrayLike, Tensor

LossFunction = Callable[[Tensor, Tensor], Tensor]
AccuracyFunction = Callable[[Tensor, Tensor], float]


class Dataset(ABC):
    """Base interface for indexable input-target pairs."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class TensorDataset(Dataset):
    """An in-memory dataset backed by tensors or NumPy-compatible arrays."""

    def __init__(self, inputs: Tensor | ArrayLike, targets: Tensor | ArrayLike) -> None:
        self.inputs = np.asarray(inputs.data if isinstance(inputs, Tensor) else inputs)
        self.targets = np.asarray(targets.data if isinstance(targets, Tensor) else targets)

        if self.inputs.ndim == 0 or self.targets.ndim == 0:
            raise ValueError("inputs and targets must include a sample dimension")
        if len(self.inputs) != len(self.targets):
            raise ValueError("inputs and targets must contain the same number of samples")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.inputs[index], self.targets[index]


class DataLoader:
    """Iterate over a dataset in optionally shuffled mini-batches."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            samples = [self.dataset[int(index)] for index in batch_indices]
            inputs, targets = zip(*samples, strict=True)
            yield Tensor(np.stack(inputs)), Tensor(np.stack(targets))


@dataclass
class History:
    """Per-epoch training and evaluation metrics."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)

    def append(
        self,
        *,
        train_loss: float,
        val_loss: float | None = None,
        accuracy: float | None = None,
    ) -> None:
        self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if accuracy is not None:
            self.accuracy.append(accuracy)


def train_epoch(
    model: Module,
    dataloader: DataLoader,
    loss_fn: LossFunction,
    optimizer: Optimizer,
) -> float:
    """Train for one pass over a data loader and return average sample loss."""
    total_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = inputs.data.shape[0]
        total_loss += loss.data.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("cannot train on an empty dataset")
    return total_loss / total_samples


def evaluate(
    model: Module,
    dataloader: DataLoader,
    loss_fn: LossFunction,
    accuracy_fn: AccuracyFunction | None = None,
) -> float | tuple[float, float]:
    """Evaluate average loss and optionally accuracy without updating parameters."""
    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        batch_size = inputs.data.shape[0]

        total_loss += loss.data.item() * batch_size
        if accuracy_fn is not None:
            total_accuracy += accuracy_fn(predictions, targets) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("cannot evaluate an empty dataset")

    average_loss = total_loss / total_samples
    if accuracy_fn is None:
        return average_loss
    return average_loss, total_accuracy / total_samples
