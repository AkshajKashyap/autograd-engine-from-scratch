import numpy as np
import pytest

from autograd_engine import Tensor, binary_accuracy, multiclass_accuracy


def test_binary_accuracy():
    predictions = Tensor([[0.1], [0.7], [0.8], [0.4]])
    targets = np.array([[0], [1], [0], [0]])

    assert binary_accuracy(predictions, targets) == pytest.approx(0.75)


def test_binary_accuracy_supports_custom_threshold():
    predictions = Tensor([0.2, 0.6, 0.8])

    assert binary_accuracy(predictions, [0, 0, 1], threshold=0.7) == pytest.approx(1.0)


def test_multiclass_accuracy_with_integer_labels():
    logits = Tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 4.0]])

    assert multiclass_accuracy(logits, [0, 2, 2]) == pytest.approx(2 / 3)


def test_multiclass_accuracy_with_one_hot_labels():
    logits = Tensor([[3.0, 1.0], [0.0, 2.0]])
    targets = np.array([[1, 0], [0, 1]])

    assert multiclass_accuracy(logits, targets) == pytest.approx(1.0)
