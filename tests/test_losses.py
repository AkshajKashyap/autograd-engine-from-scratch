import numpy as np
import pytest

from autograd_engine import (
    Tensor,
    binary_cross_entropy,
    cross_entropy,
    mse_loss,
    softmax,
)


def test_mse_loss_forward_value_and_gradients():
    predictions = Tensor([1.0, 2.0, 3.0])
    loss = mse_loss(predictions, np.array([0.0, 2.0, 4.0]))

    loss.backward()

    assert loss.data.shape == ()
    assert loss.data == pytest.approx(2 / 3)
    np.testing.assert_allclose(predictions.grad, [2 / 3, 0, -2 / 3])


def test_binary_cross_entropy_forward_value_and_gradients():
    predictions = Tensor([0.25, 0.75])
    loss = binary_cross_entropy(predictions, [0.0, 1.0])

    loss.backward()

    assert loss.data.shape == ()
    assert loss.data == pytest.approx(-np.log(0.75 + 1e-7))
    assert np.all(np.isfinite(predictions.grad))
    assert predictions.grad[0] > 0
    assert predictions.grad[1] < 0


def test_binary_cross_entropy_remains_finite_at_probability_boundaries():
    loss = binary_cross_entropy(Tensor([0.0, 1.0]), [0.0, 1.0])

    assert np.isfinite(loss.data)


def test_softmax_rows_sum_to_one():
    probabilities = softmax(Tensor([[1.0, 2.0, 3.0], [1000.0, 1001.0, 999.0]]))

    np.testing.assert_allclose(probabilities.data.sum(axis=1), [1.0, 1.0])
    assert np.all(np.isfinite(probabilities.data))


def test_cross_entropy_forward_value_and_finite_gradients():
    logits = Tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    loss = cross_entropy(logits, np.array([0, 2]))

    loss.backward()

    expected = -np.log(np.exp(2) / np.exp([2, 1, 0]).sum())
    assert loss.data == pytest.approx(expected)
    assert np.all(np.isfinite(logits.grad))


def test_integer_and_one_hot_cross_entropy_match():
    values = [[1.5, -0.5, 0.2], [0.1, 1.2, -0.3]]
    integer_loss = cross_entropy(Tensor(values), np.array([0, 1]))
    one_hot_loss = cross_entropy(Tensor(values), np.array([[1, 0, 0], [0, 1, 0]]))

    assert integer_loss.data == pytest.approx(one_hot_loss.data)
