import numpy as np
import pytest

from autograd_engine import Tensor, binary_cross_entropy, mse_loss


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
