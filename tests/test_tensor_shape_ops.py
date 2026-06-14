import numpy as np

from autograd_engine import Tensor


def test_reshape_forward_and_backward():
    x = Tensor(np.arange(6).reshape(2, 3))
    reshaped = x.reshape(3, 2)
    loss = (reshaped * Tensor([[1, 2], [3, 4], [5, 6]])).sum()

    loss.backward()

    assert reshaped.data.shape == (3, 2)
    np.testing.assert_array_equal(reshaped.data, np.arange(6).reshape(3, 2))
    np.testing.assert_array_equal(x.grad, [[1, 2, 3], [4, 5, 6]])


def test_transpose_forward_and_backward():
    x = Tensor(np.arange(6).reshape(2, 3))
    transposed = x.transpose(1, 0)
    weights = Tensor([[1, 2], [3, 4], [5, 6]])

    (transposed * weights).sum().backward()

    np.testing.assert_array_equal(transposed.data, x.data.T)
    np.testing.assert_array_equal(x.grad, weights.data.T)


def test_matrix_transpose_property():
    x = Tensor([[1, 2], [3, 4], [5, 6]])

    np.testing.assert_array_equal(x.T.data, [[1, 3, 5], [2, 4, 6]])
