import numpy as np
import pytest

from autograd_engine import Tensor


def test_forward_arithmetic_operations():
    x = Tensor(np.array([2.0, 4.0]))

    np.testing.assert_allclose((x + 2).data, [4, 6])
    np.testing.assert_allclose((2 + x).data, [4, 6])
    np.testing.assert_allclose((x - 1).data, [1, 3])
    np.testing.assert_allclose((5 - x).data, [3, 1])
    np.testing.assert_allclose((-x).data, [-2, -4])
    np.testing.assert_allclose((x * 3).data, [6, 12])
    np.testing.assert_allclose((12 / x).data, [6, 3])
    np.testing.assert_allclose((x / 2).data, [1, 2])
    np.testing.assert_allclose((x**2).data, [4, 16])


def test_forward_matrix_multiplication_and_reductions():
    left = Tensor([[1, 2], [3, 4]])
    right = Tensor([[2, 0], [1, 2]])

    np.testing.assert_allclose((left @ right).data, [[4, 4], [10, 8]])
    assert left.sum().data == pytest.approx(10)
    np.testing.assert_allclose(left.sum(axis=0).data, [4, 6])
    np.testing.assert_allclose(left.mean(axis=1, keepdims=True).data, [[1.5], [3.5]])


def test_simple_elementwise_gradients_and_reuse():
    x = Tensor([1.0, 2.0, 3.0])
    output = (x * x + x).sum()

    output.backward()

    np.testing.assert_allclose(x.grad, [3, 5, 7])


def test_matrix_multiplication_gradients():
    left = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    right = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    output = (left @ right).sum()

    output.backward()

    np.testing.assert_allclose(left.grad, np.ones((2, 2)) @ right.data.T)
    np.testing.assert_allclose(right.grad, left.data.T @ np.ones((2, 2)))


def test_sum_and_mean_gradients():
    x = Tensor(np.arange(6).reshape(2, 3))
    output = x.sum(axis=0).mean()

    output.backward()

    np.testing.assert_allclose(x.grad, np.full((2, 3), 1 / 3))


def test_broadcasting_addition_gradients():
    matrix = Tensor(np.arange(6).reshape(2, 3))
    vector = Tensor([10.0, 20.0, 30.0])
    output = (matrix + vector).sum()

    output.backward()

    np.testing.assert_allclose(matrix.grad, np.ones((2, 3)))
    np.testing.assert_allclose(vector.grad, [2, 2, 2])


def test_bias_style_broadcasting_gradients():
    values = Tensor(np.ones((4, 3)))
    bias = Tensor(np.zeros((1, 3)))
    output = ((values + bias) * 2).sum()

    output.backward()

    np.testing.assert_allclose(values.grad, np.full((4, 3), 2))
    np.testing.assert_allclose(bias.grad, np.full((1, 3), 8))


def test_scalar_multiplication_broadcasting_gradient():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    scale = Tensor(3.0)
    output = (x * scale).sum()

    output.backward()

    np.testing.assert_allclose(x.grad, np.full((2, 2), 3))
    assert scale.grad == pytest.approx(10)


def test_nonlinear_operations_and_gradients():
    x = Tensor([-1.0, 0.0, 1.0])
    relu = x.relu()
    relu.sum().backward()

    np.testing.assert_allclose(relu.data, [0, 0, 1])
    np.testing.assert_allclose(x.grad, [0, 0, 1])

    y = Tensor([0.2, 0.5, 1.0])
    output = y.exp().log().tanh().sum()
    output.backward()

    expected = np.tanh(y.data)
    np.testing.assert_allclose(output.data, expected.sum())
    np.testing.assert_allclose(y.grad, 1 - expected**2)


def test_multi_operation_gradient_matches_finite_difference():
    initial = np.array([[0.2, 0.5, 0.8], [1.1, 1.4, 1.7]])

    def tensor_expression(data):
        return ((data * data + data / 2).tanh().mean() + (data + 2).log().sum())

    x = Tensor(initial.copy())
    output = tensor_expression(x)
    output.backward()

    epsilon = 1e-6
    numerical_grad = np.zeros_like(initial)
    for index in np.ndindex(initial.shape):
        plus = initial.copy()
        minus = initial.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        forward = tensor_expression(Tensor(plus)).data
        backward = tensor_expression(Tensor(minus)).data
        numerical_grad[index] = (forward - backward) / (2 * epsilon)

    np.testing.assert_allclose(x.grad, numerical_grad, rtol=1e-5, atol=1e-7)
