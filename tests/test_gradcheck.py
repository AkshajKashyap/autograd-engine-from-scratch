import numpy as np

from autograd_engine import Tensor, finite_difference_grad, gradcheck


def test_gradcheck_passes_for_elementwise_expression():
    x = Tensor([[0.2, 0.5], [0.8, 1.1]])

    result = gradcheck(lambda value: (value.tanh() * value**2).sum(), [x])

    assert result.passed
    assert result.max_absolute_error < 1e-5
    assert len(result.inputs) == 1


def test_gradcheck_passes_for_matrix_multiplication():
    left = Tensor([[0.2, -0.4, 0.7], [1.0, 0.3, -0.2]])
    right = Tensor([[0.5, -0.1], [0.8, 0.4], [-0.3, 0.9]])

    result = gradcheck(lambda a, b: ((a @ b) ** 2).mean(), (left, right))

    assert result.passed
    assert all(input_result.passed for input_result in result.inputs)


def test_gradcheck_detects_an_intentionally_wrong_gradient():
    x = Tensor([0.5, 1.5])

    def wrong_square(value):
        output = value**2

        def wrong_backward():
            value.grad += np.zeros_like(value.data) * output.grad

        output._backward = wrong_backward
        return output.sum()

    result = gradcheck(wrong_square, [x])

    assert not result.passed
    assert result.max_absolute_error > 1


def test_finite_difference_gradient_shapes_match_inputs():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = Tensor([0.5, 1.5])

    gradients = finite_difference_grad(lambda a, b: (a * b).sum(), [x, y])

    assert gradients[0].shape == x.data.shape
    assert gradients[1].shape == y.data.shape


def test_gradcheck_restores_original_input_data():
    x = Tensor([[0.3, 0.7], [1.1, 1.4]])
    original = x.data.copy()

    gradcheck(lambda value: value.exp().sum(), [x])

    np.testing.assert_array_equal(x.data, original)
