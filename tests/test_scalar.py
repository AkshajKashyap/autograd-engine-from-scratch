import math

import pytest

from autograd_engine import Value


def test_basic_forward_operations():
    x = Value(6)

    assert (x + 2).data == pytest.approx(8)
    assert (2 + x).data == pytest.approx(8)
    assert (x - 2).data == pytest.approx(4)
    assert (10 - x).data == pytest.approx(4)
    assert (x * 2).data == pytest.approx(12)
    assert (2 * x).data == pytest.approx(12)
    assert (x / 2).data == pytest.approx(3)
    assert (12 / x).data == pytest.approx(2)
    assert (-x).data == pytest.approx(-6)
    assert (x**2).data == pytest.approx(36)


def test_gradients_for_simple_expression():
    x = Value(2)
    y = Value(-3)
    output = x * y + x**2

    output.backward()

    assert output.data == pytest.approx(-2)
    assert x.grad == pytest.approx(1)
    assert y.grad == pytest.approx(2)


def test_gradient_accumulates_when_value_is_reused():
    x = Value(3)
    output = x * x + x

    output.backward()

    assert x.grad == pytest.approx(7)


@pytest.mark.parametrize(
    ("input_value", "expected_output", "expected_grad"),
    [
        (2.0, 2.0, 1.0),
        (-2.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_relu(input_value, expected_output, expected_grad):
    x = Value(input_value)
    output = x.relu()

    output.backward()

    assert output.data == pytest.approx(expected_output)
    assert x.grad == pytest.approx(expected_grad)


def test_tanh_log_and_exp():
    x = Value(0.5)
    output = x.exp().log().tanh()

    output.backward()

    expected_tanh = math.tanh(0.5)
    assert output.data == pytest.approx(expected_tanh)
    assert x.grad == pytest.approx(1 - expected_tanh**2)


def test_multi_operation_gradient_matches_finite_difference():
    def expression(number):
        return ((number * number + 2 * number - 1).tanh() + (number + 3).log()) / 2

    x = Value(0.7)
    output = expression(x)
    output.backward()

    epsilon = 1e-6
    forward = expression(Value(x.data + epsilon)).data
    backward = expression(Value(x.data - epsilon)).data
    numerical_grad = (forward - backward) / (2 * epsilon)

    assert x.grad == pytest.approx(numerical_grad, rel=1e-5, abs=1e-7)
