import numpy as np

from autograd_engine import Adam, Linear, MomentumSGD, Parameter, SGD, Tensor


def test_sgd_updates_parameter_and_supports_weight_decay():
    parameter = Parameter([1.0, -2.0])
    parameter.grad = np.array([0.5, -0.25])
    optimizer = SGD([parameter], learning_rate=0.1, weight_decay=0.2)

    optimizer.step()

    np.testing.assert_allclose(parameter.data, [0.93, -1.935])


def test_optimizer_zero_grad_clears_gradients():
    parameter = Parameter([1.0, 2.0])
    parameter.grad = np.array([3.0, 4.0])
    optimizer = SGD([parameter])

    optimizer.zero_grad()

    np.testing.assert_allclose(parameter.grad, [0, 0])


def test_momentum_sgd_accumulates_velocity_across_steps():
    parameter = Parameter([1.0])
    optimizer = MomentumSGD([parameter], learning_rate=0.1, momentum=0.9)

    parameter.grad.fill(1.0)
    optimizer.step()
    first_value = parameter.data.copy()
    optimizer.step()

    np.testing.assert_allclose(first_value, [0.9])
    np.testing.assert_allclose(parameter.data, [0.71])
    np.testing.assert_allclose(optimizer.velocities[0], [1.9])


def test_adam_updates_parameter_and_maintains_state():
    parameter = Parameter([1.0, -1.0])
    parameter.grad = np.array([0.5, -0.25])
    optimizer = Adam([parameter], learning_rate=0.1)

    optimizer.step()

    np.testing.assert_allclose(parameter.data, [0.9, -0.9])
    assert optimizer.step_count == 1
    assert np.any(optimizer.first_moments[0] != 0)
    assert np.any(optimizer.second_moments[0] != 0)


def test_tiny_linear_model_reduces_regression_loss():
    np.random.seed(0)
    inputs = Tensor(np.linspace(-1, 1, 20).reshape(-1, 1))
    targets = Tensor(3 * inputs.data - 0.5)
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), learning_rate=0.2)

    initial_loss = ((model(inputs) - targets) ** 2).mean().data.item()

    for _ in range(30):
        predictions = model(inputs)
        loss = ((predictions - targets) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = ((model(inputs) - targets) ** 2).mean().data.item()

    assert final_loss < initial_loss * 0.01
