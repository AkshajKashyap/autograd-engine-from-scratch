import numpy as np

from autograd_engine import (
    Adam,
    Linear,
    MLP,
    SGD,
    Sigmoid,
    Tanh,
    Tensor,
    binary_cross_entropy,
    cross_entropy,
    mse_loss,
)


def test_tiny_regression_model_reduces_loss():
    np.random.seed(3)
    inputs = Tensor(np.linspace(-1, 1, 20).reshape(-1, 1))
    targets = 2 * inputs.data - 1
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), learning_rate=0.2)

    initial_loss = mse_loss(model(inputs), targets).data.item()
    for _ in range(25):
        loss = mse_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = mse_loss(model(inputs), targets).data.item()

    assert final_loss < initial_loss * 0.02


def test_tiny_xor_model_trains_with_finite_loss():
    np.random.seed(5)
    inputs = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = Tensor([[0.0], [1.0], [1.0], [0.0]])
    model = MLP([2, 4, 1], activation=Tanh, output_activation=Sigmoid)
    optimizer = Adam(model.parameters(), learning_rate=0.05)

    initial_loss = binary_cross_entropy(model(inputs), targets).data.item()
    for _ in range(50):
        loss = binary_cross_entropy(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = binary_cross_entropy(model(inputs), targets).data.item()

    assert np.isfinite(final_loss)
    assert final_loss < initial_loss


def test_tiny_multiclass_model_reduces_loss():
    np.random.seed(9)
    features = np.array(
        [
            [-1.0, -1.0],
            [-0.8, -1.2],
            [1.0, -1.0],
            [0.8, -1.1],
            [0.0, 1.0],
            [0.2, 1.2],
        ]
    )
    labels = np.array([0, 0, 1, 1, 2, 2])
    inputs = Tensor(features)
    model = MLP([2, 8, 3], activation=Tanh)
    optimizer = Adam(model.parameters(), learning_rate=0.05)

    initial_loss = cross_entropy(model(inputs), labels).data.item()
    for _ in range(40):
        loss = cross_entropy(model(inputs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = cross_entropy(model(inputs), labels).data.item()

    assert np.isfinite(final_loss)
    assert final_loss < initial_loss * 0.25
