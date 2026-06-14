import numpy as np

from autograd_engine import (
    Adam,
    Conv2D,
    DataLoader,
    Flatten,
    Linear,
    MaxPool2D,
    ReLU,
    Sequential,
    Tensor,
    TensorDataset,
    cross_entropy,
    gradcheck,
    train_epoch,
)


def test_flatten_forward_shape():
    output = Flatten()(Tensor(np.zeros((3, 2, 4, 5))))

    assert output.data.shape == (3, 40)


def test_conv2d_forward_and_gradient_shapes():
    np.random.seed(1)
    layer = Conv2D(2, 3, kernel_size=3, padding=1)
    inputs = Tensor(np.random.randn(2, 2, 4, 5))
    output = layer(inputs)

    output.sum().backward()

    assert output.data.shape == (2, 3, 4, 5)
    assert inputs.grad.shape == inputs.data.shape
    assert layer.weight.grad.shape == layer.weight.data.shape
    assert layer.bias.grad.shape == layer.bias.data.shape
    assert np.all(np.isfinite(inputs.grad))
    assert np.all(np.isfinite(layer.weight.grad))
    assert np.all(np.isfinite(layer.bias.grad))


def test_conv2d_input_gradient_matches_finite_difference():
    np.random.seed(2)
    layer = Conv2D(1, 1, kernel_size=2, bias=False)
    inputs = Tensor(np.array([[[[0.2, -0.1, 0.4], [0.7, 0.3, -0.2], [0.5, 0.1, 0.6]]]]))

    result = gradcheck(lambda value: (layer(value) ** 2).sum(), [inputs])

    assert result.passed
    assert result.max_absolute_error < 1e-5


def test_conv2d_weight_gradient_matches_finite_difference():
    layer = Conv2D(1, 1, kernel_size=2, bias=False)
    layer.weight.data[...] = [[[[0.2, -0.3], [0.4, 0.1]]]]
    inputs = Tensor([[[[0.5, 0.2, -0.1], [0.3, 0.7, 0.4], [-0.2, 0.1, 0.6]]]])
    output = (layer(inputs) ** 2).sum()
    output.backward()
    analytical = layer.weight.grad[0, 0, 0, 1]

    epsilon = 1e-6
    original = layer.weight.data[0, 0, 0, 1]
    layer.weight.data[0, 0, 0, 1] = original + epsilon
    forward = (layer(Tensor(inputs.data)) ** 2).sum().data.item()
    layer.weight.data[0, 0, 0, 1] = original - epsilon
    backward = (layer(Tensor(inputs.data)) ** 2).sum().data.item()
    layer.weight.data[0, 0, 0, 1] = original

    assert np.isclose(analytical, (forward - backward) / (2 * epsilon), atol=1e-6)


def test_max_pool_forward_and_backward():
    inputs = Tensor(
        [
            [
                [
                    [1.0, 3.0, 2.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [0.0, 2.0, 9.0, 1.0],
                    [3.0, 4.0, 5.0, 6.0],
                ]
            ]
        ]
    )
    output = MaxPool2D(2)(inputs)

    output.sum().backward()

    np.testing.assert_array_equal(output.data, [[[[6, 8], [4, 9]]]])
    np.testing.assert_array_equal(
        inputs.grad,
        [
            [
                [
                    [0, 0, 0, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                ]
            ]
        ],
    )


def _tiny_cnn() -> Sequential:
    return Sequential(
        Conv2D(1, 2, kernel_size=3),
        ReLU(),
        MaxPool2D(2),
        Flatten(),
        Linear(8, 3),
    )


def test_tiny_cnn_forward_shape():
    np.random.seed(3)
    model = _tiny_cnn()

    logits = model(Tensor(np.zeros((4, 1, 6, 6))))

    assert logits.data.shape == (4, 3)


def test_tiny_cnn_training_step_has_finite_loss():
    np.random.seed(4)
    inputs = np.random.randn(4, 1, 6, 6)
    labels = np.array([0, 1, 2, 1])
    dataloader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
    model = _tiny_cnn()
    optimizer = Adam(model.parameters(), learning_rate=0.001)

    loss = train_epoch(model, dataloader, cross_entropy, optimizer)

    assert np.isfinite(loss)
