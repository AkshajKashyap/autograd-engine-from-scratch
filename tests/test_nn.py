import numpy as np

from autograd_engine import Linear, MLP, ReLU, Sequential, Sigmoid, Tanh, Tensor


def test_linear_forward_shape_and_parameters():
    layer = Linear(3, 2)
    output = layer(Tensor(np.ones((4, 3))))

    assert output.data.shape == (4, 2)
    assert layer.parameters() == [layer.weight, layer.bias]
    assert layer.weight.data.shape == (3, 2)
    assert layer.bias.data.shape == (2,)
    assert np.any(layer.weight.data != 0)


def test_sequential_forward_chains_modules():
    model = Sequential(Linear(3, 4), ReLU(), Linear(4, 2), Tanh())
    output = model(Tensor(np.ones((5, 3))))

    assert output.data.shape == (5, 2)
    assert np.all(output.data >= -1)
    assert np.all(output.data <= 1)


def test_sigmoid_forward_values():
    output = Sigmoid()(Tensor([-1.0, 0.0, 1.0]))
    expected = 1 / (1 + np.exp(-np.array([-1.0, 0.0, 1.0])))

    np.testing.assert_allclose(output.data, expected)


def test_mlp_builds_expected_layers_and_parameters():
    model = MLP([3, 5, 4, 2], activation=Tanh)

    assert len(model.modules) == 5
    assert sum(isinstance(module, Linear) for module in model.modules) == 3
    assert len(model.parameters()) == 6
    assert model(Tensor(np.ones((7, 3)))).data.shape == (7, 2)


def test_module_zero_grad_clears_all_parameter_gradients():
    model = MLP([2, 3, 1])
    loss = model(Tensor([[1.0, 2.0]])).sum()
    loss.backward()

    assert any(np.any(parameter.grad != 0) for parameter in model.parameters())

    model.zero_grad()

    assert all(np.all(parameter.grad == 0) for parameter in model.parameters())
