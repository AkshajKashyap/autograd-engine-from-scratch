import numpy as np

from autograd_engine import (
    MLP,
    Tanh,
    Tensor,
    load_state_dict,
    load_state_dict_from_file,
    save_state_dict,
    state_dict,
)


def test_state_dict_has_deterministic_keys_and_copied_arrays():
    np.random.seed(1)
    model = MLP([2, 3, 1], activation=Tanh)

    first = state_dict(model)
    second = state_dict(model)

    assert list(first) == ["parameter_0", "parameter_1", "parameter_2", "parameter_3"]
    assert list(first) == list(second)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
        assert first[key] is not model.parameters()[int(key.split("_")[1])].data


def test_load_state_dict_restores_parameters_exactly():
    np.random.seed(2)
    model = MLP([2, 4, 1], activation=Tanh)
    saved = state_dict(model)

    for parameter in model.parameters():
        parameter.data += 10
    load_state_dict(model, saved)

    for key, parameter in zip(saved, model.parameters(), strict=True):
        np.testing.assert_array_equal(parameter.data, saved[key])


def test_file_round_trip_reproduces_predictions(tmp_path):
    np.random.seed(3)
    source = MLP([2, 5, 2], activation=Tanh)
    inputs = Tensor([[0.2, -0.5], [1.0, 0.3]])
    expected = source(inputs).data.copy()
    path = tmp_path / "model.npz"

    save_state_dict(source, path)

    np.random.seed(9)
    restored = MLP([2, 5, 2], activation=Tanh)
    load_state_dict_from_file(restored, path)

    np.testing.assert_array_equal(restored(inputs).data, expected)
