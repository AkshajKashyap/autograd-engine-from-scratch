import numpy as np

from autograd_engine import (
    DataLoader,
    History,
    Linear,
    SGD,
    TensorDataset,
    binary_accuracy,
    evaluate,
    mse_loss,
    train_epoch,
)


def test_dataloader_batches_have_expected_shapes():
    dataset = TensorDataset(np.arange(20).reshape(10, 2), np.arange(10).reshape(10, 1))
    dataloader = DataLoader(dataset, batch_size=4)
    batches = list(dataloader)

    assert len(dataloader) == 3
    assert [inputs.data.shape for inputs, _ in batches] == [(4, 2), (4, 2), (2, 2)]
    assert [targets.data.shape for _, targets in batches] == [(4, 1), (4, 1), (2, 1)]


def test_dataloader_shuffling_is_deterministic_with_seed():
    dataset = TensorDataset(np.arange(12).reshape(6, 2), np.arange(6))
    first_loader = DataLoader(dataset, batch_size=2, shuffle=True, seed=17)
    second_loader = DataLoader(dataset, batch_size=2, shuffle=True, seed=17)

    first_order = np.concatenate([targets.data for _, targets in first_loader])
    second_order = np.concatenate([targets.data for _, targets in second_loader])

    np.testing.assert_array_equal(first_order, second_order)
    assert not np.array_equal(first_order, np.arange(6))


def test_train_epoch_reduces_tiny_regression_loss():
    np.random.seed(2)
    inputs = np.linspace(-1, 1, 20).reshape(-1, 1)
    targets = 2 * inputs - 0.5
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=5, shuffle=True, seed=3)
    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), learning_rate=0.15)

    initial_loss = evaluate(model, dataloader, mse_loss)
    for _ in range(12):
        train_epoch(model, dataloader, mse_loss, optimizer)
    final_loss = evaluate(model, dataloader, mse_loss)

    assert isinstance(initial_loss, float)
    assert isinstance(final_loss, float)
    assert final_loss < initial_loss * 0.05


def test_evaluate_returns_finite_loss_without_changing_parameters():
    np.random.seed(4)
    model = Linear(2, 1)
    dataloader = DataLoader(
        TensorDataset(np.ones((6, 2)), np.zeros((6, 1))),
        batch_size=2,
    )
    before = [parameter.data.copy() for parameter in model.parameters()]

    loss = evaluate(model, dataloader, mse_loss)

    assert np.isfinite(loss)
    for parameter, original in zip(model.parameters(), before, strict=True):
        np.testing.assert_array_equal(parameter.data, original)


def test_evaluate_optionally_returns_accuracy():
    model = Linear(1, 1)
    model.weight.data[...] = 1
    model.bias.data[...] = 0
    dataloader = DataLoader(
        TensorDataset([[0.0], [1.0]], [[0.0], [1.0]]),
        batch_size=2,
    )

    loss, accuracy = evaluate(model, dataloader, mse_loss, binary_accuracy)

    assert loss == 0
    assert accuracy == 1


def test_history_stores_epoch_metrics():
    history = History()

    history.append(train_loss=1.0, val_loss=1.2, accuracy=0.75)

    assert history.train_loss == [1.0]
    assert history.val_loss == [1.2]
    assert history.accuracy == [0.75]
