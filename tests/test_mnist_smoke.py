import importlib.util
from pathlib import Path

import numpy as np

from autograd_engine import (
    Adam,
    DataLoader,
    Tensor,
    TensorDataset,
    cross_entropy,
    train_epoch,
)


SCRIPT_PATH = Path(__file__).parents[1] / "examples" / "train_mnist_mlp.py"
SPEC = importlib.util.spec_from_file_location("train_mnist_mlp", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
mnist_example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mnist_example)


def test_mnist_model_forward_shape():
    model = mnist_example.build_model(hidden_features=8, seed=2)

    output = model(Tensor(np.zeros((4, 784))))

    assert output.data.shape == (4, 10)


def test_fake_mnist_training_step_has_finite_loss():
    rng = np.random.default_rng(3)
    inputs = rng.random((6, 784))
    labels = np.array([0, 1, 2, 3, 4, 5])
    dataloader = DataLoader(TensorDataset(inputs, labels), batch_size=3)
    model = mnist_example.build_model(hidden_features=8, seed=3)
    optimizer = Adam(model.parameters(), learning_rate=0.001)

    loss = train_epoch(model, dataloader, cross_entropy, optimizer)

    assert np.isfinite(loss)
