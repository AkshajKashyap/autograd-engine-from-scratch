import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "examples" / "train_spiral.py"
SPEC = importlib.util.spec_from_file_location("train_spiral", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train_spiral_example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_spiral_example)


def test_spiral_training_core_runs_and_reduces_loss():
    features, labels = train_spiral_example.make_spiral_data(
        points_per_class=8,
        noise=0.15,
        seed=4,
    )
    model, losses = train_spiral_example.train_spiral(features, labels, steps=25, seed=4)

    assert len(losses) == 25
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    accuracy = train_spiral_example.classification_accuracy(model, features, labels)
    assert 0 <= accuracy <= 1
