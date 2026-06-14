"""Save and restore model parameters without pickle."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from autograd_engine.nn import Module


def state_dict(model: Module) -> dict[str, np.ndarray]:
    """Return deterministic copies of a model's parameter arrays."""
    return {
        f"parameter_{index}": parameter.data.copy()
        for index, parameter in enumerate(model.parameters())
    }


def load_state_dict(model: Module, state: Mapping[str, np.ndarray]) -> None:
    """Load parameter arrays after validating keys and shapes."""
    parameters = model.parameters()
    expected_keys = [f"parameter_{index}" for index in range(len(parameters))]
    if set(state) != set(expected_keys):
        raise ValueError(f"state keys must be exactly {expected_keys}")

    for key, parameter in zip(expected_keys, parameters, strict=True):
        value = np.asarray(state[key])
        if value.shape != parameter.data.shape:
            raise ValueError(
                f"shape mismatch for {key}: expected {parameter.data.shape}, got {value.shape}"
            )
        parameter.data[...] = value


def save_state_dict(model: Module, path: str | Path) -> None:
    """Save model parameters to a NumPy .npz archive."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        np.savez(file, **state_dict(model))


def load_state_dict_from_file(model: Module, path: str | Path) -> None:
    """Load model parameters from a NumPy .npz archive."""
    with np.load(Path(path), allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    load_state_dict(model, state)
