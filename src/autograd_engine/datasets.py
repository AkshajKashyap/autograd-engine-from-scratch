"""Dataset download and parsing helpers."""

import gzip
from pathlib import Path
import shutil
import struct
import tempfile
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f"{destination.name}.",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urlopen(url, timeout=30) as response:  # noqa: S310
                shutil.copyfileobj(response, temporary_file)
        temporary_path.replace(destination)
    except (OSError, URLError) as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download MNIST file from {url}. "
            f"Download it manually and place it at {destination}."
        ) from error


def _read_idx_images(path: str | Path) -> np.ndarray:
    """Read gzip-compressed IDX image data."""
    with gzip.open(path, "rb") as file:
        header = file.read(16)
        if len(header) != 16:
            raise ValueError(f"Invalid IDX image header in {path}")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Invalid IDX image magic number in {path}: {magic}")

        raw_data = file.read()
        expected_size = count * rows * columns
        if len(raw_data) != expected_size:
            raise ValueError(
                f"Invalid IDX image payload in {path}: "
                f"expected {expected_size} bytes, got {len(raw_data)}"
            )

    return np.frombuffer(raw_data, dtype=np.uint8).reshape(count, rows, columns).copy()


def _read_idx_labels(path: str | Path) -> np.ndarray:
    """Read gzip-compressed IDX label data."""
    with gzip.open(path, "rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid IDX label header in {path}")
        magic, count = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Invalid IDX label magic number in {path}: {magic}")

        raw_data = file.read()
        if len(raw_data) != count:
            raise ValueError(
                f"Invalid IDX label payload in {path}: expected {count} bytes, got {len(raw_data)}"
            )

    return np.frombuffer(raw_data, dtype=np.uint8).copy()


def load_mnist(
    data_dir: str | Path = "data/raw/mnist",
    limit_train: int | None = None,
    limit_test: int | None = None,
    normalize: bool = True,
    flatten: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download if needed, parse, and return the MNIST train/test splits."""
    if limit_train is not None and limit_train < 0:
        raise ValueError("limit_train must be non-negative")
    if limit_test is not None and limit_test < 0:
        raise ValueError("limit_test must be non-negative")

    directory = Path(data_dir)
    paths = {name: directory / filename for name, filename in MNIST_FILES.items()}
    for path in paths.values():
        if not path.exists():
            _download_file(f"{MNIST_BASE_URL}/{path.name}", path)

    x_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    x_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("MNIST image and label counts do not match")

    x_train = x_train[:limit_train]
    y_train = y_train[:limit_train]
    x_test = x_test[:limit_test]
    y_test = y_test[:limit_test]

    if normalize:
        x_train = x_train.astype(float) / 255.0
        x_test = x_test.astype(float) / 255.0
    if flatten:
        x_train = x_train.reshape(len(x_train), -1)
        x_test = x_test.reshape(len(x_test), -1)

    return x_train, y_train, x_test, y_test
