import gzip
from pathlib import Path
import struct

import numpy as np

from autograd_engine.datasets import MNIST_FILES, _read_idx_images, _read_idx_labels, load_mnist


def _write_images(path: Path, images: np.ndarray) -> None:
    count, rows, columns = images.shape
    with gzip.open(path, "wb") as file:
        file.write(struct.pack(">IIII", 2051, count, rows, columns))
        file.write(images.astype(np.uint8).tobytes())


def _write_labels(path: Path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as file:
        file.write(struct.pack(">II", 2049, len(labels)))
        file.write(labels.astype(np.uint8).tobytes())


def _write_tiny_mnist(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    directory.mkdir()
    train_images = np.array(
        [
            [[0, 64], [128, 255]],
            [[255, 128], [64, 0]],
            [[1, 2], [3, 4]],
        ],
        dtype=np.uint8,
    )
    train_labels = np.array([1, 2, 3], dtype=np.uint8)
    test_images = train_images[:2]
    test_labels = train_labels[:2]

    _write_images(directory / MNIST_FILES["train_images"], train_images)
    _write_labels(directory / MNIST_FILES["train_labels"], train_labels)
    _write_images(directory / MNIST_FILES["test_images"], test_images)
    _write_labels(directory / MNIST_FILES["test_labels"], test_labels)
    return train_images, train_labels


def test_idx_helpers_parse_tiny_files(tmp_path):
    images = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    labels = np.array([2, 1, 0], dtype=np.uint8)
    image_path = tmp_path / "images.gz"
    label_path = tmp_path / "labels.gz"
    _write_images(image_path, images)
    _write_labels(label_path, labels)

    np.testing.assert_array_equal(_read_idx_images(image_path), images)
    np.testing.assert_array_equal(_read_idx_labels(label_path), labels)


def test_load_mnist_normalizes_flattens_and_limits_local_files(tmp_path):
    images, labels = _write_tiny_mnist(tmp_path / "mnist")

    x_train, y_train, x_test, y_test = load_mnist(
        tmp_path / "mnist",
        limit_train=2,
        limit_test=1,
    )

    assert x_train.shape == (2, 4)
    assert x_test.shape == (1, 4)
    assert x_train.dtype == float
    np.testing.assert_allclose(x_train[0], images[0].reshape(-1) / 255)
    np.testing.assert_array_equal(y_train, labels[:2])
    np.testing.assert_array_equal(y_test, labels[:1])


def test_load_mnist_can_preserve_image_shape_and_byte_values(tmp_path):
    images, _ = _write_tiny_mnist(tmp_path / "mnist")

    x_train, _, x_test, _ = load_mnist(
        tmp_path / "mnist",
        normalize=False,
        flatten=False,
    )

    assert x_train.shape == (3, 2, 2)
    assert x_test.shape == (2, 2, 2)
    assert x_train.dtype == np.uint8
    np.testing.assert_array_equal(x_train, images)
