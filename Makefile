.PHONY: install test lint check examples mnist-mlp mnist-cnn clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

check: lint test

examples:
	python examples/run_all.py

mnist-mlp:
	python examples/train_mnist_mlp.py

mnist-cnn:
	python examples/train_mnist_cnn.py

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete
