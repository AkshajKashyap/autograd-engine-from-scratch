"""Train an MLP on a deterministic three-class spiral dataset."""

import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "autograd-matplotlib"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from autograd_engine import Adam, MLP, Tanh, Tensor, cross_entropy  # noqa: E402


def make_spiral_data(
    points_per_class: int = 100,
    n_classes: int = 3,
    noise: float = 0.2,
    seed: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate interleaved 2D spiral arms."""
    rng = np.random.default_rng(seed)
    features = np.zeros((points_per_class * n_classes, 2))
    labels = np.zeros(points_per_class * n_classes, dtype=int)

    for class_index in range(n_classes):
        indices = slice(class_index * points_per_class, (class_index + 1) * points_per_class)
        radius = np.linspace(0.0, 1.0, points_per_class)
        angle = (
            np.linspace(class_index * 4, (class_index + 1) * 4, points_per_class)
            + rng.normal(0.0, noise, points_per_class)
        )
        features[indices] = np.column_stack((radius * np.sin(angle), radius * np.cos(angle)))
        labels[indices] = class_index

    return features, labels


def train_spiral(
    features: np.ndarray,
    labels: np.ndarray,
    steps: int = 800,
    seed: int = 21,
) -> tuple[MLP, list[float]]:
    """Train and return a small spiral classifier and its loss history."""
    np.random.seed(seed)
    inputs = Tensor(features)
    model = MLP([2, 32, 32, 3], activation=Tanh)
    optimizer = Adam(model.parameters(), learning_rate=0.03, weight_decay=1e-4)
    losses: list[float] = []

    for _ in range(steps):
        loss = cross_entropy(model(inputs), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())

    return model, losses


def classification_accuracy(model: MLP, features: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.argmax(model(Tensor(features)).data, axis=1)
    return float(np.mean(predictions == labels))


def main() -> None:
    features, labels = make_spiral_data()

    np.random.seed(21)
    initial_model = MLP([2, 32, 32, 3], activation=Tanh)
    initial_loss = cross_entropy(initial_model(Tensor(features)), labels).data.item()

    model, losses = train_spiral(features, labels)
    final_loss = cross_entropy(model(Tensor(features)), labels).data.item()
    final_accuracy = classification_accuracy(model, features, labels)

    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print(f"Final accuracy: {final_accuracy:.2%}")

    figures_path = ROOT / "reports" / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Spiral classification training")
    plt.tight_layout()
    plt.savefig(figures_path / "spiral_loss.png")
    plt.close()

    padding = 0.15
    x_min, x_max = features[:, 0].min() - padding, features[:, 0].max() + padding
    y_min, y_max = features[:, 1].min() - padding, features[:, 1].max() + padding
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    regions = np.argmax(model(Tensor(grid)).data, axis=1).reshape(grid_x.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(grid_x, grid_y, regions, levels=np.arange(4) - 0.5, alpha=0.3)
    plt.scatter(features[:, 0], features[:, 1], c=labels, s=18, edgecolors="black", linewidths=0.2)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Spiral decision boundary")
    plt.tight_layout()
    plt.savefig(figures_path / "spiral_decision_boundary.png")
    plt.close()

    report_path = ROOT / "reports" / "spiral_experiment.md"
    report_path.write_text(
        "\n".join(
            [
                "# Spiral Classification Experiment",
                "",
                "A multilayer perceptron was trained on a deterministic three-class spiral dataset.",
                "",
                f"- Initial cross-entropy: `{initial_loss:.6f}`",
                f"- Final cross-entropy: `{final_loss:.6f}`",
                f"- Final accuracy: `{final_accuracy:.2%}`",
                "- Optimizer: Adam",
                "- Training steps: 800",
                "",
                "![Spiral loss](figures/spiral_loss.png)",
                "",
                "![Spiral decision boundary](figures/spiral_decision_boundary.png)",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
