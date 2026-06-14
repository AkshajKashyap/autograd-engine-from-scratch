"""Train a small multilayer perceptron on an MNIST subset."""

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
from autograd_engine import (  # noqa: E402
    Adam,
    DataLoader,
    History,
    MLP,
    ReLU,
    TensorDataset,
    cross_entropy,
    evaluate,
    load_mnist,
    multiclass_accuracy,
    save_state_dict,
    train_epoch,
)


def build_model(hidden_features: int = 64, seed: int = 23) -> MLP:
    """Build the default MNIST MLP with deterministic initialization."""
    np.random.seed(seed)
    return MLP([784, hidden_features, 10], activation=ReLU)


def run_experiment(
    limit_train: int = 5_000,
    limit_test: int = 1_000,
    epochs: int = 3,
    batch_size: int = 64,
    *,
    write_artifacts: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """Train the subset MNIST experiment and return its final metrics."""
    x_train, y_train, x_test, y_test = load_mnist(
        limit_train=limit_train,
        limit_test=limit_test,
    )
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        seed=23,
    )
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)

    model = build_model()
    optimizer = Adam(model.parameters(), learning_rate=0.002)
    history = History()

    initial_test_loss, initial_accuracy = evaluate(
        model,
        test_loader,
        cross_entropy,
        multiclass_accuracy,
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, cross_entropy, optimizer)
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            cross_entropy,
            multiclass_accuracy,
        )
        history.append(
            train_loss=train_loss,
            val_loss=test_loss,
            accuracy=test_accuracy,
        )
        if verbose:
            print(
                f"Epoch {epoch}/{epochs} - "
                f"train loss: {train_loss:.4f} - "
                f"test loss: {test_loss:.4f} - "
                f"test accuracy: {test_accuracy:.2%}"
            )

    final_test_loss = history.val_loss[-1]
    final_accuracy = history.accuracy[-1]

    if write_artifacts:
        figures_path = ROOT / "reports" / "figures"
        figures_path.mkdir(parents=True, exist_ok=True)

        epoch_numbers = np.arange(1, epochs + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(epoch_numbers, history.train_loss, marker="o", label="train")
        plt.plot(epoch_numbers, history.val_loss, marker="o", label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("MNIST MLP loss")
        plt.xticks(epoch_numbers)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_path / "mnist_mlp_loss.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(epoch_numbers, history.accuracy, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Test accuracy")
        plt.title("MNIST MLP accuracy")
        plt.xticks(epoch_numbers)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(figures_path / "mnist_mlp_accuracy.png")
        plt.close()

        save_state_dict(model, ROOT / "artifacts" / "mnist_mlp_state.npz")

        report_path = ROOT / "reports" / "mnist_mlp_experiment.md"
        report_path.write_text(
            "\n".join(
                [
                    "# MNIST MLP Experiment",
                    "",
                    "A small fully connected network was trained on a deterministic MNIST subset.",
                    "",
                    f"- Training samples: `{limit_train}`",
                    f"- Test samples: `{limit_test}`",
                    f"- Epochs: `{epochs}`",
                    f"- Initial test loss: `{initial_test_loss:.6f}`",
                    f"- Initial test accuracy: `{initial_accuracy:.2%}`",
                    f"- Final test loss: `{final_test_loss:.6f}`",
                    f"- Final test accuracy: `{final_accuracy:.2%}`",
                    "",
                    "This subset experiment prioritizes runtime and framework verification,",
                    "not state-of-the-art MNIST performance.",
                    "",
                    "![MNIST loss](figures/mnist_mlp_loss.png)",
                    "",
                    "![MNIST accuracy](figures/mnist_mlp_accuracy.png)",
                    "",
                ]
            )
        )

    return {
        "initial_loss": initial_test_loss,
        "final_loss": final_test_loss,
        "accuracy": final_accuracy,
    }


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
