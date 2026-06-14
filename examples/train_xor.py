"""Train a small MLP to model the XOR truth table."""

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
from autograd_engine import Adam, MLP, Sigmoid, Tanh, Tensor, binary_cross_entropy  # noqa: E402


def main() -> None:
    np.random.seed(11)
    inputs = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = Tensor([[0.0], [1.0], [1.0], [0.0]])
    model = MLP([2, 8, 1], activation=Tanh, output_activation=Sigmoid)
    optimizer = Adam(model.parameters(), learning_rate=0.05)

    losses: list[float] = []
    initial_loss = binary_cross_entropy(model(inputs), targets).data.item()

    for _ in range(1_000):
        loss = binary_cross_entropy(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())

    predictions = model(inputs).data
    final_loss = binary_cross_entropy(model(inputs), targets).data.item()
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")
    print("Final predictions:")
    print(np.round(predictions, 4))

    figure_path = ROOT / "reports" / "figures" / "xor_loss.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Training step")
    plt.ylabel("Binary cross-entropy")
    plt.title("XOR training")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()

    prediction_text = np.array2string(np.round(predictions, 4))
    report_path = ROOT / "reports" / "xor_experiment.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# XOR Experiment",
                "",
                "A small MLP with one hidden layer was trained on the XOR truth table.",
                "",
                f"- Initial BCE: `{initial_loss:.6f}`",
                f"- Final BCE: `{final_loss:.6f}`",
                "- Optimizer: Adam",
                "- Training steps: 1000",
                "",
                "Final predictions:",
                "",
                "```text",
                prediction_text,
                "```",
                "",
                "![XOR loss](figures/xor_loss.png)",
                "",
            ]
        )
    )


if __name__ == "__main__":
    main()
