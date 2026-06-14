"""Train a small MLP on deterministic synthetic regression data."""

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
from autograd_engine import Adam, MLP, Tanh, Tensor, mse_loss  # noqa: E402


def run_experiment(
    steps: int = 300,
    *,
    write_artifacts: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """Train the regression model and return its headline metrics."""
    np.random.seed(7)
    x_data = np.linspace(-1.0, 1.0, 80).reshape(-1, 1)
    noise = np.random.normal(0.0, 0.08, size=x_data.shape)
    y_data = 2 * x_data - 1 + noise

    inputs = Tensor(x_data)
    targets = Tensor(y_data)
    model = MLP([1, 12, 1], activation=Tanh)
    optimizer = Adam(model.parameters(), learning_rate=0.03)

    losses: list[float] = []
    initial_loss = mse_loss(model(inputs), targets).data.item()

    for _ in range(steps):
        loss = mse_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())

    final_loss = mse_loss(model(inputs), targets).data.item()
    if verbose:
        print(f"Initial loss: {initial_loss:.6f}")
        print(f"Final loss:   {final_loss:.6f}")

    if write_artifacts:
        figure_path = ROOT / "reports" / "figures" / "regression_loss.png"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(losses)
        plt.xlabel("Training step")
        plt.ylabel("MSE loss")
        plt.title("Synthetic regression training")
        plt.tight_layout()
        plt.savefig(figure_path)
        plt.close()

        report_path = ROOT / "reports" / "regression_experiment.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "\n".join(
                [
                    "# Regression Experiment",
                    "",
                    "A small MLP was trained on noisy samples from `y = 2x - 1`.",
                    "",
                    f"- Initial MSE: `{initial_loss:.6f}`",
                    f"- Final MSE: `{final_loss:.6f}`",
                    "- Optimizer: Adam",
                    f"- Training steps: {steps}",
                    "",
                    "![Regression loss](figures/regression_loss.png)",
                    "",
                ]
            )
        )

    return {"initial_loss": initial_loss, "final_loss": final_loss}


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
