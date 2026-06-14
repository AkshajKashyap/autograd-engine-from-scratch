"""Run every training example and print a compact result summary."""

from train_regression import run_experiment as run_regression
from train_spiral import run_experiment as run_spiral
from train_xor import run_experiment as run_xor


def main() -> None:
    experiments = [
        ("Regression", run_regression),
        ("XOR", run_xor),
        ("Spiral", run_spiral),
    ]
    results: list[tuple[str, dict[str, float]]] = []

    for name, run in experiments:
        print(f"Running {name.lower()} experiment...")
        results.append((name, run(verbose=False)))

    print()
    print(f"{'Experiment':<12} {'Initial loss':>14} {'Final loss':>14} {'Accuracy':>12}")
    print("-" * 56)
    for name, result in results:
        accuracy = result.get("accuracy")
        accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "-"
        print(
            f"{name:<12} "
            f"{result['initial_loss']:>14.6f} "
            f"{result['final_loss']:>14.6f} "
            f"{accuracy_text:>12}"
        )


if __name__ == "__main__":
    main()
