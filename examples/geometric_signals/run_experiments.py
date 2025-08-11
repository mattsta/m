"""
Main experiment runner for geometric signal learning.
Orchestrates training and evaluation across multiple configurations.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .evaluation import evaluate_model
from .training import train_from_config


def run_single_experiment(
    config_path: Path, experiment_name: str | None = None
) -> dict[str, Any]:
    """Run a single training experiment."""
    if experiment_name is None:
        experiment_name = config_path.stem

    print(f"\n{'=' * 60}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Config: {config_path}")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        # Train model
        training_results = train_from_config(config_path, experiment_name)

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final train loss: {training_results['final_train_loss']:.6f}")
        print(f"Final val loss: {training_results['final_val_loss']:.6f}")
        print(f"Best val loss: {training_results['best_val_loss']:.6f}")

        # Load config for evaluation
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Determine model paths
        output_dir = (
            Path(config.get("output_dir", "outputs/geometric_signals"))
            / experiment_name
        )
        model_path = output_dir / "best_model.pt"

        # Evaluate model if training succeeded
        if model_path.exists():
            print("\nStarting evaluation...")
            eval_start = time.time()

            eval_output_dir = output_dir / "evaluation"
            evaluation_results = evaluate_model(
                model_path, config_path, eval_output_dir
            )

            eval_time = time.time() - eval_start
            print(f"Evaluation completed in {eval_time:.2f} seconds")

            # Print evaluation summary
            print("\nEvaluation Results:")
            for dataset_name, metrics in evaluation_results["metrics"].items():
                print(f"  {dataset_name.title()}:")
                print(f"    MSE Loss: {metrics.mse_loss:.6f}")
                print(f"    R² Score: {metrics.r2_score:.6f}")
                print(f"    Correlation: {metrics.signal_correlation:.6f}")

        else:
            print(f"Warning: Model not found at {model_path}")
            evaluation_results = None

        total_time = time.time() - start_time

        return {
            "experiment_name": experiment_name,
            "config_path": str(config_path),
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "total_time": total_time,
            "success": True,
        }

    except Exception as e:
        import traceback

        error_time = time.time() - start_time
        print(f"\nError in experiment {experiment_name}: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return {
            "experiment_name": experiment_name,
            "config_path": str(config_path),
            "error": str(e),
            "total_time": error_time,
            "success": False,
        }


def run_comparison_study(config_dir: Path | None = None) -> dict[str, Any]:
    """Run comparison study across all configurations."""
    if config_dir is None:
        config_dir = Path(__file__).parent / "configs"

    config_files = list(config_dir.glob("*.yaml"))

    if not config_files:
        print(f"No config files found in {config_dir}")
        return {}

    print(f"Found {len(config_files)} configurations:")
    for config_file in config_files:
        print(f"  - {config_file.name}")

    all_results = {}

    for config_file in config_files:
        experiment_name = f"comparison_{config_file.stem}"
        result = run_single_experiment(config_file, experiment_name)
        all_results[config_file.stem] = result

    # Generate comparison report
    _generate_comparison_report(all_results)

    return all_results


def _generate_comparison_report(results: dict[str, dict[str, Any]]):
    """Generate comparison report across experiments."""
    output_dir = Path("outputs/geometric_signals/comparison_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "comparison_report.txt"

    with open(report_path, "w") as f:
        f.write("=== GEOMETRIC SIGNAL LEARNING COMPARISON STUDY ===\n\n")
        f.write(f"Total Experiments: {len(results)}\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary table
        f.write("EXPERIMENT SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Config':<20} {'Status':<10} {'Time (s)':<10} {'Best Val Loss':<15} {'R² Score':<10}\n"
        )
        f.write("-" * 80 + "\n")

        successful_results = []

        for config_name, result in results.items():
            status = "Success" if result["success"] else "Failed"
            total_time = result["total_time"]

            if result["success"] and result["training_results"]:
                best_val_loss = result["training_results"]["best_val_loss"]

                # Get R² score from mixed dataset if available
                r2_score = "N/A"
                if (
                    result["evaluation_results"]
                    and "mixed" in result["evaluation_results"]["metrics"]
                ):
                    r2_score = f"{result['evaluation_results']['metrics']['mixed'].r2_score:.4f}"

                successful_results.append((config_name, best_val_loss, r2_score))

            else:
                best_val_loss = "N/A"
                r2_score = "N/A"

            f.write(
                f"{config_name:<20} {status:<10} {total_time:<10.1f} {best_val_loss!s:<15} {r2_score!s:<10}\n"
            )

        f.write("-" * 80 + "\n\n")

        # Detailed results for successful experiments
        for config_name, result in results.items():
            if not result["success"]:
                continue

            f.write(f"=== {config_name.upper()} ===\n")

            if result["training_results"]:
                tr = result["training_results"]
                f.write(f"Training Time: {tr['total_time']:.2f}s\n")
                f.write(f"Final Train Loss: {tr['final_train_loss']:.6f}\n")
                f.write(f"Final Val Loss: {tr['final_val_loss']:.6f}\n")
                f.write(f"Best Val Loss: {tr['best_val_loss']:.6f}\n")

            if result["evaluation_results"]:
                f.write("\nEvaluation Results:\n")
                for dataset_name, metrics in result["evaluation_results"][
                    "metrics"
                ].items():
                    f.write(f"  {dataset_name.title()}:\n")
                    f.write(f"    MSE Loss: {metrics.mse_loss:.6f}\n")
                    f.write(f"    MAE Loss: {metrics.mae_loss:.6f}\n")
                    f.write(f"    R² Score: {metrics.r2_score:.6f}\n")
                    f.write(
                        f"    Signal Correlation: {metrics.signal_correlation:.6f}\n"
                    )

            f.write("\n")

        # Performance ranking
        if successful_results:
            f.write("PERFORMANCE RANKING (by best validation loss):\n")
            f.write("-" * 50 + "\n")

            # Sort by validation loss (lower is better)
            successful_results.sort(
                key=lambda x: float(x[1])
                if isinstance(x[1], int | float)
                else float("inf")
            )

            for i, (config_name, val_loss, r2_score) in enumerate(
                successful_results, 1
            ):
                f.write(
                    f"{i}. {config_name:<20} Val Loss: {val_loss:.6f}  R²: {r2_score}\n"
                )

    print(f"\nComparison report saved to: {report_path}")


def create_demo_dataset():
    """Create a small demo dataset for quick testing."""
    print("Creating demo signal dataset...")

    from .datasets import create_signal_dataset

    # Create small demo dataset
    dataset = create_signal_dataset(
        dataset_type="mixed", sequence_length=64, prediction_length=16, num_samples=1000
    )

    # Save a few examples
    output_dir = Path("outputs/geometric_signals/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    input_seqs, target_seqs = batch

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]

        input_seq = input_seqs[i].squeeze()
        target_seq = target_seqs[i].squeeze()

        input_time = range(len(input_seq))
        target_time = range(len(input_seq), len(input_seq) + len(target_seq))

        ax.plot(input_time, input_seq, "b-", label="Input", alpha=0.8)
        ax.plot(target_time, target_seq, "g-", label="Target", alpha=0.8)
        ax.axvline(len(input_seq), color="gray", linestyle=":", alpha=0.5)

        ax.set_title(f"Demo Signal {i + 1}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Demo Signal Dataset Examples", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "demo_signals.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Demo dataset examples saved to: {output_dir}/demo_signals.png")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Geometric Signal Learning Experiments"
    )
    parser.add_argument(
        "command",
        choices=["train", "evaluate", "compare", "demo"],
        help="Command to run",
    )
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument(
        "--model", type=Path, help="Model checkpoint path (for evaluation)"
    )
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--output", type=Path, help="Output directory")

    args = parser.parse_args()

    if args.command == "demo":
        create_demo_dataset()

    elif args.command == "train":
        if not args.config:
            print("Error: --config is required for training")
            return 1

        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}")
            return 1

        experiment_name = args.name or args.config.stem
        result = run_single_experiment(args.config, experiment_name)

        if result["success"]:
            print("\nTraining completed successfully!")
            return 0
        else:
            print("\nTraining failed!")
            return 1

    elif args.command == "evaluate":
        if not args.model or not args.config:
            print("Error: --model and --config are required for evaluation")
            return 1

        if not args.model.exists():
            print(f"Error: Model file not found: {args.model}")
            return 1

        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}")
            return 1

        output_dir = args.output or (Path("outputs/evaluation") / args.model.stem)
        results = evaluate_model(args.model, args.config, output_dir)

        print("\nEvaluation completed successfully!")
        return 0

    elif args.command == "compare":
        config_dir = args.config or (Path(__file__).parent / "configs")
        results = run_comparison_study(config_dir)

        successful_count = sum(1 for r in results.values() if r["success"])
        total_count = len(results)

        print(
            f"\nComparison study completed: {successful_count}/{total_count} experiments successful"
        )
        return 0 if successful_count > 0 else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
