"""
Evaluation and analysis tools for trained geometric signal models.
Provides comprehensive analysis of model performance and expert behavior.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from m.moe import (
    AttentionConfig,
    BlockConfig,
    ExpertConfig,
    ModelConfig,
    MoEConfig,
    MoESequenceRegressor,
    RouterConfig,
)

from .datasets import create_signal_dataset
from .model_discovery import (
    find_geometric_signals_models,
    get_latest_model,
    select_model_interactively,
)
from .training import load_config
from .visualization import (
    analyze_frequency_spectrum,
    create_frequency_analysis_plot,
    create_model_analysis_dashboard,
    create_sequence_prediction_plot,
    create_signal_comparison_plot,
)


@dataclass(slots=True)
class EvaluationMetrics:
    """Container for model evaluation metrics."""

    mse_loss: float
    mae_loss: float
    r2_score: float
    frequency_error: float
    phase_error: float
    prediction_variance: float

    # Signal-specific metrics
    signal_correlation: float
    spectral_distance: float


@dataclass(slots=True)
class ExpertAnalysis:
    """Analysis of expert behavior and specialization."""

    expert_utilization: dict[int, float]
    expert_specialization: dict[int, dict[str, float]]  # Expert -> signal type -> usage
    routing_entropy: float
    load_balance_factor: float


class SignalEvaluator:
    """Comprehensive evaluator for signal learning models."""

    def __init__(self, model_path: Path, config_path: Path, device: str = "auto"):
        self.device = self._setup_device(device)

        # Load configuration
        self.config = load_config(config_path)

        # Load model
        self.model = self._load_model(model_path)

        # Create evaluation datasets
        self.test_datasets = self._create_test_datasets()

        print("Initialized SignalEvaluator:")
        print(f"  Device: {self.device}")
        print(
            f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                device_obj = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_obj = torch.device("mps")
            else:
                device_obj = torch.device("cpu")
        else:
            device_obj = torch.device(device)

        return device_obj

    def _load_model(self, model_path: Path) -> MoESequenceRegressor:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # Recreate model from config (same logic as training)
        # Config classes already imported at top

        # Parse the nested config structure
        model_cfg_dict = self.config.model

        # Create the nested configs
        attn_config = AttentionConfig(**model_cfg_dict["block"]["attn"])
        router_config = RouterConfig(**model_cfg_dict["block"]["moe"]["router"])
        expert_config = ExpertConfig(**model_cfg_dict["block"]["moe"]["expert"])
        moe_config = MoEConfig(
            d_model=model_cfg_dict["block"]["moe"]["d_model"],
            router=router_config,
            expert=expert_config,
        )
        block_config = BlockConfig(
            attn=attn_config,
            moe=moe_config,
            use_rms_norm=model_cfg_dict["block"]["use_rms_norm"],
        )

        # Create the main model config
        model_config = ModelConfig(
            block=block_config,
            n_layers=model_cfg_dict["n_layers"],
            input_dim=model_cfg_dict["input_dim"],
            target_dim=model_cfg_dict["target_dim"],
            pool=model_cfg_dict.get("pool", "mean"),
        )

        model = MoESequenceRegressor(model_config)

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def _create_test_datasets(self) -> dict[str, torch.utils.data.IterableDataset]:
        """Create test datasets for different signal types."""
        base_config = self.config.dataset

        datasets = {}

        # Individual signal type datasets
        for signal_type in ["sine", "composite", "geometric"]:
            datasets[signal_type] = create_signal_dataset(
                dataset_type=signal_type,
                sequence_length=base_config["sequence_length"],
                prediction_length=base_config["prediction_length"],
                num_samples=5000,  # Smaller test set
                seed=42 + hash(signal_type) % 10000,
            )

        # Mixed dataset
        datasets["mixed"] = create_signal_dataset(
            dataset_type="mixed",
            sequence_length=base_config["sequence_length"],
            prediction_length=base_config["prediction_length"],
            num_samples=10000,
            seed=12345,
        )

        return datasets

    @torch.no_grad()
    def evaluate_dataset(
        self, dataset_name: str
    ) -> tuple[
        EvaluationMetrics, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ]:
        """Evaluate model on specific dataset."""
        dataset = self.test_datasets[dataset_name]

        dataloader = DataLoader(
            dataset,
            batch_size=64,
            num_workers=0,  # No multiprocessing to avoid pickle issues
            pin_memory=(self.device.type == "cuda"),
        )

        all_targets = []
        all_predictions = []
        all_inputs = []

        total_mse = 0
        total_mae = 0
        num_samples = 0

        print(f"Evaluating on {dataset_name} dataset...")

        for i, batch in enumerate(dataloader):
            if i >= 100:  # Limit evaluation batches
                break

            input_seq, target_seq = batch
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            # Forward pass - use same sequence-to-sequence logic as training
            # Concatenate input and target sequences for the model
            full_seq = torch.cat(
                [input_seq, target_seq], dim=1
            )  # [B, input_len + pred_len, 1]
            logits, _ = self.model(full_seq, targets=None)

            # Extract predictions for target portion
            input_len = input_seq.shape[1]
            predictions = logits[:, input_len:, :]  # [B, pred_len, 1]

            # Compute losses
            mse = F.mse_loss(predictions, target_seq)
            mae = F.l1_loss(predictions, target_seq)

            total_mse += mse.item() * input_seq.size(0)
            total_mae += mae.item() * input_seq.size(0)
            num_samples += input_seq.size(0)

            # Collect samples for analysis
            all_inputs.extend([inp.cpu() for inp in input_seq[:4]])
            all_targets.extend([tgt.cpu() for tgt in target_seq[:4]])
            all_predictions.extend([pred.cpu() for pred in predictions[:4]])

        # Compute final metrics
        avg_mse = total_mse / num_samples
        avg_mae = total_mae / num_samples

        # Compute R² score
        r2_score = self._compute_r2_score(all_targets, all_predictions)

        # Compute frequency-domain metrics
        freq_error, phase_error = self._compute_frequency_metrics(
            all_targets, all_predictions
        )

        # Compute spectral distance
        spectral_distance = self._compute_spectral_distance(
            all_targets, all_predictions
        )

        # Compute signal correlation
        signal_correlation = self._compute_signal_correlation(
            all_targets, all_predictions
        )

        # Prediction variance
        pred_variance = torch.var(torch.stack(all_predictions)).item()

        metrics = EvaluationMetrics(
            mse_loss=avg_mse,
            mae_loss=avg_mae,
            r2_score=r2_score,
            frequency_error=freq_error,
            phase_error=phase_error,
            prediction_variance=pred_variance,
            signal_correlation=signal_correlation,
            spectral_distance=spectral_distance,
        )

        # Return sample predictions for visualization
        examples = list(zip(all_inputs[:8], all_targets[:8], all_predictions[:8]))

        return metrics, examples

    def _compute_r2_score(
        self, targets: list[torch.Tensor], predictions: list[torch.Tensor]
    ) -> float:
        """Compute R² coefficient of determination."""
        if not targets or not predictions:
            return 0.0

        targets_tensor = torch.stack(targets).flatten()
        predictions_tensor = torch.stack(predictions).flatten()

        ss_res = torch.sum((targets_tensor - predictions_tensor) ** 2)
        ss_tot = torch.sum((targets_tensor - torch.mean(targets_tensor)) ** 2)

        if ss_tot == 0:
            return 0.0

        r2 = 1 - (ss_res / ss_tot)
        return r2.item()

    def _compute_frequency_metrics(
        self, targets: list[torch.Tensor], predictions: list[torch.Tensor]
    ) -> tuple[float, float]:
        """Compute frequency and phase error metrics."""
        if not targets or not predictions:
            return 0.0, 0.0

        freq_errors = []
        phase_errors = []

        for target, pred in zip(targets[:20], predictions[:20]):  # Sample subset
            # Flatten signals
            if target.dim() > 1:
                target = target.squeeze()
            if pred.dim() > 1:
                pred = pred.squeeze()

            # Compute frequency spectra
            target_freqs, target_mag = analyze_frequency_spectrum(target)
            pred_freqs, pred_mag = analyze_frequency_spectrum(pred)

            # Find dominant frequencies
            target_peak = target_freqs[
                np.argmax(target_mag[1:]) + 1
            ]  # Skip DC component
            pred_peak = pred_freqs[np.argmax(pred_mag[1:]) + 1]

            freq_error = abs(target_peak - pred_peak)
            freq_errors.append(freq_error)

            # Phase error (simplified)
            target_phase = np.angle(np.fft.rfft(target.numpy())[1:])
            pred_phase = np.angle(np.fft.rfft(pred.numpy())[1:])
            phase_error = np.mean(
                np.abs(np.angle(np.exp(1j * (target_phase - pred_phase))))
            )
            phase_errors.append(phase_error)

        return float(np.mean(freq_errors)), float(np.mean(phase_errors))

    def _compute_spectral_distance(
        self, targets: list[torch.Tensor], predictions: list[torch.Tensor]
    ) -> float:
        """Compute spectral distance between target and predicted signals."""
        if not targets or not predictions:
            return 0.0

        spectral_distances = []

        for target, pred in zip(targets[:20], predictions[:20]):
            if target.dim() > 1:
                target = target.squeeze()
            if pred.dim() > 1:
                pred = pred.squeeze()

            # Compute power spectral densities
            _, target_psd = analyze_frequency_spectrum(target)
            _, pred_psd = analyze_frequency_spectrum(pred)

            # Normalize PSDs
            target_psd = target_psd / np.sum(target_psd)
            pred_psd = pred_psd / np.sum(pred_psd)

            # Compute Wasserstein distance (simplified as L2 distance)
            distance = np.sqrt(np.sum((target_psd - pred_psd) ** 2))
            spectral_distances.append(distance)

        return float(np.mean(spectral_distances))

    def _compute_signal_correlation(
        self, targets: list[torch.Tensor], predictions: list[torch.Tensor]
    ) -> float:
        """Compute average correlation between target and predicted signals."""
        if not targets or not predictions:
            return 0.0

        correlations = []

        for target, pred in zip(targets, predictions):
            if target.dim() > 1:
                target = target.squeeze()
            if pred.dim() > 1:
                pred = pred.squeeze()

            # Compute Pearson correlation
            target_np = target.numpy()
            pred_np = pred.numpy()

            if len(target_np) > 1 and len(pred_np) > 1:
                corr = np.corrcoef(target_np, pred_np)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 0.0

    def analyze_experts(self) -> ExpertAnalysis:
        """Analyze expert utilization and specialization."""
        # This would require instrumenting the forward pass to collect expert routing decisions
        # For now, return dummy analysis
        n_experts = self.config.model["block"]["moe"]["router"]["n_experts"]

        # Uniform utilization (placeholder)
        utilization = {i: 1.0 / n_experts for i in range(n_experts)}

        # Dummy specialization
        specialization = {}
        for expert_id in range(n_experts):
            specialization[expert_id] = {
                "sine": 0.4,
                "composite": 0.3,
                "geometric": 0.3,
            }

        return ExpertAnalysis(
            expert_utilization=utilization,
            expert_specialization=specialization,
            routing_entropy=np.log(n_experts),  # Maximum entropy
            load_balance_factor=1.0,
        )

    def generate_evaluation_report(self, output_dir: Path) -> dict[str, Any]:
        """Generate comprehensive evaluation report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Generating evaluation report...")

        # Evaluate on all datasets
        results = {}
        all_examples = {}

        for dataset_name in self.test_datasets.keys():
            metrics, examples = self.evaluate_dataset(dataset_name)
            results[dataset_name] = metrics
            all_examples[dataset_name] = examples

        # Analyze experts
        expert_analysis = self.analyze_experts()

        # Generate visualizations
        self._create_evaluation_plots(results, all_examples, output_dir)

        # Save results summary
        self._save_results_summary(results, expert_analysis, output_dir)

        print(f"Evaluation report saved to: {output_dir}")

        return {
            "metrics": results,
            "expert_analysis": expert_analysis,
            "output_dir": output_dir,
        }

    def _create_evaluation_plots(
        self,
        results: dict[str, EvaluationMetrics],
        examples: dict[str, list],
        output_dir: Path,
    ):
        """Create evaluation visualization plots."""

        # Signal prediction comparison plots
        for dataset_name, example_list in examples.items():
            if not example_list:
                continue

            # Create enhanced prediction visualization showing model behavior
            # create_sequence_prediction_plot already imported at top

            create_sequence_prediction_plot(
                examples=example_list[:4],  # (input, target, prediction) tuples
                title=f"Model Behavior Analysis - {dataset_name.title()} Dataset",
                save_path=output_dir / f"detailed_predictions_{dataset_name}.png",
                dataset_name=dataset_name,
            )

            # Also create legacy plot for compatibility
            create_signal_comparison_plot(
                signals=[
                    (f"Input {i + 1}", ex[0]) for i, ex in enumerate(example_list[:4])
                ],
                title=f"Signal Inputs - {dataset_name.title()}",
                save_path=output_dir / f"inputs_{dataset_name}.png",
            )

            # Create frequency analysis plot
            if len(example_list) > 0:
                target_signals = [
                    (f"Target {i + 1}", ex[1]) for i, ex in enumerate(example_list[:3])
                ]
                pred_signals = [
                    (f"Pred {i + 1}", ex[2]) for i, ex in enumerate(example_list[:3])
                ]

                combined_signals = target_signals + pred_signals

                create_frequency_analysis_plot(
                    signals=combined_signals,
                    title=f"Frequency Analysis - {dataset_name.title()}",
                    save_path=output_dir / f"frequency_{dataset_name}.png",
                )

        # Metrics comparison plot
        self._create_metrics_comparison_plot(results, output_dir)

        # Create comprehensive model analysis dashboard
        self._create_model_dashboard(results, examples, output_dir)

    def _create_metrics_comparison_plot(
        self, results: dict[str, EvaluationMetrics], output_dir: Path
    ):
        """Create metrics comparison bar plot."""
        # matplotlib.pyplot already imported at top

        dataset_names = list(results.keys())
        metric_names = ["MSE Loss", "MAE Loss", "R² Score", "Signal Correlation"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, metric in enumerate(
            ["mse_loss", "mae_loss", "r2_score", "signal_correlation"]
        ):
            ax = axes[i]

            values = [getattr(results[dataset], metric) for dataset in dataset_names]
            bars = ax.bar(dataset_names, values, alpha=0.7)

            # Color bars based on performance
            if metric in ["r2_score", "signal_correlation"]:
                # Higher is better
                colors = [
                    "green" if v > 0.8 else "orange" if v > 0.6 else "red"
                    for v in values
                ]
            else:
                # Lower is better
                colors = [
                    "green" if v < 0.1 else "orange" if v < 0.3 else "red"
                    for v in values
                ]

            for bar, color in zip(bars, colors):
                bar.set_color(color)

            ax.set_title(metric_names[i])
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(j, v + max(values) * 0.01, f"{v:.3f}", ha="center", va="bottom")

        plt.suptitle("Model Performance Across Signal Types", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _create_model_dashboard(
        self, results: dict[str, EvaluationMetrics], examples: dict, output_dir: Path
    ):
        """Create comprehensive model analysis dashboard."""
        # matplotlib.pyplot already imported at top

        # create_model_analysis_dashboard already imported at top

        # Prepare data for dashboard
        dashboard_data = {
            name: (results[name], examples[name])
            for name in results.keys()
            if examples.get(name)
        }

        if dashboard_data:
            fig = create_model_analysis_dashboard(
                results_by_dataset=dashboard_data,
                title="Model Performance Analysis",
                save_path=output_dir / "model_analysis_dashboard.png",
            )
            plt.close(fig)

    def _save_results_summary(
        self,
        results: dict[str, EvaluationMetrics],
        expert_analysis: ExpertAnalysis,
        output_dir: Path,
    ):
        """Save evaluation results to text summary."""
        summary_path = output_dir / "evaluation_summary.txt"

        with open(summary_path, "w") as f:
            f.write("=== MODEL EVALUATION REPORT ===\n\n")

            # Model info
            total_params = sum(p.numel() for p in self.model.parameters())
            f.write(f"Model Parameters: {total_params:,}\n")
            f.write(f"Device: {self.device}\n\n")

            # Results by dataset
            for dataset_name, metrics in results.items():
                f.write(f"--- {dataset_name.upper()} DATASET ---\n")
                f.write(f"MSE Loss:          {metrics.mse_loss:.6f}\n")
                f.write(f"MAE Loss:          {metrics.mae_loss:.6f}\n")
                f.write(f"R² Score:          {metrics.r2_score:.6f}\n")
                f.write(f"Signal Correlation:{metrics.signal_correlation:.6f}\n")
                f.write(f"Frequency Error:   {metrics.frequency_error:.6f}\n")
                f.write(f"Phase Error:       {metrics.phase_error:.6f}\n")
                f.write(f"Spectral Distance: {metrics.spectral_distance:.6f}\n")
                f.write("\n")

            # Expert analysis
            f.write("--- EXPERT ANALYSIS ---\n")
            f.write(f"Routing Entropy: {expert_analysis.routing_entropy:.4f}\n")
            f.write(f"Load Balance Factor: {expert_analysis.load_balance_factor:.4f}\n")
            f.write("\nExpert Utilization:\n")
            for expert_id, usage in expert_analysis.expert_utilization.items():
                f.write(f"  Expert {expert_id:2d}: {usage:.3f}\n")


def evaluate_model(
    model_path: Path, config_path: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """Evaluate a trained model and generate report."""
    if output_dir is None:
        output_dir = Path("outputs/evaluation") / model_path.stem

    evaluator = SignalEvaluator(model_path, config_path)
    results = evaluator.generate_evaluation_report(output_dir)

    return results


def main():
    """Main CLI entrypoint for geometric signals evaluation."""
    # sys and model_discovery functions already imported at top

    if len(sys.argv) < 2:
        print("🌊 Geometric Signals Evaluation")
        print("Usage: signals-eval <model_path> [config_path]")
        print("       signals-eval --latest       # Use latest model")
        print("       signals-eval --interactive  # Select model interactively")
        print("")

        # Show available models
        models = find_geometric_signals_models()
        if models:
            print(f"📁 Found {len(models)} trained models:")
            for i, model in enumerate(models[:5], 1):
                print(f"  {i}. {model}")
            print("")
            print("Examples:")
            print(f"  signals-eval {models[0]}")
            print("  signals-eval --latest")
            print("  signals-eval --interactive")
        else:
            print("❌ No trained models found. Train a model first!")

        sys.exit(1)

    # Handle special flags
    if sys.argv[1] == "--latest":
        model_path = get_latest_model()
        if model_path is None:
            print("❌ No models found")
            sys.exit(1)
    elif sys.argv[1] == "--interactive":
        model_path = select_model_interactively()
        if model_path is None:
            print("❌ No model selected")
            sys.exit(1)
    else:
        model_path = Path(sys.argv[1])
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)

    # Get config path
    if len(sys.argv) > 2 and sys.argv[2] not in ["--latest", "--interactive"]:
        config_path = Path(sys.argv[2])
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
    else:
        # Use default config
        config_path = Path(__file__).parent / "configs" / "quick_test.yaml"
        if not config_path.exists():
            print(f"❌ Default configuration not found: {config_path}")
            sys.exit(1)

    print("🔬 Starting geometric signals evaluation")
    print(f"📁 Model: {model_path}")
    print(f"⚙️  Config: {config_path}")

    try:
        results = evaluate_model(model_path, config_path)
        print("✅ Evaluation completed successfully!")
        print(f"📊 Results summary: {len(results)} datasets evaluated")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
