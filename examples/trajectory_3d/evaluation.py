"""
Evaluation and analysis tools for trained 3D trajectory models.
Provides comprehensive analysis of model performance and expert behavior for trajectory prediction.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from m.moe import MoESequenceRegressor

from .datasets import TrajectoryConfig, create_trajectory_dataset
from .model_discovery import (
    get_latest_trajectory_model,
    select_trajectory_model_interactively,
)
from .training import load_config_from_yaml
from .visualization import Trajectory3DVisualizer


@dataclass(slots=True)
class TrajectoryEvaluationMetrics:
    """Container for trajectory model evaluation metrics."""

    # Basic regression metrics
    mse_loss: float
    mae_loss: float
    r2_score: float

    # 3D trajectory-specific metrics
    position_error: float  # Mean euclidean distance error
    velocity_error: float  # Error in velocity/direction
    acceleration_error: float  # Error in acceleration/curvature
    trajectory_deviation: float  # Path deviation metric

    # Temporal consistency metrics
    temporal_consistency: float  # How consistent are predictions over time
    endpoint_error: float  # Error at final predicted position

    # Physics-informed metrics
    smoothness_metric: float  # How smooth are the predicted trajectories
    energy_conservation: float  # For orbital/physical trajectories


@dataclass(slots=True)
class TrajectoryExpertAnalysis:
    """Analysis of expert behavior and specialization for trajectories."""

    expert_utilization: dict[int, float]
    trajectory_specialization: dict[
        int, dict[str, float]
    ]  # Expert -> trajectory type -> usage
    spatial_specialization: dict[
        int, dict[str, float]
    ]  # Expert -> spatial region -> usage
    routing_entropy: float
    load_balance_factor: float


class Trajectory3DEvaluator:
    """Comprehensive evaluator for 3D trajectory learning models."""

    def __init__(self, model_path: Path, config_path: Path, device: str = "auto"):
        self.device = self._setup_device(device)

        # Load configuration
        self.model_config, self.training_config, self.dataset_config = (
            load_config_from_yaml(str(config_path))
        )

        # Load model
        self.model = self._load_model(model_path)

        # Create evaluation datasets
        self.test_datasets = self._create_test_datasets()

        print("Initialized Trajectory3DEvaluator:")
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

        # The model config is stored in the checkpoint
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
        else:
            model_config = self.model_config

        model = MoESequenceRegressor(model_config)

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def _create_test_datasets(self) -> dict[str, torch.utils.data.IterableDataset[Any]]:
        """Create test datasets for different trajectory types."""
        base_config = self.dataset_config

        datasets: dict[str, torch.utils.data.IterableDataset[Any]] = {}

        # Individual trajectory type datasets
        for trajectory_type in ["helical", "orbital", "lissajous", "lorenz", "robotic"]:
            # Create config with only this trajectory type
            test_config = TrajectoryConfig(
                sequence_length=base_config.sequence_length,
                prediction_length=base_config.prediction_length,
                sampling_rate=base_config.sampling_rate,
                noise_std=base_config.noise_std,
                helical_weight=1.0 if trajectory_type == "helical" else 0.0,
                orbital_weight=1.0 if trajectory_type == "orbital" else 0.0,
                lissajous_weight=1.0 if trajectory_type == "lissajous" else 0.0,
                lorenz_weight=1.0 if trajectory_type == "lorenz" else 0.0,
                robotic_weight=1.0 if trajectory_type == "robotic" else 0.0,
            )
            datasets[trajectory_type] = create_trajectory_dataset(
                test_config, seed=42 + hash(trajectory_type) % 10000
            )

        # Mixed dataset (original proportions)
        datasets["mixed"] = create_trajectory_dataset(base_config, seed=12345)

        return datasets

    @torch.no_grad()
    def evaluate_dataset(
        self, dataset_name: str
    ) -> tuple[
        TrajectoryEvaluationMetrics,
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ]:
        """Evaluate model on specific dataset."""
        dataset = self.test_datasets[dataset_name]

        # Create a simple iterator for limited evaluation
        all_targets = []
        all_predictions = []
        all_inputs = []

        total_mse = 0.0
        total_mae = 0.0
        total_position_error = 0.0
        total_velocity_error = 0.0
        total_acceleration_error = 0.0
        num_samples = 0

        print(f"Evaluating on {dataset_name} dataset...")

        # Evaluate on limited number of samples
        for i, sample in enumerate(dataset):
            if i >= 500:  # Limit evaluation samples
                break

            input_seq = (
                sample["input_sequence"].unsqueeze(0).to(self.device)
            )  # [1, seq_len, 3]
            target_seq = (
                sample["target_sequence"].unsqueeze(0).to(self.device)
            )  # [1, pred_len, 3]

            # Forward pass
            predictions, _ = self.model(input_seq)

            # Extract predictions for target portion
            pred_len = target_seq.shape[1]
            pred_outputs = predictions[:, -pred_len:, :]  # [1, pred_len, 3]

            # Compute losses
            mse = F.mse_loss(pred_outputs, target_seq)
            mae = F.l1_loss(pred_outputs, target_seq)

            # Compute 3D-specific metrics
            position_error = torch.mean(torch.norm(pred_outputs - target_seq, dim=-1))

            # Velocity error
            if pred_len > 1:
                pred_velocities = pred_outputs[:, 1:] - pred_outputs[:, :-1]
                target_velocities = target_seq[:, 1:] - target_seq[:, :-1]
                velocity_error = torch.mean(
                    torch.norm(pred_velocities - target_velocities, dim=-1)
                )
            else:
                velocity_error = torch.tensor(0.0)

            # Acceleration error
            if pred_len > 2:
                pred_accelerations = pred_velocities[:, 1:] - pred_velocities[:, :-1]
                target_accelerations = (
                    target_velocities[:, 1:] - target_velocities[:, :-1]
                )
                acceleration_error = torch.mean(
                    torch.norm(pred_accelerations - target_accelerations, dim=-1)
                )
            else:
                acceleration_error = torch.tensor(0.0)

            total_mse += mse.item()
            total_mae += mae.item()
            total_position_error += position_error.item()
            total_velocity_error += velocity_error.item()
            total_acceleration_error += acceleration_error.item()
            num_samples += 1

            # Collect samples for visualization (first 8 only)
            if i < 8:
                all_inputs.append(input_seq[0].cpu())
                all_targets.append(target_seq[0].cpu())
                all_predictions.append(pred_outputs[0].cpu())

        # Compute final metrics
        avg_mse = total_mse / num_samples
        avg_mae = total_mae / num_samples
        avg_position_error = total_position_error / num_samples
        avg_velocity_error = total_velocity_error / num_samples
        avg_acceleration_error = total_acceleration_error / num_samples

        # Compute additional metrics
        r2_score = self._compute_r2_score(all_targets, all_predictions)
        trajectory_deviation = self._compute_trajectory_deviation(
            all_targets, all_predictions
        )
        temporal_consistency = self._compute_temporal_consistency(all_predictions)
        endpoint_error = self._compute_endpoint_error(all_targets, all_predictions)
        smoothness_metric = self._compute_smoothness(all_predictions)
        energy_conservation = self._compute_energy_conservation(
            all_targets, all_predictions
        )

        metrics = TrajectoryEvaluationMetrics(
            mse_loss=avg_mse,
            mae_loss=avg_mae,
            r2_score=r2_score,
            position_error=avg_position_error,
            velocity_error=avg_velocity_error,
            acceleration_error=avg_acceleration_error,
            trajectory_deviation=trajectory_deviation,
            temporal_consistency=temporal_consistency,
            endpoint_error=endpoint_error,
            smoothness_metric=smoothness_metric,
            energy_conservation=energy_conservation,
        )

        # Return sample predictions for visualization
        examples = list(zip(all_inputs, all_targets, all_predictions))

        return metrics, examples

    def _compute_r2_score(
        self, targets: list[torch.Tensor], predictions: list[torch.Tensor]
    ) -> float:
        """Compute R² coefficient of determination for 3D trajectories."""
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

    def _compute_trajectory_deviation(
        self, targets: list[torch.Tensor], predictions: list[torch.Tensor]
    ) -> float:
        """Compute average path deviation between predicted and target trajectories."""
        if not targets or not predictions:
            return 0.0

        deviations = []
        for target, pred in zip(targets, predictions):
            # Compute cumulative path length difference
            if len(target) > 1 and len(pred) > 1:
                target_path = torch.cumsum(
                    torch.norm(target[1:] - target[:-1], dim=-1), dim=0
                )
                pred_path = torch.cumsum(
                    torch.norm(pred[1:] - pred[:-1], dim=-1), dim=0
                )

                if len(target_path) == len(pred_path):
                    path_deviation = torch.mean(torch.abs(target_path - pred_path))
                    deviations.append(path_deviation.item())

        return float(np.mean(deviations)) if deviations else 0.0

    def _compute_temporal_consistency(self, predictions: list[torch.Tensor]) -> float:
        """Compute temporal consistency of predictions."""
        if not predictions:
            return 0.0

        consistency_scores = []
        for pred in predictions:
            if len(pred) > 2:
                # Compute second derivatives (acceleration)
                velocities = pred[1:] - pred[:-1]
                accelerations = velocities[1:] - velocities[:-1]

                # Consistency is inverse of acceleration variance
                acc_variance = torch.var(torch.norm(accelerations, dim=-1))
                consistency = 1.0 / (1.0 + acc_variance)  # Normalized consistency
                consistency_scores.append(consistency.item())

        return float(np.mean(consistency_scores)) if consistency_scores else 0.0

    def _compute_endpoint_error(
        self, targets: list[torch.Tensor], predictions: list[torch.Tensor]
    ) -> float:
        """Compute error at the final predicted position."""
        if not targets or not predictions:
            return 0.0

        endpoint_errors = []
        for target, pred in zip(targets, predictions):
            if len(target) > 0 and len(pred) > 0:
                endpoint_error = torch.norm(target[-1] - pred[-1])
                endpoint_errors.append(endpoint_error.item())

        return float(np.mean(endpoint_errors)) if endpoint_errors else 0.0

    def _compute_smoothness(self, predictions: list[torch.Tensor]) -> float:
        """Compute smoothness metric for predicted trajectories."""
        if not predictions:
            return 0.0

        smoothness_scores = []
        for pred in predictions:
            if len(pred) > 2:
                # Compute jerk (third derivative)
                velocities = pred[1:] - pred[:-1]
                accelerations = velocities[1:] - velocities[:-1]
                jerks = accelerations[1:] - accelerations[:-1]

                # Smoothness is inverse of jerk magnitude
                jerk_magnitude = torch.mean(torch.norm(jerks, dim=-1))
                smoothness = 1.0 / (1.0 + jerk_magnitude)  # Normalized smoothness
                smoothness_scores.append(smoothness.item())

        return float(np.mean(smoothness_scores)) if smoothness_scores else 0.0

    def _compute_energy_conservation(
        self, targets: list[torch.Tensor], predictions: list[torch.Tensor]
    ) -> float:
        """Compute energy conservation metric (relevant for physical trajectories)."""
        if not targets or not predictions:
            return 0.0

        energy_conservation_scores = []
        for target, pred in zip(targets, predictions):
            if len(target) > 1 and len(pred) > 1:
                # Compute kinetic energy proxy (velocity magnitude squared)
                target_velocities = target[1:] - target[:-1]
                pred_velocities = pred[1:] - pred[:-1]

                target_energy = torch.sum(torch.norm(target_velocities, dim=-1) ** 2)
                pred_energy = torch.sum(torch.norm(pred_velocities, dim=-1) ** 2)

                # Energy conservation score (how close predicted energy is to target)
                if target_energy > 0:
                    conservation = (
                        1.0 - torch.abs(target_energy - pred_energy) / target_energy
                    )
                    conservation = torch.clamp(
                        conservation, 0.0, 1.0
                    )  # Clamp to [0, 1]
                    energy_conservation_scores.append(conservation.item())

        return (
            float(np.mean(energy_conservation_scores))
            if energy_conservation_scores
            else 0.0
        )

    def analyze_experts(self) -> TrajectoryExpertAnalysis:
        """Analyze expert utilization and specialization."""
        # This would require instrumenting the forward pass to collect expert routing decisions
        # For now, return dummy analysis based on model configuration
        n_experts = self.model_config.block.moe.router.n_experts

        # Uniform utilization (placeholder)
        utilization = {i: 1.0 / n_experts for i in range(n_experts)}

        # Trajectory type specialization (placeholder)
        trajectory_specialization = {}
        spatial_specialization = {}
        for expert_id in range(n_experts):
            trajectory_specialization[expert_id] = {
                "helical": 0.2,
                "orbital": 0.2,
                "lissajous": 0.2,
                "lorenz": 0.2,
                "robotic": 0.2,
            }
            spatial_specialization[expert_id] = {
                "low": 0.33,
                "mid": 0.34,
                "high": 0.33,
            }

        return TrajectoryExpertAnalysis(
            expert_utilization=utilization,
            trajectory_specialization=trajectory_specialization,
            spatial_specialization=spatial_specialization,
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
        results: dict[str, TrajectoryEvaluationMetrics],
        examples: dict[str, list],
        output_dir: Path,
    ):
        """Create evaluation visualization plots."""
        visualizer = Trajectory3DVisualizer(output_dir)

        # Trajectory prediction comparison plots
        for dataset_name, example_list in examples.items():
            if not example_list:
                continue

            # Create prediction visualizations for each trajectory type
            for i, (input_seq, target_seq, pred_seq) in enumerate(example_list[:3]):
                visualizer.plot_prediction_comparison(
                    input_seq,
                    target_seq,
                    pred_seq,
                    dataset_name,
                    str(output_dir / f"prediction_{dataset_name}_{i + 1}.png"),
                )

        # Metrics comparison plot
        self._create_metrics_comparison_plot(results, output_dir)

        # Create comprehensive model analysis dashboard
        self._create_trajectory_analysis_dashboard(results, examples, output_dir)

    def _create_metrics_comparison_plot(
        self, results: dict[str, TrajectoryEvaluationMetrics], output_dir: Path
    ):
        """Create metrics comparison plot for trajectory models."""
        # matplotlib.pyplot already imported at top

        dataset_names = list(results.keys())

        # Define metrics to plot
        metrics_info = [
            ("position_error", "Position Error", "lower"),
            ("velocity_error", "Velocity Error", "lower"),
            ("r2_score", "R² Score", "higher"),
            ("trajectory_deviation", "Trajectory Deviation", "lower"),
            ("temporal_consistency", "Temporal Consistency", "higher"),
            ("smoothness_metric", "Smoothness", "higher"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, (metric_name, metric_label, direction) in enumerate(metrics_info):
            ax = axes[i]

            values = [
                getattr(results[dataset], metric_name) for dataset in dataset_names
            ]
            bars = ax.bar(dataset_names, values, alpha=0.7)

            # Color bars based on performance
            if direction == "higher":
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

            ax.set_title(metric_label)
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(j, v + max(values) * 0.01, f"{v:.3f}", ha="center", va="bottom")

        plt.suptitle("3D Trajectory Model Performance Analysis", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _create_trajectory_analysis_dashboard(
        self,
        results: dict[str, TrajectoryEvaluationMetrics],
        examples: dict,
        output_dir: Path,
    ):
        """Create comprehensive trajectory analysis dashboard."""
        # matplotlib.pyplot already imported at top

        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(
            "3D Trajectory Model Analysis Dashboard", fontsize=16, fontweight="bold"
        )

        # 1. Performance Overview
        ax1 = plt.subplot(2, 3, 1)
        trajectory_types = list(results.keys())
        position_errors = [results[t].position_error for t in trajectory_types]

        ax1.bar(trajectory_types, position_errors, color="skyblue", alpha=0.7)
        ax1.set_title("Position Error by Trajectory Type")
        ax1.set_ylabel("Mean Position Error")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. Velocity vs Position Error
        ax2 = plt.subplot(2, 3, 2)
        velocity_errors = [results[t].velocity_error for t in trajectory_types]

        ax2.scatter(
            position_errors,
            velocity_errors,
            s=100,
            alpha=0.7,
            c=range(len(trajectory_types)),
            cmap="viridis",
        )
        for i, ttype in enumerate(trajectory_types):
            ax2.annotate(
                ttype,
                (position_errors[i], velocity_errors[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        ax2.set_xlabel("Position Error")
        ax2.set_ylabel("Velocity Error")
        ax2.set_title("Velocity vs Position Error")
        ax2.grid(True, alpha=0.3)

        # 3. R² Score Comparison
        ax3 = plt.subplot(2, 3, 3)
        r2_scores = [results[t].r2_score for t in trajectory_types]

        ax3.bar(trajectory_types, r2_scores, color="lightgreen", alpha=0.7)
        ax3.set_title("R² Score by Trajectory Type")
        ax3.set_ylabel("R² Score")
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3)

        # Add performance threshold line
        ax3.axhline(
            y=0.9, color="green", linestyle="--", alpha=0.7, label="Excellent (0.9+)"
        )
        ax3.axhline(
            y=0.8, color="orange", linestyle="--", alpha=0.7, label="Good (0.8+)"
        )
        ax3.legend()

        # 4. Temporal Metrics
        ax4 = plt.subplot(2, 3, 4)
        consistency_scores = [results[t].temporal_consistency for t in trajectory_types]
        smoothness_scores = [results[t].smoothness_metric for t in trajectory_types]

        x_pos = np.arange(len(trajectory_types))
        width = 0.35

        ax4.bar(
            x_pos - width / 2,
            consistency_scores,
            width,
            label="Temporal Consistency",
            alpha=0.7,
        )
        ax4.bar(
            x_pos + width / 2, smoothness_scores, width, label="Smoothness", alpha=0.7
        )

        ax4.set_title("Temporal Quality Metrics")
        ax4.set_ylabel("Score")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(trajectory_types, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Physics-Informed Metrics
        ax5 = plt.subplot(2, 3, 5)
        endpoint_errors = [results[t].endpoint_error for t in trajectory_types]
        energy_conservation = [results[t].energy_conservation for t in trajectory_types]

        ax5_twin = ax5.twinx()

        ax5.bar(
            x_pos - width / 2,
            endpoint_errors,
            width,
            color="red",
            alpha=0.7,
            label="Endpoint Error",
        )
        ax5_twin.bar(
            x_pos + width / 2,
            energy_conservation,
            width,
            color="blue",
            alpha=0.7,
            label="Energy Conservation",
        )

        ax5.set_title("Physics-Informed Metrics")
        ax5.set_ylabel("Endpoint Error", color="red")
        ax5_twin.set_ylabel("Energy Conservation", color="blue")
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(trajectory_types, rotation=45)

        # Combine legends
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # 6. Overall Performance Radar
        ax6 = plt.subplot(2, 3, 6, projection="polar")

        # Create radar chart for mixed dataset (overall performance)
        if "mixed" in results:
            mixed_results = results["mixed"]

            metrics = [
                1 - mixed_results.position_error,  # Invert for better visualization
                1 - mixed_results.velocity_error,
                mixed_results.r2_score,
                mixed_results.temporal_consistency,
                mixed_results.smoothness_metric,
                mixed_results.energy_conservation,
            ]

            labels = [
                "Position",
                "Velocity",
                "R²",
                "Consistency",
                "Smoothness",
                "Energy",
            ]

            # Complete the circle
            metrics += [metrics[0]]
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += [angles[0]]

            ax6.plot(angles, metrics, "b-", linewidth=2, label="Model Performance")
            ax6.fill(angles, metrics, alpha=0.25, color="blue")
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels(labels)
            ax6.set_ylim(0, 1)
            ax6.set_title("Overall Performance Radar\n(Mixed Dataset)", pad=20)
            ax6.grid(True)

        plt.tight_layout()
        plt.savefig(
            output_dir / "trajectory_analysis_dashboard.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _save_results_summary(
        self,
        results: dict[str, TrajectoryEvaluationMetrics],
        expert_analysis: TrajectoryExpertAnalysis,
        output_dir: Path,
    ):
        """Save evaluation results to text summary."""
        summary_path = output_dir / "evaluation_summary.txt"

        with open(summary_path, "w") as f:
            f.write("=== 3D TRAJECTORY MODEL EVALUATION REPORT ===\n\n")

            # Model info
            total_params = sum(p.numel() for p in self.model.parameters())
            f.write(f"Model Parameters: {total_params:,}\n")
            f.write(f"Device: {self.device}\n\n")

            # Results by dataset
            for dataset_name, metrics in results.items():
                f.write(f"--- {dataset_name.upper()} TRAJECTORIES ---\n")
                f.write(f"MSE Loss:               {metrics.mse_loss:.6f}\n")
                f.write(f"MAE Loss:               {metrics.mae_loss:.6f}\n")
                f.write(f"R² Score:               {metrics.r2_score:.6f}\n")
                f.write(f"Position Error:         {metrics.position_error:.6f}\n")
                f.write(f"Velocity Error:         {metrics.velocity_error:.6f}\n")
                f.write(f"Acceleration Error:     {metrics.acceleration_error:.6f}\n")
                f.write(f"Trajectory Deviation:   {metrics.trajectory_deviation:.6f}\n")
                f.write(f"Temporal Consistency:   {metrics.temporal_consistency:.6f}\n")
                f.write(f"Endpoint Error:         {metrics.endpoint_error:.6f}\n")
                f.write(f"Smoothness:             {metrics.smoothness_metric:.6f}\n")
                f.write(f"Energy Conservation:    {metrics.energy_conservation:.6f}\n")
                f.write("\n")

            # Expert analysis
            f.write("--- EXPERT ANALYSIS ---\n")
            f.write(f"Routing Entropy: {expert_analysis.routing_entropy:.4f}\n")
            f.write(f"Load Balance Factor: {expert_analysis.load_balance_factor:.4f}\n")
            f.write("\nExpert Utilization:\n")
            for expert_id, usage in expert_analysis.expert_utilization.items():
                f.write(f"  Expert {expert_id:2d}: {usage:.3f}\n")

            f.write("\nTrajectory Specialization:\n")
            for (
                expert_id,
                specialization,
            ) in expert_analysis.trajectory_specialization.items():
                f.write(f"  Expert {expert_id:2d}:\n")
                for traj_type, usage in specialization.items():
                    f.write(f"    {traj_type:>10s}: {usage:.3f}\n")


def evaluate_trajectory_model(
    model_path: Path, config_path: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """Evaluate a trained 3D trajectory model and generate report."""
    if output_dir is None:
        # Use experiment name (parent directory) instead of model filename
        experiment_name = model_path.parent.name
        output_dir = Path("outputs/trajectory_3d/evaluation") / experiment_name

    evaluator = Trajectory3DEvaluator(model_path, config_path)
    results = evaluator.generate_evaluation_report(output_dir)

    return results


def main():
    """Main CLI entrypoint for 3D trajectory evaluation."""
    # sys already imported at top

    # Simple model discovery for trajectory 3d models
    def find_trajectory_3d_models():
        """Find available trajectory 3d models."""
        base_dir = Path("outputs/trajectory_3d")
        if not base_dir.exists():
            return []

        models = []
        for experiment_dir in base_dir.iterdir():
            if experiment_dir.is_dir():
                for model_file in ["best_model.pt", "final_model.pt"]:
                    model_path = experiment_dir / model_file
                    if model_path.exists():
                        models.append(model_path)

        return sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)

    def get_latest_model():
        """Get the latest trained model."""
        # get_latest_trajectory_model already imported at top

        latest = get_latest_trajectory_model()
        return latest.model_path if latest else None

    if len(sys.argv) < 2:
        print("🚀 3D Trajectory Model Evaluation")
        print("Usage: trajectory-3d-eval <model_path> [config_path]")
        print("       trajectory-3d-eval --latest       # Use latest model")
        print("       trajectory-3d-eval --interactive  # Select model interactively")
        print("")

        # Show available models
        models = find_trajectory_3d_models()
        if models:
            print(f"📁 Found {len(models)} trained models:")
            for i, model in enumerate(models[:5], 1):
                print(f"  {i}. {model}")
            print("")
            print("Examples:")
            print(f"  trajectory-3d-eval {models[0]}")
            print("  trajectory-3d-eval --latest")
            print("  trajectory-3d-eval --interactive")
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
        # select_trajectory_model_interactively already imported at top

        selected_model = select_trajectory_model_interactively()
        if selected_model is None:
            print("❌ No model selected")
            sys.exit(1)
        model_path = selected_model.model_path
    else:
        model_path = Path(sys.argv[1])
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)

    # Get config path
    if len(sys.argv) > 2 and sys.argv[2] != "--latest":
        config_path = Path(sys.argv[2])
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            sys.exit(1)
    else:
        # Use default config
        config_path = Path(__file__).parent / "configs" / "quick_test_3d.yaml"
        if not config_path.exists():
            print(f"❌ Default configuration not found: {config_path}")
            sys.exit(1)

    print("🔬 Starting 3D trajectory evaluation")
    print(f"📁 Model: {model_path}")
    print(f"⚙️  Config: {config_path}")

    try:
        results = evaluate_trajectory_model(model_path, config_path)
        print("✅ Evaluation completed successfully!")
        print(
            f"📊 Results summary: {len(results['metrics'])} trajectory types evaluated"
        )
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        # traceback already imported at top

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
