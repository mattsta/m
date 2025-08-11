"""
Visualization tools for training progress and signal analysis.
Provides real-time monitoring and post-training analysis capabilities.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from .evaluation import EvaluationMetrics

# Rich imports for CLI display
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table


@dataclass(slots=True)
class TrainingMetrics:
    """Container for comprehensive training metrics."""

    step: int
    epoch: int
    train_loss: float
    val_loss: float | None = None
    learning_rate: float | None = None
    aux_loss: float | None = None
    expert_utilization: dict[int, float] = field(default_factory=dict)

    # Throughput metrics
    samples_per_second: float | None = None
    batches_per_second: float | None = None
    steps_per_second: float | None = None
    tokens_per_second: float | None = None  # sequence_length * samples_per_second

    # System metrics
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    gpu_utilization_percent: float | None = None

    # Training efficiency metrics
    loss_per_second: float | None = None  # how fast loss is decreasing
    gradient_norm: float | None = None


@dataclass(slots=True)
class VisualizationConfig:
    """Configuration for visualization settings."""

    update_interval: int = 100  # Steps between plot updates
    save_interval: int = 1000  # Steps between saving plots
    plot_window: int = 1000  # Number of recent steps to show
    signal_examples: int = 4  # Number of example signals to plot
    output_dir: Path = Path("outputs/geometric_signals")
    show_plots: bool = True  # Whether to display plots during training
    save_animations: bool = False  # Whether to save training animations


class LiveTrainingVisualizer:
    """Real-time visualization of training progress."""

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.metrics_history: list[TrainingMetrics] = []
        self.example_predictions: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

        # Rich console for CLI output
        self.console = Console()
        self.progress: Progress | None = None
        self.task_id: TaskID | None = None

        # Matplotlib setup for live plotting
        if self.config.show_plots:
            plt.ion()  # Turn on interactive mode
            self.fig = None
            self.axes = None
            self._setup_plots()

    def _setup_plots(self):
        """Initialize the plotting layout."""
        self.fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)

        # Training loss plot
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        self.ax_loss.set_title("Training Progress")
        self.ax_loss.set_xlabel("Step")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True)

        # Expert utilization plot
        self.ax_experts = self.fig.add_subplot(gs[0, 2:])
        self.ax_experts.set_title("Expert Utilization")
        self.ax_experts.set_xlabel("Expert ID")
        self.ax_experts.set_ylabel("Usage Rate")
        self.ax_experts.grid(True)

        # Signal prediction examples
        self.ax_signals = []
        for i in range(4):
            row = 1 + i // 2
            col = (i % 2) * 2
            ax = self.fig.add_subplot(gs[row, col : col + 2])
            ax.set_title(f"Signal Prediction {i + 1}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
            self.ax_signals.append(ax)

    def start_progress(self, total_steps: int, batch_size: int = 64):
        """Initialize progress tracking."""
        self.batch_size = batch_size  # Store for throughput calculations
        if not self.config.show_plots:
            # Use rich progress bar for CLI output
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("• Training Steps •"),
                TimeRemainingColumn(),
                console=self.console,
                refresh_per_second=4,
            )
            self.progress = progress
            self.task_id = progress.add_task("Training Steps", total=total_steps)
            progress.start()

    def update(
        self,
        metrics: TrainingMetrics,
        predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        | None = None,
    ):
        """Update visualizations with new metrics."""
        self.metrics_history.append(metrics)

        if predictions:
            self.example_predictions = predictions[: self.config.signal_examples]

        # Update CLI progress display
        if not self.config.show_plots:
            self._update_cli_display(metrics)

        if self.config.show_plots and metrics.step % self.config.update_interval == 0:
            self._update_plots()

        if metrics.step % self.config.save_interval == 0:
            self._save_plots(metrics.step)

    def _update_cli_display(self, metrics: TrainingMetrics):
        """Update rich CLI progress display."""
        if self.progress and self.task_id is not None:
            # Update progress bar with comprehensive live metrics
            throughput_text = ""
            if metrics.steps_per_second:
                throughput_text = f" | {metrics.steps_per_second:.1f} steps/s | {metrics.samples_per_second:.0f} samples/s"

            memory_text = ""
            if metrics.gpu_memory_used_mb:
                memory_text = f" | GPU: {metrics.gpu_memory_used_mb:.0f}MB"

            loss_rate_text = ""
            if metrics.loss_per_second:
                loss_rate_text = f" | ΔLoss: {metrics.loss_per_second:.6f}/s"

            self.progress.update(
                self.task_id,
                advance=1,
                description=f"Step {metrics.step} | Loss: {metrics.train_loss:.4f}"
                + (f" | Val: {metrics.val_loss:.4f}" if metrics.val_loss else "")
                + (
                    f" | LR: {metrics.learning_rate:.6f}"
                    if metrics.learning_rate
                    else ""
                )
                + throughput_text
                + memory_text
                + loss_rate_text,
            )

            # Print detailed status every 25 steps
            if metrics.step % 25 == 0:
                # Main training table
                table = Table(
                    show_header=True, header_style="bold blue", title="Training Status"
                )
                table.add_column("Step")
                table.add_column("Epoch")
                table.add_column("Train Loss")
                table.add_column("Val Loss")
                table.add_column("LR")
                table.add_column("Loss/s")

                table.add_row(
                    str(metrics.step),
                    str(metrics.epoch),
                    f"{metrics.train_loss:.6f}",
                    f"{metrics.val_loss:.6f}" if metrics.val_loss else "N/A",
                    f"{metrics.learning_rate:.6f}" if metrics.learning_rate else "N/A",
                    f"{metrics.loss_per_second:.6f}"
                    if metrics.loss_per_second
                    else "N/A",
                )

                self.console.print("\n")
                self.console.print(Panel(table, expand=False))

                # Throughput metrics table
                throughput_table = Table(
                    show_header=True,
                    header_style="bold green",
                    title="Throughput Metrics",
                )
                throughput_table.add_column("Steps/s")
                throughput_table.add_column("Batches/s")
                throughput_table.add_column("Samples/s")
                throughput_table.add_column("Tokens/s")

                throughput_table.add_row(
                    f"{metrics.steps_per_second:.2f}"
                    if metrics.steps_per_second
                    else "N/A",
                    f"{metrics.batches_per_second:.2f}"
                    if metrics.batches_per_second
                    else "N/A",
                    f"{metrics.samples_per_second:.1f}"
                    if metrics.samples_per_second
                    else "N/A",
                    f"{metrics.tokens_per_second:.0f}"
                    if metrics.tokens_per_second
                    else "N/A",
                )

                self.console.print(Panel(throughput_table, expand=False))

                # System metrics table (if available)
                if metrics.gpu_memory_used_mb is not None:
                    system_table = Table(
                        show_header=True,
                        header_style="bold yellow",
                        title="System Resources",
                    )
                    system_table.add_column("GPU Memory")
                    system_table.add_column("GPU Util%")
                    system_table.add_column("Grad Norm")

                    gpu_mem_text = (
                        f"{metrics.gpu_memory_used_mb:.1f}MB"
                        if metrics.gpu_memory_used_mb
                        else "N/A"
                    )
                    if metrics.gpu_memory_total_mb:
                        gpu_mem_text += f" / {metrics.gpu_memory_total_mb:.1f}MB"

                    system_table.add_row(
                        gpu_mem_text,
                        f"{metrics.gpu_utilization_percent:.1f}%"
                        if metrics.gpu_utilization_percent
                        else "N/A",
                        f"{metrics.gradient_norm:.4f}"
                        if metrics.gradient_norm
                        else "N/A",
                    )

                    self.console.print(Panel(system_table, expand=False))

                # Show expert utilization if available
                if metrics.expert_utilization:
                    exp_table = Table(show_header=True, header_style="bold green")
                    exp_table.add_column("Expert")
                    exp_table.add_column("Utilization")

                    for expert_id, utilization in metrics.expert_utilization.items():
                        exp_table.add_row(str(expert_id), f"{utilization:.3f}")

                    self.console.print(
                        Panel(exp_table, title="Expert Utilization", expand=False)
                    )

    def _update_plots(self):
        """Update all plot contents."""
        if not self.fig:
            return

        # Update training loss
        self._update_loss_plot()

        # Update expert utilization
        self._update_expert_plot()

        # Update signal predictions
        self._update_signal_plots()

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _update_loss_plot(self):
        """Update the training loss plot."""
        if not self.metrics_history:
            return

        self.ax_loss.clear()

        steps = [m.step for m in self.metrics_history[-self.config.plot_window :]]
        train_losses = [
            m.train_loss for m in self.metrics_history[-self.config.plot_window :]
        ]
        val_losses = [
            m.val_loss
            for m in self.metrics_history[-self.config.plot_window :]
            if m.val_loss is not None
        ]
        val_steps = [
            m.step
            for m in self.metrics_history[-self.config.plot_window :]
            if m.val_loss is not None
        ]

        self.ax_loss.plot(steps, train_losses, "b-", label="Train Loss", alpha=0.7)
        if val_losses:
            self.ax_loss.plot(
                val_steps, val_losses, "r-", label="Val Loss", alpha=0.7, linewidth=2
            )

        self.ax_loss.set_title("Training Progress")
        self.ax_loss.set_xlabel("Step")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend()
        self.ax_loss.grid(True)
        self.ax_loss.set_yscale("log")

    def _update_expert_plot(self):
        """Update the expert utilization plot."""
        if not self.metrics_history or not self.metrics_history[-1].expert_utilization:
            return

        self.ax_experts.clear()

        latest_metrics = self.metrics_history[-1]
        expert_ids = sorted(latest_metrics.expert_utilization.keys())
        utilizations = [latest_metrics.expert_utilization[eid] for eid in expert_ids]

        bars = self.ax_experts.bar(expert_ids, utilizations, alpha=0.7)

        # Color bars based on utilization level
        for bar, util in zip(bars, utilizations):
            if util < 0.1:
                bar.set_color("red")
            elif util < 0.3:
                bar.set_color("orange")
            else:
                bar.set_color("green")

        self.ax_experts.set_title(f"Expert Utilization (Step {latest_metrics.step})")
        self.ax_experts.set_xlabel("Expert ID")
        self.ax_experts.set_ylabel("Usage Rate")
        self.ax_experts.set_ylim(0, max(utilizations) * 1.1 if utilizations else 1)
        self.ax_experts.grid(True, alpha=0.3)

    def _update_signal_plots(self):
        """Update the signal prediction plots."""
        if not self.example_predictions:
            return

        for i, (input_seq, target_seq, pred_seq) in enumerate(self.example_predictions):
            if i >= len(self.ax_signals):
                break

            ax = self.ax_signals[i]
            ax.clear()

            # Convert tensors to numpy
            if input_seq.dim() > 1:
                input_seq = input_seq.squeeze()
            if target_seq.dim() > 1:
                target_seq = target_seq.squeeze()
            if pred_seq.dim() > 1:
                pred_seq = pred_seq.squeeze()

            input_np = input_seq.detach().cpu().numpy()
            target_np = target_seq.detach().cpu().numpy()
            pred_np = pred_seq.detach().cpu().numpy()

            # Create time axes
            input_time = np.arange(len(input_np))
            target_time = np.arange(len(input_np), len(input_np) + len(target_np))

            # Plot signals
            ax.plot(input_time, input_np, "b-", label="Input", alpha=0.8)
            ax.plot(target_time, target_np, "g-", label="Target", alpha=0.8)
            ax.plot(target_time, pred_np, "r--", label="Prediction", alpha=0.8)

            # Add vertical line at prediction boundary
            ax.axvline(len(input_np), color="gray", linestyle=":", alpha=0.5)

            ax.set_title(f"Signal Prediction {i + 1}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _save_plots(self, step: int):
        """Save current plots to disk."""
        # Only save plots if matplotlib plotting is enabled
        if not self.config.show_plots or not hasattr(self, "fig") or not self.fig:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = (
            self.config.output_dir
            / f"training_progress_step_{step:06d}_{timestamp}.png"
        )
        self.fig.savefig(filename, dpi=150, bbox_inches="tight")

    def save_final_report(self, model_name: str, total_time: float):
        """Generate and save final training report."""
        if not self.metrics_history:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Complete loss curve - ensure all values are CPU scalars
        def to_cpu_scalar(val):
            """Convert tensor to CPU scalar if needed."""
            if hasattr(val, "cpu"):
                return val.cpu().item()
            return float(val)

        steps = [m.step for m in self.metrics_history]
        train_losses = [to_cpu_scalar(m.train_loss) for m in self.metrics_history]
        val_losses = [
            to_cpu_scalar(m.val_loss)
            for m in self.metrics_history
            if m.val_loss is not None
        ]
        val_steps = [m.step for m in self.metrics_history if m.val_loss is not None]

        ax1.plot(steps, train_losses, "b-", label="Train Loss", alpha=0.7)
        if val_losses:
            ax1.plot(
                val_steps, val_losses, "r-", label="Val Loss", alpha=0.7, linewidth=2
            )
        ax1.set_title("Complete Training History")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        ax1.set_yscale("log")

        # Learning rate schedule
        lrs = [
            to_cpu_scalar(m.learning_rate)
            for m in self.metrics_history
            if m.learning_rate is not None
        ]
        lr_steps = [m.step for m in self.metrics_history if m.learning_rate is not None]
        if lrs:
            ax2.plot(lr_steps, lrs, "g-", alpha=0.7)
            ax2.set_title("Learning Rate Schedule")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Learning Rate")
            ax2.grid(True)
            ax2.set_yscale("log")

        # Training throughput
        throughputs = [
            to_cpu_scalar(m.samples_per_second)
            for m in self.metrics_history
            if m.samples_per_second is not None
        ]
        throughput_steps = [
            m.step for m in self.metrics_history if m.samples_per_second is not None
        ]
        if throughputs:
            ax3.plot(throughput_steps, throughputs, "purple", alpha=0.7)
            ax3.set_title("Training Throughput")
            ax3.set_xlabel("Step")
            ax3.set_ylabel("Samples/sec")
            ax3.grid(True)

        # Final expert utilization
        if self.metrics_history[-1].expert_utilization:
            expert_ids = sorted(self.metrics_history[-1].expert_utilization.keys())
            utilizations = [
                self.metrics_history[-1].expert_utilization[eid] for eid in expert_ids
            ]
            bars = ax4.bar(expert_ids, utilizations, alpha=0.7)

            # Color bars
            for bar, util in zip(bars, utilizations):
                if util < 0.1:
                    bar.set_color("red")
                elif util < 0.3:
                    bar.set_color("orange")
                else:
                    bar.set_color("green")

            ax4.set_title("Final Expert Utilization")
            ax4.set_xlabel("Expert ID")
            ax4.set_ylabel("Usage Rate")
            ax4.grid(True, alpha=0.3)

        plt.suptitle(f"Training Report: {model_name} ({total_time:.1f}s)", fontsize=16)
        plt.tight_layout()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = (
            self.config.output_dir / f"training_report_{model_name}_{timestamp}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Training report saved to: {filename}")

    def close(self):
        """Clean up visualization resources."""
        # Close rich progress display
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.task_id = None

        # Close matplotlib resources
        if self.config.show_plots and hasattr(self, "fig") and self.fig:
            plt.ioff()
            plt.close(self.fig)


def create_sequence_prediction_plot(
    examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    title: str = "Sequence Prediction Examples",
    save_path: Path | None = None,
    dataset_name: str = "unknown",
) -> Figure:
    """
    Create comprehensive visualization of sequence-to-sequence prediction.

    Args:
        examples: List of (input_seq, target_seq, prediction_seq) tuples
        title: Plot title
        save_path: Where to save the plot
        dataset_name: Type of dataset for context
    """

    n_examples = min(len(examples), 4)
    # Use better aspect ratio: wider and taller for better visibility
    fig, axes = plt.subplots(n_examples, 1, figsize=(20, 6 * n_examples))
    if n_examples == 1:
        axes = [axes]

    for i, (input_seq, target_seq, pred_seq) in enumerate(examples[:n_examples]):
        ax = axes[i]

        # Convert to numpy
        if input_seq.dim() > 1:
            input_seq = input_seq.squeeze()
        if target_seq.dim() > 1:
            target_seq = target_seq.squeeze()
        if pred_seq.dim() > 1:
            pred_seq = pred_seq.squeeze()

        input_np = input_seq.detach().cpu().numpy()
        target_np = target_seq.detach().cpu().numpy()
        pred_np = pred_seq.detach().cpu().numpy()

        # Create continuous time axis for the full signal
        input_len = len(input_np)
        target_len = len(target_np)
        total_len = input_len + target_len

        # Full continuous time axis
        full_time = np.arange(total_len)
        input_time = full_time[:input_len]  # [0, 1, 2, ..., 31]

        # Reconstruct the full continuous signal for visualization
        full_target = np.concatenate([input_np, target_np])
        full_pred = np.concatenate([input_np, pred_np])

        # Plot the full ground truth signal (input + target continuation)
        ax.plot(
            full_time,
            full_target,
            "g-",
            linewidth=1.5,
            alpha=0.6,
            label="Ground Truth Full Signal",
        )

        # Plot the full predicted signal (input + prediction continuation)
        ax.plot(
            full_time,
            full_pred,
            "r--",
            linewidth=1.5,
            alpha=0.8,
            label="Model Prediction Full Signal",
        )

        # Highlight the input context portion that model actually saw
        ax.plot(
            input_time,
            input_np,
            "b-",
            linewidth=3,
            label="Input Context (Model Sees)",
            alpha=0.9,
        )

        # Add prediction boundary line exactly where input context ends
        boundary_position = input_len - 1  # Last timestep of input (31)
        ax.axvline(
            boundary_position,
            color="orange",
            linestyle=":",
            linewidth=2,
            label="Prediction Boundary",
            alpha=0.7,
        )

        # Calculate accuracy metrics for this example
        mse = np.mean((target_np - pred_np) ** 2)
        correlation = (
            np.corrcoef(target_np, pred_np)[0, 1] if len(target_np) > 1 else 1.0
        )

        # Clean individual subplot title with just essential info
        ax.set_title(
            f"Example {i + 1} ({dataset_name.title()}) • MSE: {mse:.6f} • Correlation: {correlation:.4f}",
            fontsize=10,
            pad=10,
        )

        ax.set_xlabel("Time Steps", fontsize=10)
        ax.set_ylabel("Signal Amplitude", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Visual distinction is clear from colors and boundary line - no annotations needed

    # Clean main title with proper spacing
    plt.suptitle(
        f"{title}\nSequence-to-Sequence Prediction: Predict next 8 steps given 32 input steps",
        fontsize=12,
        y=0.95,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))  # Leave space for suptitle

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Detailed prediction visualization saved to: {save_path}")

    return fig


def create_signal_comparison_plot(
    signals: list[tuple[str, torch.Tensor]],
    predictions: list[tuple[str, torch.Tensor]] | None = None,
    title: str = "Signal Comparison",
    save_path: Path | None = None,
) -> Figure:
    """Create a comparison plot of multiple signals (legacy function)."""

    fig, axes = plt.subplots(len(signals), 1, figsize=(12, 3 * len(signals)))
    if len(signals) == 1:
        axes = [axes]

    for i, (label, signal) in enumerate(signals):
        ax = axes[i]

        if signal.dim() > 1:
            signal = signal.squeeze()
        signal_np = signal.detach().cpu().numpy()

        time_axis = np.arange(len(signal_np))
        ax.plot(time_axis, signal_np, "b-", label=label, alpha=0.8)

        if predictions and i < len(predictions):
            pred_label, pred_signal = predictions[i]
            if pred_signal.dim() > 1:
                pred_signal = pred_signal.squeeze()
            pred_np = pred_signal.detach().cpu().numpy()

            # Assume prediction starts where signal ends
            pred_time = np.arange(len(signal_np), len(signal_np) + len(pred_np))
            ax.plot(pred_time, pred_np, "r--", label=pred_label, alpha=0.8)
            ax.axvline(len(signal_np), color="gray", linestyle=":", alpha=0.5)

        ax.set_title(f"{label}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_model_analysis_dashboard(
    results_by_dataset: dict[str, tuple[EvaluationMetrics, list]],
    title: str = "Model Performance Dashboard",
    save_path: Path | None = None,
) -> Figure:
    """
    Create comprehensive dashboard showing model performance across datasets.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # Performance comparison across datasets
    ax1 = fig.add_subplot(gs[0, :2])
    datasets = list(results_by_dataset.keys())
    mse_scores = [results_by_dataset[d][0].mse_loss for d in datasets]
    r2_scores = [results_by_dataset[d][0].r2_score for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2, mse_scores, width, label="MSE Loss", alpha=0.8, color="red"
    )
    bars2 = ax1.bar(
        x + width / 2, r2_scores, width, label="R² Score", alpha=0.8, color="green"
    )

    ax1.set_xlabel("Dataset Type")
    ax1.set_title("Model Performance by Dataset")
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.title() for d in datasets])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Correlation analysis
    ax2 = fig.add_subplot(gs[0, 2:])
    correlations = [results_by_dataset[d][0].signal_correlation for d in datasets]
    colors = ["blue", "green", "orange", "purple"][: len(datasets)]

    bars = ax2.bar(datasets, correlations, color=colors, alpha=0.7)
    ax2.set_title("Signal Correlation by Dataset")
    ax2.set_ylabel("Correlation")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    for bar, corr in zip(bars, correlations):
        ax2.annotate(
            f"{corr:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Sample predictions for each dataset (mini plots)
    for i, (dataset_name, (metrics, examples)) in enumerate(results_by_dataset.items()):
        if not examples:
            continue

        ax = fig.add_subplot(gs[1 + i // 2, i % 2])

        # Take first example
        input_seq, target_seq, pred_seq = examples[0]

        if input_seq.dim() > 1:
            input_seq = input_seq.squeeze()
        if target_seq.dim() > 1:
            target_seq = target_seq.squeeze()
        if pred_seq.dim() > 1:
            pred_seq = pred_seq.squeeze()

        input_np = input_seq.detach().cpu().numpy()
        target_np = target_seq.detach().cpu().numpy()
        pred_np = pred_seq.detach().cpu().numpy()

        # Use same continuous visualization logic as detailed plots
        input_len = len(input_np)
        target_len = len(target_np)
        total_len = input_len + target_len

        # Create continuous time axis and full signals
        full_time = np.arange(total_len)
        input_time = full_time[:input_len]
        full_target = np.concatenate([input_np, target_np])
        full_pred = np.concatenate([input_np, pred_np])

        # Plot continuous signals
        ax.plot(full_time, full_target, "g-", linewidth=1, alpha=0.6, label="Target")
        ax.plot(full_time, full_pred, "r--", linewidth=1, alpha=0.8, label="Prediction")
        ax.plot(input_time, input_np, "b-", linewidth=2, label="Input", alpha=0.9)
        ax.axvline(input_len - 1, color="orange", linestyle=":", alpha=0.7)

        mse = np.mean((target_np - pred_np) ** 2)
        ax.set_title(f"{dataset_name.title()} Example\nMSE: {mse:.6f}", fontsize=10)
        ax.set_xlabel("Time Steps", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"{title}\nComprehensive Model Analysis Across Signal Types",
        fontsize=16,
        y=0.95,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Model analysis dashboard saved to: {save_path}")

    return fig


def analyze_frequency_spectrum(
    signal: torch.Tensor, sampling_rate: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and return frequency spectrum of signal."""
    if signal.dim() > 1:
        signal = signal.squeeze()

    signal_np = signal.detach().cpu().numpy()

    # Compute FFT
    fft = np.fft.rfft(signal_np)
    freqs = np.fft.rfftfreq(len(signal_np), 1 / sampling_rate)
    magnitude = np.abs(fft)

    return freqs, magnitude


def create_frequency_analysis_plot(
    signals: list[tuple[str, torch.Tensor]],
    sampling_rate: float = 100.0,
    title: str = "Frequency Analysis",
    save_path: Path | None = None,
) -> Figure:
    """Create frequency domain analysis plot."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Time domain plot
    for label, signal in signals:
        if signal.dim() > 1:
            signal = signal.squeeze()
        signal_np = signal.detach().cpu().numpy()
        time_axis = np.arange(len(signal_np)) / sampling_rate
        ax1.plot(time_axis, signal_np, label=label, alpha=0.7)

    ax1.set_title("Time Domain")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)

    # Frequency domain plot
    for label, signal in signals:
        freqs, magnitude = analyze_frequency_spectrum(signal, sampling_rate)
        ax2.plot(freqs, magnitude, label=label, alpha=0.7)

    ax2.set_title("Frequency Domain")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale("log")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_metrics_csv(metrics_history: list[TrainingMetrics], save_path: Path):
    """Save training metrics to CSV for further analysis."""
    # csv already imported at top

    with open(save_path, "w", newline="") as csvfile:
        fieldnames = [
            "step",
            "epoch",
            "train_loss",
            "val_loss",
            "learning_rate",
            "aux_loss",
            "samples_per_second",
            "steps_per_second",
            "tokens_per_second",
            "loss_per_second",
            "gpu_memory_used_mb",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for metrics in metrics_history:
            writer.writerow(
                {
                    "step": metrics.step,
                    "epoch": metrics.epoch,
                    "train_loss": metrics.train_loss,
                    "val_loss": metrics.val_loss,
                    "learning_rate": metrics.learning_rate,
                    "aux_loss": metrics.aux_loss,
                    "samples_per_second": metrics.samples_per_second,
                    "steps_per_second": metrics.steps_per_second,
                    "tokens_per_second": metrics.tokens_per_second,
                    "loss_per_second": metrics.loss_per_second,
                    "gpu_memory_used_mb": metrics.gpu_memory_used_mb,
                }
            )

    print(f"Metrics saved to: {save_path}")
