"""
Real-time Training Visualization Module

Provides flexible, real-time matplotlib visualization for training pipelines.
Supports customizable plot layouts, metric tracking, and interactive displays.

## Quick Start

Basic loss visualization:
```python
from m.training_viz import create_loss_visualizer

visualizer = create_loss_visualizer(update_interval=50)
visualizer.update_metrics({
    "train_loss": 0.245,
    "val_loss": 0.267,
    "learning_rate": 0.001
})
visualizer.close()
```

MoE training visualization:
```python
from m.training_viz import create_moe_visualizer

visualizer = create_moe_visualizer()
visualizer.update_metrics({
    "train_loss": 0.245,
    "expert_entropy": 2.1,
    "load_balance": 0.85,
    "position_error": 0.134
})
visualizer.close()
```

Custom visualization:
```python
from m.training_viz import create_custom_visualizer

visualizer = create_custom_visualizer(
    metrics_config={
        "accuracy": (0, "blue"),
        "f1_score": (0, "red"),
        "learning_rate": (1, "green")
    },
    plot_titles=["Classification", "Optimization"],
    subplot_layout=(1, 2)
)
```

See docs/TRAINING_VISUALIZATION.md for comprehensive documentation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True)
class PlotConfig:
    """Configuration for a single plot panel."""

    title: str
    ylabel: str
    xlabel: str = "Step"
    yscale: str = "linear"  # "linear", "log", "symlog"
    show_grid: bool = True
    show_legend: bool = True
    color_cycle: list[str] = field(
        default_factory=lambda: [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
    )


@dataclass(slots=True)
class MetricConfig:
    """Configuration for a metric to be plotted."""

    metric_name: str
    plot_index: int  # Which subplot (0-based)
    color: str | None = None
    label: str | None = None
    line_style: str = "-"
    alpha: float = 1.0
    y_axis: str = "left"  # "left" or "right" for dual Y-axis support


@dataclass(slots=True)
class TrainingVisualizerConfig:
    """Complete configuration for training visualization."""

    # Layout configuration
    subplot_rows: int = 2
    subplot_cols: int = 2
    figure_size: tuple[int, int] = (12, 8)
    title: str = "Real-time Training Progress"

    # Update behavior
    update_interval: int = 100  # Steps between visual updates
    max_points: int | None = None  # Limit points shown (None = unlimited)

    # Plot configurations
    plots: list[PlotConfig] = field(default_factory=list)
    metrics: list[MetricConfig] = field(default_factory=list)

    # Performance
    use_blit: bool = False  # Advanced matplotlib optimization
    pause_duration: float = 0.01  # Seconds to pause between updates


class RealTimeTrainingVisualizer:
    """
    Generic real-time training visualization system.

    Provides flexible, customizable matplotlib visualization for any training pipeline.
    Supports multiple metrics, custom plot layouts, and interactive displays.

    Args:
        config: TrainingVisualizerConfig specifying layout, metrics, and behavior

    Example:
        ```python
        from m.training_viz import create_moe_visualizer

        visualizer = create_moe_visualizer(update_interval=50)

        # During training loop
        for step in range(1000):
            # ... training code ...
            visualizer.update_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "expert_entropy": entropy
            })

        visualizer.close()
        ```

    Methods:
        update_metrics: Add new metric values and update plots
        force_update: Force immediate plot update
        save_plot: Save current visualization to file
        get_metrics_summary: Get statistical summary of all metrics
        close: Clean up resources and show summary
    """

    def __init__(self, config: TrainingVisualizerConfig):
        self.config = config
        self.metrics_history: dict[str, list[float]] = {}
        self.step_count = 0
        self.start_time = time.time()

        # Matplotlib setup
        plt.ion()  # Interactive mode
        self.fig: plt.Figure | None = None
        self.axes: np.ndarray | None = None
        self.secondary_axes: dict[int, Any] = {}  # Track secondary axes by plot index
        self.lines: dict[str, Any] = {}  # Store line objects for each metric

        # Validate configuration
        self._validate_config()

    def update_metrics(self, metrics: dict[str, float]) -> None:
        """Update metrics and refresh visualization if needed."""
        self.step_count += 1

        # Store metrics
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(float(value))

            # Limit history size if configured
            if (
                self.config.max_points
                and len(self.metrics_history[key]) > self.config.max_points
            ):
                self.metrics_history[key] = self.metrics_history[key][
                    -self.config.max_points :
                ]

        # Update visualization periodically
        if self.step_count % self.config.update_interval == 0:
            self._update_plots()

    def force_update(self) -> None:
        """Force an immediate visualization update."""
        self._update_plots()

    def save_plot(self, save_path: str | Path) -> None:
        """Save current visualization to file."""
        if self.fig:
            self.fig.savefig(save_path, dpi=300, bbox_inches="tight")

    def get_metrics_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all tracked metrics."""
        summary = {}
        for metric_name, values in self.metrics_history.items():
            if values:
                summary[metric_name] = {
                    "current": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "count": len(values),
                }
        return summary

    def close(self) -> None:
        """Clean up visualization resources."""
        if self.fig:
            plt.close(self.fig)
        plt.ioff()

        # Print summary
        total_time = time.time() - self.start_time
        print(
            f"📊 Training visualization closed after {total_time:.1f}s, {self.step_count} steps"
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        total_plots = self.config.subplot_rows * self.config.subplot_cols

        if len(self.config.plots) > total_plots:
            raise ValueError(
                f"Too many plots configured: {len(self.config.plots)} > {total_plots}"
            )

        for metric in self.config.metrics:
            if metric.plot_index >= total_plots:
                raise ValueError(
                    f"Metric {metric.metric_name} references invalid plot_index {metric.plot_index}"
                )

    def _update_plots(self) -> None:
        """Update real-time training plots."""
        if self.fig is None:
            self._setup_plots()

        # Ensure initialization completed
        assert self.fig is not None and self.axes is not None, (
            "Plot initialization failed"
        )

        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Update each configured metric
        for metric_config in self.config.metrics:
            if metric_config.metric_name in self.metrics_history:
                self._plot_metric(metric_config)

        # Apply plot configurations
        for i, plot_config in enumerate(self.config.plots):
            if i < len(self.axes.flat):
                self._configure_subplot(self.axes.flat[i], plot_config, i)

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Small pause for GUI updates
        plt.pause(self.config.pause_duration)

    def _plot_metric(self, metric_config: MetricConfig) -> None:
        """Plot a single metric on its configured subplot."""
        assert self.axes is not None, "Axes not initialized"

        metric_name = metric_config.metric_name
        values = self.metrics_history[metric_name]

        if not values:
            return

        # Get base axis
        ax = self.axes.flat[metric_config.plot_index]

        # Just use the primary axis for all metrics
        plot_ax = ax

        # Generate x-axis (steps)
        steps = range(len(values))

        # Determine color and label
        color = metric_config.color
        if color is None:
            plot_config = (
                self.config.plots[metric_config.plot_index]
                if metric_config.plot_index < len(self.config.plots)
                else PlotConfig("", "")
            )
            color_idx = len(
                [
                    m
                    for m in self.config.metrics
                    if m.plot_index == metric_config.plot_index
                    and m.metric_name <= metric_name
                ]
            ) % len(plot_config.color_cycle)
            color = plot_config.color_cycle[color_idx]

        label = metric_config.label or metric_name.replace("_", " ").title()

        # Plot the metric
        plot_ax.plot(
            steps,
            values,
            color=color,
            linestyle=metric_config.line_style,
            alpha=metric_config.alpha,
            label=label,
            marker="o" if len(values) == 1 else None,  # Show markers for single points
            markersize=4,
        )

    def _configure_subplot(
        self, ax: plt.Axes, plot_config: PlotConfig, plot_index: int
    ) -> None:
        """Apply configuration to a subplot."""
        ax.set_title(plot_config.title)
        ax.set_ylabel(plot_config.ylabel)
        ax.set_xlabel(plot_config.xlabel)
        ax.set_yscale(plot_config.yscale)

        if plot_config.show_grid:
            ax.grid(True, alpha=0.3)

        if plot_config.show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                ax.legend(loc="upper left")

    def _setup_plots(self) -> None:
        """Set up the initial plotting interface."""
        fig, axes = plt.subplots(
            self.config.subplot_rows,
            self.config.subplot_cols,
            figsize=self.config.figure_size,
        )

        # Handle single subplot case - ensure axes is always an array
        if self.config.subplot_rows * self.config.subplot_cols == 1:
            axes = np.array([axes])
        elif self.config.subplot_rows == 1 or self.config.subplot_cols == 1:
            axes = axes.reshape(-1)

        # Now we can safely assign
        self.fig = fig
        self.axes = axes

        self.fig.suptitle(self.config.title, fontsize=14, fontweight="bold")
        plt.tight_layout()


# Pre-configured visualizer factories for common use cases
def create_loss_visualizer(update_interval: int = 100) -> RealTimeTrainingVisualizer:
    """
    Create a simple loss-focused visualizer with 1x2 layout.

    Creates two plots:
    - Plot 1: Loss curves (train_loss, val_loss) with log scale
    - Plot 2: Learning rate with log scale

    Args:
        update_interval: Steps between visual updates (default: 100)

    Returns:
        RealTimeTrainingVisualizer configured for basic loss monitoring

    Example:
        ```python
        visualizer = create_loss_visualizer(update_interval=50)
        visualizer.update_metrics({
            "train_loss": 0.245,
            "val_loss": 0.267,
            "learning_rate": 0.001
        })
        visualizer.close()
        ```
    """
    config = TrainingVisualizerConfig(
        subplot_rows=1,
        subplot_cols=2,
        update_interval=update_interval,
        title="Training Loss Monitoring",
        plots=[
            PlotConfig(title="Loss Curves", ylabel="Loss", yscale="log"),
            PlotConfig(title="Learning Rate", ylabel="LR", yscale="log"),
        ],
        metrics=[
            MetricConfig(
                metric_name="train_loss", plot_index=0, color="blue", label="Training"
            ),
            MetricConfig(
                metric_name="val_loss", plot_index=0, color="red", label="Validation"
            ),
            MetricConfig(
                metric_name="learning_rate", plot_index=1, color="green", label="LR"
            ),
        ],
    )
    return RealTimeTrainingVisualizer(config)


def create_moe_visualizer(
    update_matplotlib: int = 1,
    update_rich_tables: int = 25,
    n_experts: int | None = None,
) -> RealTimeTrainingVisualizer:
    """
    Create MoE-specific training visualizer with 2x2 expert metrics layout.

    Creates four plots:
    - Plot 1: Loss curves (train_loss, val_loss, aux_loss) with log scale
    - Plot 2: Task-specific metrics (position_error, velocity_error, accuracy, etc.)
    - Plot 3: Expert utilization (expert_entropy, load_balance, routing_loss)
    - Plot 4: Training throughput (samples_per_sec, tokens_per_sec)

    Args:
        update_matplotlib: Steps between matplotlib chart updates (default: 1 = every step)
        update_rich_tables: Steps between rich CLI table updates (default: 25)

    Returns:
        RealTimeTrainingVisualizer configured for MoE training monitoring

    Example:
        ```python
        visualizer = create_moe_visualizer(update_matplotlib=1, update_rich_tables=25)
        visualizer.update_metrics({
            "train_loss": 0.245,
            "val_loss": 0.267,
            "aux_loss": 0.012,
            "position_error": 0.134,
            "expert_entropy": 2.1,
            "load_balance": 0.85,
            "samples_per_sec": 120.5
        })
        visualizer.close()
        ```
    """
    config = TrainingVisualizerConfig(
        subplot_rows=2,
        subplot_cols=2,
        update_interval=update_matplotlib,
        title="MoE Training Progress",
        plots=[
            PlotConfig(title="Loss Curves", ylabel="Loss", yscale="log"),
            PlotConfig(title="Task-Specific Metrics", ylabel="Error"),
            PlotConfig(title="Expert Utilization", ylabel="Utilization"),
            PlotConfig(title="Training Throughput", ylabel="Samples/sec"),
        ],
        metrics=[
            # Loss metrics
            MetricConfig(
                metric_name="train_loss", plot_index=0, color="blue", label="Train Loss"
            ),
            MetricConfig(
                metric_name="val_loss", plot_index=0, color="red", label="Val Loss"
            ),
            MetricConfig(
                metric_name="aux_loss", plot_index=0, color="orange", label="Aux Loss"
            ),
            # Task-specific metrics (flexible names)
            MetricConfig(
                metric_name="position_error",
                plot_index=1,
                color="green",
                label="Position Error",
            ),
            MetricConfig(
                metric_name="velocity_error",
                plot_index=1,
                color="cyan",
                label="Velocity Error",
            ),
            MetricConfig(
                metric_name="accuracy", plot_index=1, color="purple", label="Accuracy"
            ),
            # Throughput
            MetricConfig(
                metric_name="samples_per_sec",
                plot_index=3,
                color="blue",
                label="Samples/sec",
            ),
        ],
    )

    # Dynamically add individual expert utilization metrics if n_experts is provided
    if n_experts is not None:
        expert_colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        for expert_id in range(n_experts):
            color = expert_colors[
                expert_id % len(expert_colors)
            ]  # Cycle through colors if more experts than colors
            config.metrics.append(
                MetricConfig(
                    metric_name=f"expert_{expert_id}_utilization",
                    plot_index=2,
                    color=color,
                    label=f"Expert {expert_id}",
                )
            )

    return RealTimeTrainingVisualizer(config)


def create_custom_visualizer(
    metrics_config: dict[str, tuple[int, str]],  # metric_name -> (plot_index, color)
    plot_titles: list[str],
    subplot_layout: tuple[int, int] = (2, 2),
    update_interval: int = 100,
    title: str = "Custom Training Visualization",
) -> RealTimeTrainingVisualizer:
    """Create a custom visualizer with specified metrics and layout."""

    # Create plot configs
    plots = []
    for i, plot_title in enumerate(plot_titles):
        plots.append(PlotConfig(title=plot_title, ylabel="Value"))

    # Create metric configs
    metrics = []
    for metric_name, (plot_index, color) in metrics_config.items():
        metrics.append(
            MetricConfig(metric_name=metric_name, plot_index=plot_index, color=color)
        )

    config = TrainingVisualizerConfig(
        subplot_rows=subplot_layout[0],
        subplot_cols=subplot_layout[1],
        update_interval=update_interval,
        title=title,
        plots=plots,
        metrics=metrics,
    )

    return RealTimeTrainingVisualizer(config)
