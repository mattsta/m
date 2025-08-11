"""
Rich trainer visualization component for real-time training monitoring.
Provides reusable, well-encapsulated visualization with tables and animated status bars.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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


@dataclass(slots=True, frozen=True)
class ThroughputMetrics:
    """Throughput metrics for training performance."""

    steps_per_second: float | None = None
    batches_per_second: float | None = None
    samples_per_second: float | None = None
    tokens_per_second: float | None = None


@dataclass(slots=True, frozen=True)
class SystemMetrics:
    """System resource utilization metrics."""

    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    gpu_utilization_percent: float | None = None
    gradient_norm: float | None = None


@dataclass(slots=True, frozen=True)
class TrainingSnapshot:
    """Basic training snapshot with minimal required data."""

    step: int
    train_loss: float
    val_loss: float | None = None
    learning_rate: float | None = None
    aux_loss: float | None = None
    expert_utilization: dict[int, float] = field(default_factory=dict)
    batch_size: int | None = None
    sequence_length: int | None = None
    gpu_memory_mb: float | None = None
    gradient_norm: float | None = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VisualizationConfig:
    """Configuration for rich visualization behavior."""

    show_progress_bar: bool = True
    show_tables: bool = True
    table_update_interval: int = 25  # Steps between table updates
    refresh_per_second: int = 4
    show_throughput: bool = True
    show_system_metrics: bool = True
    show_expert_utilization: bool = True
    custom_table_builder: Callable[[TrainingSnapshot], Table] | None = None


class RichTrainerVisualizer:
    """
    Reusable rich console visualizer for training loops.
    Maintains the same style and behavior as the original implementation.
    """

    def __init__(
        self,
        config: VisualizationConfig | None = None,
        console: Console | None = None,
    ):
        """
        Initialize the rich trainer visualizer.

        Args:
            config: Visualization configuration
            console: Rich console instance (creates new if None)
        """
        self.config = config or VisualizationConfig()
        self.console = console or Console()

        # Progress tracking
        self.progress: Progress | None = None
        self.task_id: TaskID | None = None

        # Training history
        self.history: list[TrainingSnapshot] = []
        self.start_time: float | None = None

        # Internal state for calculations
        self._last_step_time: float | None = None
        self._last_loss: float | None = None
        self._last_loss_time: float | None = None
        self._step_times: list[float] = []
        self._loss_history: list[tuple[float, float]] = []  # (time, loss)

    def start(self, total_steps: int, description: str = "Training") -> None:
        """
        Start the visualization with progress tracking.

        Args:
            total_steps: Total number of training steps
            description: Description for the progress bar
        """
        self.start_time = time.time()
        self.last_step_time = self.start_time

        if self.config.show_progress_bar:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("• " + description + " •"),
                TimeRemainingColumn(),
                console=self.console,
                refresh_per_second=self.config.refresh_per_second,
            )
            self.task_id = self.progress.add_task(description, total=total_steps)
            self.progress.start()

    def update(self, snapshot: TrainingSnapshot) -> None:
        """
        Update visualization with new training snapshot.

        Args:
            snapshot: Basic training snapshot (visualizer calculates all derived metrics)
        """
        # Calculate all derived metrics internally
        current_time = time.time()

        # Calculate throughput metrics
        throughput = self._calculate_throughput(snapshot, current_time)

        # Calculate system metrics
        system = self._calculate_system_metrics(snapshot)

        # Calculate loss rate
        loss_per_second = self._calculate_loss_rate(snapshot.train_loss, current_time)

        # Calculate epoch (simple approximation)
        epoch = snapshot.step // 1000  # 1000 steps per epoch approximation

        # Store enriched snapshot
        enriched_snapshot = self._create_enriched_snapshot(
            snapshot, throughput, system, loss_per_second, epoch
        )
        self.history.append(enriched_snapshot)

        # Always update progress bar every step (like original)
        if self.config.show_progress_bar and self.progress and self.task_id is not None:
            self._update_progress_bar(enriched_snapshot)

        # Always show tables every step, with periodic detailed tables (like original)
        if self.config.show_tables:
            self._update_cli_display(enriched_snapshot)

    def _update_progress_bar(self, enriched_snapshot) -> None:
        """Update the rich progress bar with current metrics."""
        if not self.progress or self.task_id is None:
            return

        # Build progress description
        desc_parts = [f"Step {enriched_snapshot.step}"]
        desc_parts.append(f"Loss: {enriched_snapshot.train_loss:.4f}")

        if enriched_snapshot.val_loss is not None:
            desc_parts.append(f"Val: {enriched_snapshot.val_loss:.4f}")

        if enriched_snapshot.learning_rate is not None:
            desc_parts.append(f"LR: {enriched_snapshot.learning_rate:.6f}")

        if enriched_snapshot.throughput.steps_per_second:
            desc_parts.append(
                f"{enriched_snapshot.throughput.steps_per_second:.1f} steps/s"
            )

        if enriched_snapshot.throughput.samples_per_second:
            desc_parts.append(
                f"{enriched_snapshot.throughput.samples_per_second:.0f} samples/s"
            )

        if enriched_snapshot.system.gpu_memory_used_mb:
            desc_parts.append(
                f"GPU: {enriched_snapshot.system.gpu_memory_used_mb:.0f}MB"
            )

        if enriched_snapshot.loss_per_second:
            desc_parts.append(f"ΔLoss: {enriched_snapshot.loss_per_second:.6f}/s")

        description = " | ".join(desc_parts)

        self.progress.update(
            self.task_id,
            advance=1,
            description=description,
        )

    def _update_cli_display(self, enriched_snapshot) -> None:
        """Update CLI display - matches original geometric signals implementation."""
        # Print detailed status every table_update_interval steps (like original every 25 steps)
        if enriched_snapshot.step % self.config.table_update_interval == 0:
            self._display_tables(enriched_snapshot)

    def _display_tables(self, enriched_snapshot) -> None:
        """Display formatted tables with training metrics."""
        self.console.print("\n")

        # Main training status table
        self._display_training_table(enriched_snapshot)

        # Throughput metrics table
        if self.config.show_throughput and self._has_throughput(
            enriched_snapshot.throughput
        ):
            self._display_throughput_table(enriched_snapshot.throughput)

        # System metrics table
        if self.config.show_system_metrics and self._has_system_metrics(
            enriched_snapshot.system
        ):
            self._display_system_table(enriched_snapshot.system)

        # Expert utilization table
        if self.config.show_expert_utilization and enriched_snapshot.expert_utilization:
            self._display_expert_table(enriched_snapshot.expert_utilization)

        # Custom table if provided
        if self.config.custom_table_builder:
            # Pass the original TrainingSnapshot for custom table builders
            original_snapshot = TrainingSnapshot(
                step=enriched_snapshot.step,
                train_loss=enriched_snapshot.train_loss,
                val_loss=enriched_snapshot.val_loss,
                learning_rate=enriched_snapshot.learning_rate,
                aux_loss=enriched_snapshot.aux_loss,
                expert_utilization=enriched_snapshot.expert_utilization,
                custom_metrics=enriched_snapshot.custom_metrics,
            )
            custom_table = self.config.custom_table_builder(original_snapshot)
            if custom_table:
                self.console.print(Panel(custom_table, expand=False))

    def _display_training_table(self, enriched_snapshot) -> None:
        """Display main training status table."""
        table = Table(
            show_header=True,
            header_style="bold blue",
            title="Training Status",
        )

        table.add_column("Step")
        table.add_column("Epoch")
        table.add_column("Train Loss")
        table.add_column("Val Loss")
        table.add_column("LR")
        table.add_column("Loss/s")

        table.add_row(
            str(enriched_snapshot.step),
            str(enriched_snapshot.epoch),
            f"{enriched_snapshot.train_loss:.6f}",
            f"{enriched_snapshot.val_loss:.6f}"
            if enriched_snapshot.val_loss is not None
            else "N/A",
            f"{enriched_snapshot.learning_rate:.6f}"
            if enriched_snapshot.learning_rate is not None
            else "N/A",
            f"{enriched_snapshot.loss_per_second:.6f}"
            if enriched_snapshot.loss_per_second is not None
            else "N/A",
        )

        self.console.print(Panel(table, expand=False))

    def _display_throughput_table(self, throughput: ThroughputMetrics) -> None:
        """Display throughput metrics table."""
        table = Table(
            show_header=True,
            header_style="bold green",
            title="Throughput Metrics",
        )

        table.add_column("Steps/s")
        table.add_column("Batches/s")
        table.add_column("Samples/s")
        table.add_column("Tokens/s")

        table.add_row(
            f"{throughput.steps_per_second:.2f}"
            if throughput.steps_per_second
            else "N/A",
            f"{throughput.batches_per_second:.2f}"
            if throughput.batches_per_second
            else "N/A",
            f"{throughput.samples_per_second:.1f}"
            if throughput.samples_per_second
            else "N/A",
            f"{throughput.tokens_per_second:.0f}"
            if throughput.tokens_per_second
            else "N/A",
        )

        self.console.print(Panel(table, expand=False))

    def _display_system_table(self, system: SystemMetrics) -> None:
        """Display system resource metrics table."""
        table = Table(
            show_header=True,
            header_style="bold yellow",
            title="System Resources",
        )

        table.add_column("GPU Memory")
        table.add_column("GPU Util%")
        table.add_column("Grad Norm")

        gpu_mem_text = (
            f"{system.gpu_memory_used_mb:.1f}MB" if system.gpu_memory_used_mb else "N/A"
        )
        if system.gpu_memory_total_mb:
            gpu_mem_text += f" / {system.gpu_memory_total_mb:.1f}MB"

        table.add_row(
            gpu_mem_text,
            f"{system.gpu_utilization_percent:.1f}%"
            if system.gpu_utilization_percent
            else "N/A",
            f"{system.gradient_norm:.4f}" if system.gradient_norm else "N/A",
        )

        self.console.print(Panel(table, expand=False))

    def _display_expert_table(self, expert_utilization: dict[int, float]) -> None:
        """Display expert utilization table."""
        table = Table(
            show_header=True,
            header_style="bold green",
        )

        table.add_column("Expert")
        table.add_column("Utilization")

        for expert_id in sorted(expert_utilization.keys()):
            utilization = expert_utilization[expert_id]
            table.add_row(str(expert_id), f"{utilization:.3f}")

        self.console.print(Panel(table, title="Expert Utilization", expand=False))

    def _has_throughput(self, throughput: ThroughputMetrics) -> bool:
        """Check if throughput metrics are available."""
        return any(
            [
                throughput.steps_per_second is not None,
                throughput.batches_per_second is not None,
                throughput.samples_per_second is not None,
                throughput.tokens_per_second is not None,
            ]
        )

    def _has_system_metrics(self, system: SystemMetrics) -> bool:
        """Check if system metrics are available."""
        return any(
            [
                system.gpu_memory_used_mb is not None,
                system.gpu_utilization_percent is not None,
                system.gradient_norm is not None,
            ]
        )

    def stop(self) -> None:
        """Stop the visualization and clean up resources."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.task_id = None

    def get_history(self) -> list[TrainingSnapshot]:
        """Get the complete training history."""
        return self.history.copy()

    def _calculate_throughput(
        self, snapshot: TrainingSnapshot, current_time: float
    ) -> ThroughputMetrics:
        """Calculate throughput metrics from timing data."""
        steps_per_second = None
        samples_per_second = None
        tokens_per_second = None

        # Calculate step timing
        if self._last_step_time is not None:
            step_time = current_time - self._last_step_time
            if step_time > 0:
                steps_per_second = 1.0 / step_time
                if snapshot.batch_size:
                    samples_per_second = steps_per_second * snapshot.batch_size
                    if snapshot.sequence_length:
                        tokens_per_second = (
                            samples_per_second * snapshot.sequence_length
                        )

        self._last_step_time = current_time

        return ThroughputMetrics(
            steps_per_second=steps_per_second,
            batches_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            tokens_per_second=tokens_per_second,
        )

    def _calculate_system_metrics(self, snapshot: TrainingSnapshot) -> SystemMetrics:
        """Calculate system resource metrics."""
        return SystemMetrics(
            gpu_memory_used_mb=snapshot.gpu_memory_mb,
            gpu_memory_total_mb=None,  # Not easily available
            gpu_utilization_percent=None,  # Not easily available
            gradient_norm=snapshot.gradient_norm,
        )

    def _calculate_loss_rate(
        self, current_loss: float, current_time: float
    ) -> float | None:
        """Calculate loss change rate."""
        loss_per_second = None

        if self._last_loss is not None and self._last_loss_time is not None:
            time_diff = current_time - self._last_loss_time
            if time_diff > 0:
                loss_diff = (
                    self._last_loss - current_loss
                )  # positive when loss decreasing
                loss_per_second = loss_diff / time_diff

        self._last_loss = current_loss
        self._last_loss_time = current_time

        return loss_per_second

    def _create_enriched_snapshot(
        self,
        snapshot: TrainingSnapshot,
        throughput: ThroughputMetrics,
        system: SystemMetrics,
        loss_per_second: float | None,
        epoch: int,
    ) -> Any:  # Return the old TrainingSnapshot format for compatibility
        """Create enriched snapshot with calculated metrics."""

        # Create a temporary class with all the fields for backward compatibility
        @dataclass
        class EnrichedSnapshot:
            step: int
            epoch: int
            train_loss: float
            val_loss: float | None
            learning_rate: float | None
            aux_loss: float | None
            loss_per_second: float | None
            throughput: ThroughputMetrics
            system: SystemMetrics
            expert_utilization: dict[int, float]
            custom_metrics: dict[str, Any]

        return EnrichedSnapshot(
            step=snapshot.step,
            epoch=epoch,
            train_loss=snapshot.train_loss,
            val_loss=snapshot.val_loss,
            learning_rate=snapshot.learning_rate,
            aux_loss=snapshot.aux_loss,
            loss_per_second=loss_per_second,
            throughput=throughput,
            system=system,
            expert_utilization=snapshot.expert_utilization,
            custom_metrics=snapshot.custom_metrics,
        )

    def print_final_summary(self) -> None:
        """Print a final summary of the training session."""
        if not self.history:
            return

        final_snapshot = self.history[-1]
        total_time = time.time() - self.start_time if self.start_time else 0

        # Create summary table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            title="Training Summary",
        )

        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Total Steps", str(final_snapshot.step))
        table.add_row("Final Train Loss", f"{final_snapshot.train_loss:.6f}")

        if final_snapshot.val_loss is not None:
            table.add_row("Final Val Loss", f"{final_snapshot.val_loss:.6f}")

        table.add_row("Total Time", f"{total_time:.1f}s")

        # Calculate average throughput
        throughput_samples = [
            s.throughput.samples_per_second
            for s in self.history
            if hasattr(s, "throughput") and s.throughput.samples_per_second is not None
        ]
        if throughput_samples:
            avg_throughput = sum(throughput_samples) / len(throughput_samples)
            table.add_row("Avg Samples/s", f"{avg_throughput:.1f}")

        self.console.print("\n")
        self.console.print(Panel(table, expand=False))


# Helper functions for common use cases


def create_default_visualizer(
    show_progress: bool = True,
    show_tables: bool = True,
    table_interval: int = 25,
) -> RichTrainerVisualizer:
    """
    Create a visualizer with default settings.

    Args:
        show_progress: Whether to show progress bar
        show_tables: Whether to show status tables
        table_interval: Steps between table updates

    Returns:
        Configured RichTrainerVisualizer instance
    """
    config = VisualizationConfig(
        show_progress_bar=show_progress,
        show_tables=show_tables,
        table_update_interval=table_interval,
    )
    return RichTrainerVisualizer(config)


def snapshot_from_trainer_state(
    step: int,
    epoch: int,
    train_loss: float,
    val_loss: float | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    step_time: float | None = None,
    gpu_stats: dict[str, float] | None = None,
    expert_stats: dict[int, float] | None = None,
    **kwargs: Any,
) -> TrainingSnapshot:
    """
    Create a training snapshot from common trainer state values.

    Args:
        step: Current training step
        epoch: Current epoch
        train_loss: Training loss value
        val_loss: Validation loss value (optional)
        learning_rate: Current learning rate (optional)
        batch_size: Batch size for throughput calculation
        sequence_length: Sequence length for token throughput
        step_time: Time taken for this step in seconds
        gpu_stats: GPU utilization statistics
        expert_stats: Expert utilization statistics
        **kwargs: Additional custom metrics

    Returns:
        TrainingSnapshot instance
    """
    # Calculate throughput metrics if timing info available (not currently used)
    ThroughputMetrics()
    if step_time and step_time > 0:
        steps_per_second = 1.0 / step_time
        batches_per_second = steps_per_second
        samples_per_second = steps_per_second * batch_size if batch_size else None
        tokens_per_second = (
            samples_per_second * sequence_length
            if samples_per_second and sequence_length
            else None
        )

        # Throughput metrics calculated but not currently used
        ThroughputMetrics(
            steps_per_second=steps_per_second,
            batches_per_second=batches_per_second,
            samples_per_second=samples_per_second,
            tokens_per_second=tokens_per_second,
        )

    # Extract system metrics from gpu_stats (not currently used)
    SystemMetrics()
    if gpu_stats:
        SystemMetrics(
            gpu_memory_used_mb=gpu_stats.get("memory_used_mb"),
            gpu_memory_total_mb=gpu_stats.get("memory_total_mb"),
            gpu_utilization_percent=gpu_stats.get("utilization_percent"),
            gradient_norm=gpu_stats.get("gradient_norm"),
        )

    return TrainingSnapshot(
        step=step,
        train_loss=train_loss,
        val_loss=val_loss,
        learning_rate=learning_rate,
        expert_utilization=expert_stats or {},
        custom_metrics=kwargs,
    )


class MinimalTrainerVisualizer:
    """
    Minimal version that only shows progress bar without tables.
    For simpler training loops that just need basic progress tracking.
    """

    def __init__(self, console: Console | None = None):
        """Initialize minimal visualizer."""
        self.console = console or Console()
        self.progress: Progress | None = None
        self.task_id: TaskID | None = None

    def start(self, total_steps: int) -> None:
        """Start progress tracking."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.task_id = self.progress.add_task("Training", total=total_steps)
        self.progress.start()

    def update(self, step: int, loss: float, **kwargs: Any) -> None:
        """Update progress with current step and loss."""
        if self.progress and self.task_id is not None:
            desc = f"Step {step} | Loss: {loss:.4f}"
            for key, value in kwargs.items():
                if value is not None:
                    if isinstance(value, float):
                        desc += f" | {key}: {value:.4f}"
                    else:
                        desc += f" | {key}: {value}"

            self.progress.update(self.task_id, advance=1, description=desc)

    def stop(self) -> None:
        """Stop progress tracking."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.task_id = None
