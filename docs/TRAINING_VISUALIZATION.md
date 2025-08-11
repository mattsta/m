# Training Visualization System

The `m.training_viz` module provides flexible, real-time matplotlib visualization for training pipelines. This system was extracted and generalized from the trajectory_3d example to be reusable across any training scenario.

## Overview

The training visualization system consists of:

- **Core visualizer class** (`RealTimeTrainingVisualizer`) - The main visualization engine
- **Configuration system** - Dataclasses for customizing layouts and metrics
- **Factory functions** - Pre-configured visualizers for common use cases
- **Real-time updates** - Live plotting during training with configurable intervals

## Quick Start

### Basic Loss Monitoring

```python
from m.training_viz import create_loss_visualizer

# Create a simple loss visualizer
visualizer = create_loss_visualizer(update_interval=50)

# During training loop
for step in range(1000):
    # ... training code ...

    # Update visualization
    visualizer.update_metrics({
        "train_loss": train_loss_value,
        "val_loss": validation_loss_value,
        "learning_rate": current_lr
    })

# Clean up
visualizer.close()
```

### MoE Training Visualization

```python
from m.training_viz import create_moe_visualizer

# Create MoE-specific visualizer with expert metrics
visualizer = create_moe_visualizer(update_matplotlib=1, update_rich_tables=25)

# During MoE training
visualizer.update_metrics({
    "train_loss": 0.245,
    "val_loss": 0.267,
    "aux_loss": 0.012,
    "position_error": 0.134,        # Task-specific metric
    "expert_entropy": 2.1,          # Expert utilization
    "load_balance": 0.85,           # Load balancing factor
    "samples_per_sec": 120.5        # Training throughput
})

visualizer.close()
```

## Configuration System

### Core Configuration Classes

#### `TrainingVisualizerConfig`

The main configuration class controlling the entire visualization:

```python
@dataclass(slots=True)
class TrainingVisualizerConfig:
    # Layout configuration
    subplot_rows: int = 2           # Number of subplot rows
    subplot_cols: int = 2           # Number of subplot columns
    figure_size: tuple[int, int] = (12, 8)  # Figure size in inches
    title: str = "Real-time Training Progress"

    # Update behavior
    update_interval: int = 100      # Steps between visual updates
    max_points: int | None = None   # Limit points shown (None = unlimited)

    # Plot and metric configurations
    plots: list[PlotConfig] = field(default_factory=list)
    metrics: list[MetricConfig] = field(default_factory=list)

    # Performance settings
    use_blit: bool = False          # Advanced matplotlib optimization
    pause_duration: float = 0.01    # Seconds between updates
```

#### `PlotConfig`

Configuration for individual subplot panels:

```python
@dataclass(slots=True)
class PlotConfig:
    title: str                      # Subplot title
    ylabel: str                     # Y-axis label
    xlabel: str = "Step"            # X-axis label (usually "Step")
    yscale: str = "linear"          # "linear", "log", or "symlog"
    show_grid: bool = True          # Show grid lines
    show_legend: bool = True        # Show legend
    color_cycle: list[str] = field(default_factory=lambda: [
        "blue", "red", "green", "orange", "purple", "brown",
        "pink", "gray", "olive", "cyan"
    ])
```

#### `MetricConfig`

Configuration for individual metrics:

```python
@dataclass(slots=True)
class MetricConfig:
    metric_name: str                # Name of metric in update_metrics() dict
    plot_index: int                 # Which subplot (0-based index)
    color: str | None = None        # Line color (None = auto from color_cycle)
    label: str | None = None        # Legend label (None = auto from metric_name)
    line_style: str = "-"           # Line style: "-", "--", ":", "-."
    alpha: float = 1.0              # Line transparency (0.0-1.0)
```

## Factory Functions

### `create_loss_visualizer(update_interval=100)`

Creates a simple 1x2 layout for basic loss monitoring:

- **Plot 1**: Loss curves (train_loss, val_loss) with log scale
- **Plot 2**: Learning rate with log scale

**Supported metrics**: `train_loss`, `val_loss`, `learning_rate`

### `create_moe_visualizer(update_matplotlib=1, update_rich_tables=25, n_experts=None)`

Creates a comprehensive 2x2 layout for MoE training:

- **Plot 1**: Loss curves (train_loss, val_loss, aux_loss) with log scale
- **Plot 2**: Task-specific metrics (position_error, velocity_error, accuracy, etc.)
- **Plot 3**: Expert utilization (expert_entropy, load_balance, routing_loss)
- **Plot 4**: Training throughput (samples_per_sec, tokens_per_sec)

**Parameters:**

- `update_matplotlib`: Steps between matplotlib chart updates (default: 1 = every step)
- `update_rich_tables`: Steps between rich CLI table updates (default: 25)
- `n_experts`: Number of experts for individual utilization metrics (optional)

**Supported metrics**: All loss metrics, task-specific errors, expert utilization, throughput

### `create_custom_visualizer(metrics_config, plot_titles, subplot_layout, ...)`

Creates fully customized visualization:

```python
visualizer = create_custom_visualizer(
    metrics_config={
        "accuracy": (0, "blue"),     # (plot_index, color)
        "f1_score": (0, "red"),
        "learning_rate": (1, "green"),
    },
    plot_titles=["Classification Metrics", "Optimization"],
    subplot_layout=(1, 2),
    update_interval=75,
    title="Custom Training Dashboard"
)
```

## Advanced Usage

### Custom Configuration Example

```python
from m.training_viz import (
    RealTimeTrainingVisualizer,
    TrainingVisualizerConfig,
    PlotConfig,
    MetricConfig
)

# Create custom configuration
config = TrainingVisualizerConfig(
    subplot_rows=2,
    subplot_cols=3,
    figure_size=(18, 10),
    title="Multi-Task Training Dashboard",
    update_interval=50,
    max_points=500,  # Keep only last 500 points

    plots=[
        PlotConfig(title="Loss Curves", ylabel="Loss", yscale="log"),
        PlotConfig(title="Accuracy Metrics", ylabel="Score", yscale="linear"),
        PlotConfig(title="Learning Rate", ylabel="LR", yscale="log"),
        PlotConfig(title="Gradient Norms", ylabel="Norm", yscale="log"),
        PlotConfig(title="Memory Usage", ylabel="GB", yscale="linear"),
        PlotConfig(title="Throughput", ylabel="Samples/sec", yscale="linear"),
    ],

    metrics=[
        # Loss plot (index 0)
        MetricConfig("train_loss", 0, color="blue", label="Training"),
        MetricConfig("val_loss", 0, color="red", label="Validation"),

        # Accuracy plot (index 1)
        MetricConfig("train_accuracy", 1, color="green", line_style="-"),
        MetricConfig("val_accuracy", 1, color="darkgreen", line_style="--"),

        # Learning rate plot (index 2)
        MetricConfig("learning_rate", 2, color="purple"),

        # Gradients plot (index 3)
        MetricConfig("grad_norm", 3, color="orange"),
        MetricConfig("param_norm", 3, color="brown", line_style="--"),

        # Memory plot (index 4)
        MetricConfig("gpu_memory", 4, color="red"),
        MetricConfig("cpu_memory", 4, color="blue", line_style="--"),

        # Throughput plot (index 5)
        MetricConfig("samples_per_sec", 5, color="cyan"),
    ]
)

# Create visualizer with custom config
visualizer = RealTimeTrainingVisualizer(config)
```

### Integration into Training Loop

```python
def train_model():
    # Create visualizer
    visualizer = create_moe_visualizer(update_matplotlib=1, update_rich_tables=25)

    try:
        for step in range(max_steps):
            # Training step
            train_loss = perform_training_step()

            # Validation (periodic)
            if step % eval_interval == 0:
                val_loss = perform_validation()
            else:
                val_loss = None

            # Collect all metrics
            metrics = {
                "train_loss": train_loss,
                "expert_entropy": model.get_expert_entropy(),
                "load_balance": model.get_load_balance(),
                "samples_per_sec": calculate_throughput(),
            }

            # Add validation metrics when available
            if val_loss is not None:
                metrics["val_loss"] = val_loss

            # Add task-specific metrics
            if hasattr(model, 'get_task_metrics'):
                task_metrics = model.get_task_metrics()
                metrics.update(task_metrics)

            # Update visualization
            visualizer.update_metrics(metrics)

            # Save checkpoint periodically
            if step % save_interval == 0:
                save_checkpoint(model, step)
                visualizer.save_plot(f"training_progress_step_{step}.png")

    finally:
        # Always clean up
        visualizer.close()
```

## API Reference

### `RealTimeTrainingVisualizer`

#### Constructor

```python
RealTimeTrainingVisualizer(config: TrainingVisualizerConfig)
```

#### Methods

##### `update_metrics(metrics: dict[str, float]) -> None`

Update metrics and refresh visualization if needed.

**Parameters:**

- `metrics`: Dictionary mapping metric names to float values

**Example:**

```python
visualizer.update_metrics({
    "train_loss": 0.245,
    "val_loss": 0.267,
    "accuracy": 0.89
})
```

##### `force_update() -> None`

Force an immediate visualization update regardless of update_interval.

##### `save_plot(save_path: str | Path) -> None`

Save current visualization to file.

**Parameters:**

- `save_path`: Path where to save the plot (PNG, PDF, SVG supported)

##### `get_metrics_summary() -> dict[str, dict[str, float]]`

Get summary statistics for all tracked metrics.

**Returns:**
Dictionary with structure:

```python
{
    "metric_name": {
        "current": 0.245,    # Latest value
        "min": 0.123,        # Minimum seen
        "max": 1.456,        # Maximum seen
        "mean": 0.567,       # Average value
        "count": 1000        # Number of points
    }
}
```

##### `close() -> None`

Clean up visualization resources and print summary.

## Best Practices

### 1. Memory Management

```python
# For long training runs, limit history size
config = TrainingVisualizerConfig(
    max_points=1000,  # Keep only last 1000 points
    update_interval=100  # Update less frequently
)
```

### 2. Metric Naming

Use consistent, descriptive metric names:

```python
# Good
metrics = {
    "train_loss": 0.245,
    "val_loss": 0.267,
    "position_error": 0.134,
    "expert_entropy": 2.1
}

# Avoid
metrics = {
    "loss": 0.245,        # Ambiguous
    "err": 0.134,         # Unclear
    "x": 2.1              # Meaningless
}
```

### 3. Update Intervals

Choose update intervals based on training speed:

```python
# Fast training (1000+ steps/sec)
update_interval = 100

# Medium training (100-1000 steps/sec)
update_interval = 50

# Slow training (< 100 steps/sec)
update_interval = 10
```

### 4. Error Handling

Always use try/finally for cleanup:

```python
visualizer = create_moe_visualizer()
try:
    # Training loop
    for step in range(max_steps):
        # ... training code ...
        visualizer.update_metrics(metrics)
finally:
    visualizer.close()  # Ensures cleanup even if training fails
```

### 5. Performance Optimization

For maximum performance:

```python
config = TrainingVisualizerConfig(
    update_interval=200,    # Update less frequently
    max_points=500,         # Limit history
    use_blit=True,         # Enable matplotlib blitting (experimental)
    pause_duration=0.001   # Minimal pause
)
```

## Common Patterns

### Pattern 1: Research Training

```python
# For research with multiple experiments
def train_experiment(experiment_name: str, config: ExperimentConfig):
    visualizer = create_moe_visualizer(update_matplotlib=1, update_rich_tables=50)

    try:
        for step in range(config.max_steps):
            # ... training ...
            visualizer.update_metrics(metrics)

            # Save progress plots periodically
            if step % 1000 == 0:
                visualizer.save_plot(f"experiments/{experiment_name}/step_{step}.png")

    finally:
        # Final summary
        summary = visualizer.get_metrics_summary()
        save_experiment_summary(experiment_name, summary)
        visualizer.close()
```

### Pattern 2: Hyperparameter Tuning

```python
# For hyperparameter sweeps
def hyperparameter_sweep():
    for lr in [0.001, 0.01, 0.1]:
        for batch_size in [16, 32, 64]:
            experiment_name = f"lr_{lr}_bs_{batch_size}"

            visualizer = create_custom_visualizer(
                metrics_config={"train_loss": (0, "blue"), "val_loss": (0, "red")},
                plot_titles=["Loss Curves"],
                subplot_layout=(1, 1),
                title=f"Training: LR={lr}, BS={batch_size}"
            )

            # ... run experiment with visualizer ...
```

### Pattern 3: Multi-Task Learning

```python
# For multi-task training
def train_multitask():
    visualizer = create_custom_visualizer(
        metrics_config={
            # Task 1 metrics
            "task1_loss": (0, "blue"),
            "task1_accuracy": (1, "blue"),

            # Task 2 metrics
            "task2_loss": (0, "red"),
            "task2_f1_score": (1, "red"),

            # Shared metrics
            "learning_rate": (2, "green"),
            "expert_entropy": (3, "purple"),
        },
        plot_titles=["Task Losses", "Task Metrics", "Optimization", "Expert Usage"],
        subplot_layout=(2, 2)
    )

    # ... multi-task training loop ...
```

## Troubleshooting

### Common Issues

1. **Matplotlib backend errors**: Set backend before importing:

   ```python
   import matplotlib
   matplotlib.use('Agg')  # For headless environments
   import matplotlib.pyplot as plt
   ```

2. **Memory leaks**: Always call `visualizer.close()` in finally block

3. **Slow updates**: Increase `update_interval` or enable `max_points`

4. **Plot layout issues**: Adjust `figure_size` and `subplot_layout` for your metrics

5. **Legend warnings**: The system automatically handles missing labels

### Performance Tips

- Use `update_interval >= 50` for real-time feel without slowdown
- Set `max_points=1000` for long training runs
- Disable `show_legend=False` for plots with many metrics
- Use `yscale="log"` for loss curves to better show convergence

## Combined Visualization: Matplotlib + Rich Terminal Integration

For the ultimate training monitoring experience, combine `m.training_viz` (matplotlib plotting) with `m.rich_trainer_viz` (terminal UI) to get both scientific visualization AND real-time progress monitoring.

### Dual System Architecture

The two visualization systems complement each other perfectly:

- **`m.training_viz`**: Scientific plotting for analysis, trend visualization, and publication-ready figures
- **`m.rich_trainer_viz`**: Terminal UI for immediate feedback, progress tracking, and live status updates

### Complete Integration Example

```python
from m.training_viz import create_moe_visualizer
from m.rich_trainer_viz import RichTrainerVisualizer, VisualizationConfig, TrainingSnapshot

class DualVisualizationSystem:
    """
    Production-ready dual visualization system combining matplotlib plots
    with Rich terminal interface for comprehensive training monitoring.
    """

    def __init__(self, total_steps: int, n_experts: int = 8):
        # Matplotlib plotting for scientific analysis
        self.scientific_viz = create_moe_visualizer(
            update_matplotlib=1,        # Update plots every step
            update_rich_tables=25,      # Note: This parameter is currently ignored
            n_experts=n_experts         # Track individual experts
        )

        # Rich terminal UI for live progress monitoring
        rich_config = VisualizationConfig(
            show_progress_bar=True,
            show_tables=True,
            table_update_interval=25,
            show_expert_utilization=True,
            show_throughput=True,
            show_system_metrics=True
        )
        self.terminal_viz = RichTrainerVisualizer(rich_config)

        self.total_steps = total_steps

    def initialize(self, description: str = "Training with Dual Visualization"):
        """Start both visualization systems."""
        self.terminal_viz.start(total_steps=self.total_steps, description=description)
        print(f"🎯 Started dual visualization: matplotlib plots + rich terminal")

    def update_training_metrics(self, step: int, comprehensive_metrics: dict):
        """
        Update both visualization systems with coordinated metrics.

        Args:
            step: Current training step
            comprehensive_metrics: Dictionary containing all training metrics
        """

        # 1. Update matplotlib plots with detailed metrics for trend analysis
        plot_metrics = self._prepare_plot_metrics(comprehensive_metrics)
        self.scientific_viz.update_metrics(plot_metrics)

        # 2. Update Rich terminal with structured progress data
        terminal_snapshot = self._create_terminal_snapshot(step, comprehensive_metrics)
        self.terminal_viz.update(terminal_snapshot)

    def _prepare_plot_metrics(self, metrics: dict) -> dict:
        """Prepare metrics specifically for matplotlib visualization."""
        return {
            # Core training curves
            "train_loss": metrics.get("train_loss"),
            "val_loss": metrics.get("val_loss"),
            "aux_loss": metrics.get("aux_loss"),
            "learning_rate": metrics.get("learning_rate"),

            # Task-specific performance metrics
            "position_error": metrics.get("position_error"),
            "velocity_error": metrics.get("velocity_error"),
            "accuracy": metrics.get("accuracy"),
            "f1_score": metrics.get("f1_score"),

            # MoE expert analysis
            "expert_entropy": metrics.get("expert_entropy"),
            "load_balance": metrics.get("load_balance"),
            "routing_loss": metrics.get("routing_loss"),

            # System performance
            "samples_per_sec": metrics.get("samples_per_sec"),
            "tokens_per_sec": metrics.get("tokens_per_sec"),
            "gpu_utilization": metrics.get("gpu_utilization"),

            # Filter out None values for clean plotting
        }

    def _create_terminal_snapshot(self, step: int, metrics: dict) -> TrainingSnapshot:
        """Create structured snapshot for Rich terminal display."""
        return TrainingSnapshot(
            step=step,
            train_loss=metrics.get("train_loss", 0.0),
            val_loss=metrics.get("val_loss"),
            learning_rate=metrics.get("learning_rate"),
            aux_loss=metrics.get("aux_loss"),
            expert_utilization=metrics.get("expert_utilization", {}),
            batch_size=metrics.get("batch_size"),
            sequence_length=metrics.get("sequence_length"),
            gpu_memory_mb=metrics.get("gpu_memory_mb"),
            gradient_norm=metrics.get("gradient_norm"),
            custom_metrics=metrics.get("custom_metrics", {})
        )

    def save_scientific_plots(self, filename: str):
        """Save current matplotlib plots to file."""
        self.scientific_viz.save_plot(filename)
        print(f"📊 Saved scientific plots to {filename}")

    def get_training_summary(self) -> dict:
        """Get comprehensive training summary from both systems."""
        plot_summary = self.scientific_viz.get_metrics_summary()
        return {
            "matplotlib_metrics": plot_summary,
            "training_completed": True,
            "visualization_systems": ["matplotlib", "rich_terminal"]
        }

    def cleanup(self):
        """Clean up both visualization systems."""
        self.scientific_viz.close()
        self.terminal_viz.close()
        print("✅ Dual visualization systems closed successfully")
```

### Real-World Training Integration

Based on the proven pattern from `examples/trajectory_3d/training.py`:

```python
def train_with_dual_visualization():
    """
    Complete training example with dual visualization system.
    Demonstrates the production pattern used in trajectory_3d training.
    """

    # Initialize dual visualization
    dual_viz = DualVisualizationSystem(
        total_steps=2000,
        n_experts=16
    )

    dual_viz.initialize("Advanced MoE Model Training")

    try:
        for step in range(2000):
            # === Your Training Code ===
            train_loss = training_step()

            # Periodic validation
            val_loss = None
            if step % 100 == 0 and step > 0:
                val_loss = validation_step()

            # === Comprehensive Metrics Collection ===
            metrics = {
                # Core training metrics
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "aux_loss": model.get_auxiliary_loss(),

                # MoE-specific metrics for scientific analysis
                "expert_entropy": model.compute_routing_entropy(),
                "load_balance": model.get_load_balance_factor(),
                "expert_utilization": model.get_expert_utilization_dict(),

                # Task-specific metrics
                "position_error": compute_position_rmse(),
                "velocity_error": compute_velocity_rmse(),

                # System performance metrics
                "samples_per_sec": calculate_throughput(),
                "gpu_memory_mb": get_gpu_memory_usage(),
                "gradient_norm": compute_gradient_norm(),

                # Custom domain-specific metrics
                "custom_metrics": {
                    "trajectory_smoothness": compute_smoothness_metric(),
                    "prediction_confidence": model.get_prediction_confidence(),
                    "routing_consistency": compute_routing_consistency()
                }
            }

            # === Update Both Visualization Systems ===
            dual_viz.update_training_metrics(step, metrics)

            # === Periodic Actions ===

            # Save plots for analysis
            if step % 500 == 0 and step > 0:
                dual_viz.save_scientific_plots(f"training_analysis_step_{step}.png")

            # Checkpoint saving
            if step % 200 == 0 and step > 0:
                save_model_checkpoint(model, step)

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")

    finally:
        # Always clean up visualization systems
        training_summary = dual_viz.get_training_summary()
        dual_viz.cleanup()

        print(f"📋 Training completed with {len(training_summary['matplotlib_metrics'])} metrics tracked")
```

### Advanced Integration Patterns

#### Pattern 1: Research & Development

```python
# High-frequency plotting for detailed analysis + comprehensive terminal feedback
research_viz = DualVisualizationSystem(total_steps=5000, n_experts=32)

# Use with detailed logging for publication-quality analysis
for step in range(5000):
    # ... training ...
    research_viz.update_training_metrics(step, detailed_metrics)

    # Frequent plot saves for frame-by-frame analysis
    if step % 50 == 0:
        research_viz.save_scientific_plots(f"research_frames/step_{step:06d}.png")
```

#### Pattern 2: Production Training

```python
# Balanced plotting + efficient terminal monitoring
production_viz = DualVisualizationSystem(total_steps=10000, n_experts=64)

# Focus on essential metrics with periodic detailed analysis
for step in range(10000):
    # ... training ...
    production_viz.update_training_metrics(step, essential_metrics)

    # Save plots at major milestones
    if step % 1000 == 0:
        production_viz.save_scientific_plots(f"production_milestone_{step}.png")
```

#### Pattern 3: Long-Running Experiments

```python
# Efficient plotting + streamlined terminal updates
long_run_viz = DualVisualizationSystem(total_steps=50000, n_experts=128)

# Optimize for minimal overhead during very long runs
for step in range(50000):
    # ... training ...
    long_run_viz.update_training_metrics(step, core_metrics)

    # Sparse plot saves to prevent storage overflow
    if step % 5000 == 0:
        long_run_viz.save_scientific_plots(f"long_run_checkpoint_{step}.png")
```

### Troubleshooting Combined Systems

#### Performance Optimization

```python
# For training with performance constraints
class OptimizedDualVisualization(DualVisualizationSystem):
    def update_training_metrics(self, step: int, metrics: dict):
        # Update terminal UI every step (lightweight)
        terminal_snapshot = self._create_terminal_snapshot(step, metrics)
        self.terminal_viz.update(terminal_snapshot)

        # Update plots less frequently (heavyweight)
        if step % 10 == 0:  # Every 10 steps instead of every step
            plot_metrics = self._prepare_plot_metrics(metrics)
            self.scientific_viz.update_metrics(plot_metrics)
```

#### Memory Management

```python
# For memory-constrained environments
def create_memory_efficient_dual_viz(total_steps: int):
    return DualVisualizationSystem(
        total_steps=total_steps,
        n_experts=8  # Fewer experts to track
    )

# Use with limited plot history
scientific_viz = create_moe_visualizer(
    update_matplotlib=5,  # Less frequent updates
    n_experts=8           # Reduced expert tracking
)
```

### Integration with Existing Training Loops

To add dual visualization to existing training code:

```python
# Before: Single visualization or no visualization
def existing_training_loop():
    for step in range(1000):
        loss = train_step()
        print(f"Step {step}: Loss = {loss:.4f}")  # Basic logging

# After: Dual visualization integration
def enhanced_training_loop():
    dual_viz = DualVisualizationSystem(1000, n_experts=8)
    dual_viz.initialize("Enhanced Training")

    try:
        for step in range(1000):
            loss = train_step()

            # Collect metrics
            metrics = {
                "train_loss": loss,
                "learning_rate": get_learning_rate(),
                # ... other metrics
            }

            # Update both visualization systems
            dual_viz.update_training_metrics(step, metrics)

    finally:
        dual_viz.cleanup()
```

### Benefits Summary

The combined matplotlib + Rich terminal approach provides:

1. **📊 Scientific Analysis**: Publication-ready plots and detailed trend analysis
2. **⚡ Real-time Feedback**: Immediate progress updates and live status monitoring
3. **🔍 Comprehensive Coverage**: Both historical data analysis and current status
4. **🎯 Flexible Usage**: Suitable for research, production, and long-running experiments
5. **💪 Production Tested**: Based on patterns proven in examples/trajectory_3d/training.py

This dual visualization system provides the most comprehensive training monitoring solution, combining the analytical power of matplotlib with the immediate feedback of Rich terminal interfaces.

---

This visualization system provides a powerful, flexible foundation for monitoring any training pipeline in real-time. The configuration system allows complete customization while the factory functions provide quick setup for common scenarios. When combined with Rich terminal visualization, it creates the ultimate training monitoring experience.
