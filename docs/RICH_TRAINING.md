# Rich Trainer Visualization

The `m.rich_trainer_viz` module provides beautiful, real-time CLI progress tracking and metrics display for training pipelines. This system offers comprehensive progress bars, live metrics tables, and customizable displays for monitoring training progress.

## Features

- 🎨 **Beautiful CLI Progress Bars**: Real-time progress tracking with spinners, ETA, and completion counters
- 📊 **Live Metrics Tables**: Comprehensive training status, throughput, system resources, and expert utilization displays
- ⚙️ **Highly Configurable**: Flexible configuration system for customizing display behavior and styling
- 🔧 **Factory Functions**: Pre-configured visualizers for common training scenarios
- 🧪 **Production Ready**: Well-tested and reliable visualization component
- 📈 **Expert Utilization**: Specialized displays for MoE (Mixture of Experts) training metrics
- 🌊 **Real-time Updates**: Live updating displays with configurable refresh rates and intervals

## Quick Start

### Basic Training Visualization

```python
from m.rich_trainer_viz import RichTrainerVisualizer, TrainingSnapshot

# Create visualizer with default settings
visualizer = RichTrainerVisualizer()
visualizer.start(total_steps=1000, description="Training Model")

for step in range(1000):
    # ... your training code ...
    loss = train_one_step()

    # Update the display
    snapshot = TrainingSnapshot(
        step=step,
        train_loss=loss,
        learning_rate=get_current_lr(),
        custom_metrics={"accuracy": 0.95}
    )
    visualizer.update(snapshot)

visualizer.close()
```

### Using the Factory Function

```python
from m.rich_trainer_viz import create_default_visualizer

# Create with custom settings
visualizer = create_default_visualizer(
    show_progress=True,
    show_tables=True,
    table_interval=25
)

# Use as above...
```

## Core Components

### TrainingSnapshot

The main data container for training metrics:

```python
@dataclass(slots=True, frozen=True)
class TrainingSnapshot:
    # Required fields
    step: int
    train_loss: float

    # Optional training metrics
    val_loss: float | None = None
    learning_rate: float | None = None
    aux_loss: float | None = None

    # MoE-specific metrics
    expert_utilization: dict[int, float] = field(default_factory=dict)

    # System and performance metrics
    batch_size: int | None = None
    sequence_length: int | None = None
    gpu_memory_mb: float | None = None
    gradient_norm: float | None = None

    # Extensible custom metrics
    custom_metrics: dict[str, Any] = field(default_factory=dict)
```

### VisualizationConfig

Configuration class for customizing display behavior:

```python
@dataclass(slots=True)
class VisualizationConfig:
    show_progress_bar: bool = True          # Show progress bar
    show_tables: bool = True                # Show metrics tables
    table_update_interval: int = 25         # Steps between table updates
    refresh_per_second: int = 4             # Updates per second
    show_throughput: bool = True            # Show throughput metrics
    show_system_metrics: bool = True        # Show system resource usage
    show_expert_utilization: bool = True    # Show expert usage (for MoE)
    custom_table_builder: Callable[[TrainingSnapshot], Table] | None = None
```

### Metric Data Classes

Additional structured metrics containers:

```python
@dataclass(slots=True, frozen=True)
class ThroughputMetrics:
    steps_per_second: float | None = None
    batches_per_second: float | None = None
    samples_per_second: float | None = None
    tokens_per_second: float | None = None

@dataclass(slots=True, frozen=True)
class SystemMetrics:
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    gpu_utilization_percent: float | None = None
    gradient_norm: float | None = None
```

## RichTrainerVisualizer API

### Constructor

```python
RichTrainerVisualizer(
    config: VisualizationConfig | None = None,
    console: Console | None = None
)
```

**Parameters:**

- `config`: Visualization configuration (uses defaults if None)
- `console`: Rich console instance (creates new if None)

### Methods

#### `start(total_steps: int, description: str = "Training") -> None`

Initialize the progress display.

**Parameters:**

- `total_steps`: Total number of training steps
- `description`: Description shown in progress bar

#### `update(snapshot: TrainingSnapshot) -> None`

Update the visualization with new training metrics.

**Parameters:**

- `snapshot`: TrainingSnapshot containing current metrics

#### `close() -> None`

Clean up and show final training summary.

## Advanced Usage

### Custom Configuration

```python
from m.rich_trainer_viz import RichTrainerVisualizer, VisualizationConfig

# Create custom configuration
config = VisualizationConfig(
    show_progress_bar=True,
    show_tables=True,
    table_update_interval=10,  # More frequent updates
    refresh_per_second=8,      # Faster refresh rate
    show_throughput=True,
    show_system_metrics=False, # Disable system metrics
    show_expert_utilization=True
)

visualizer = RichTrainerVisualizer(config)
```

### Custom Table Builder

```python
from rich.table import Table

def build_custom_table(snapshot: TrainingSnapshot) -> Table:
    table = Table(title="Custom Metrics", style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for name, value in snapshot.custom_metrics.items():
        table.add_row(name.replace("_", " ").title(), f"{value:.4f}")

    return table

config = VisualizationConfig(
    custom_table_builder=build_custom_table
)
visualizer = RichTrainerVisualizer(config)
```

### MoE Training Example

```python
from m.rich_trainer_viz import RichTrainerVisualizer, TrainingSnapshot

# Configure for MoE training
config = VisualizationConfig(
    show_expert_utilization=True,
    table_update_interval=25,
    show_throughput=True
)

visualizer = RichTrainerVisualizer(config)
visualizer.start(total_steps=5000, description="Training MoE Model")

for step in range(5000):
    # ... MoE training code ...
    loss, aux_loss, expert_util = train_moe_step()

    snapshot = TrainingSnapshot(
        step=step,
        train_loss=loss,
        aux_loss=aux_loss,
        expert_utilization={0: 0.2, 1: 0.3, 2: 0.5},  # Per-expert usage
        learning_rate=scheduler.get_last_lr()[0],
        custom_metrics={
            "routing_entropy": 2.1,
            "load_balance": 0.95,
            "position_error": 0.123
        }
    )
    visualizer.update(snapshot)

visualizer.close()
```

## Display Components

### Progress Bar

The progress bar displays:

- Animated spinner indicating active training
- Progress bar with completion percentage
- Current step / total steps
- Estimated time remaining
- Custom task description

### Training Status Table

Shows core training metrics:

- Current step and loss values
- Learning rate with precision formatting
- Auxiliary loss (for MoE models)
- Validation loss (when available)

### Expert Utilization Table

For MoE training scenarios:

- Per-expert utilization rates
- Load balancing indicators
- Expert activation patterns
- Routing statistics

### Throughput Metrics

Performance information:

- Steps per second
- Samples per second
- Training efficiency indicators

### System Resources

Hardware utilization:

- GPU memory usage
- System performance metrics
- Resource efficiency indicators

### Custom Metrics Display

Domain-specific metrics from `custom_metrics` field:

- Task-specific errors (position, velocity)
- Model-specific metrics (perplexity, accuracy)
- Any custom float metrics

## Integration Examples

### Basic Training Loop

```python
def train_model(model, dataloader, optimizer, scheduler, num_steps):
    visualizer = RichTrainerVisualizer()
    visualizer.start(total_steps=num_steps, description="Training")

    try:
        for step in range(num_steps):
            # Training step
            loss = training_step(model, dataloader, optimizer)

            # Create snapshot
            snapshot = TrainingSnapshot(
                step=step,
                train_loss=loss,
                learning_rate=scheduler.get_last_lr()[0]
            )

            # Update visualization
            visualizer.update(snapshot)

            # Step scheduler
            scheduler.step()

    finally:
        visualizer.close()
```

### With Validation

```python
def train_with_validation():
    visualizer = RichTrainerVisualizer()
    visualizer.start(total_steps=1000, description="Training with Validation")

    try:
        for step in range(1000):
            train_loss = training_step()

            # Periodic validation
            val_loss = None
            if step % 50 == 0 and step > 0:
                val_loss = validation_step()

            snapshot = TrainingSnapshot(
                step=step,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=get_lr()
            )

            visualizer.update(snapshot)
    finally:
        visualizer.close()
```

## Factory Function

### create_default_visualizer

```python
def create_default_visualizer(
    show_progress: bool = True,
    show_tables: bool = True,
    table_interval: int = 25,
) -> RichTrainerVisualizer
```

Creates a visualizer with commonly used default settings.

**Parameters:**

- `show_progress`: Whether to show progress bar
- `show_tables`: Whether to show status tables
- `table_interval`: Steps between table updates

**Returns:**

- Configured RichTrainerVisualizer instance

**Example:**

```python
# Quick setup for most use cases
visualizer = create_default_visualizer(table_interval=50)
```

## Error Handling

The visualization system gracefully handles various conditions:

```python
# Handles None values in snapshots
snapshot = TrainingSnapshot(step=1, train_loss=float('nan'))
visualizer.update(snapshot)  # Displays "nan" safely

# Handles missing expert data
snapshot = TrainingSnapshot(
    step=2,
    train_loss=0.5,
    expert_utilization={}  # Empty dict
)
visualizer.update(snapshot)  # Shows appropriate message

# Handles extreme values
snapshot = TrainingSnapshot(step=3, train_loss=float('inf'))
visualizer.update(snapshot)  # Displays as "inf"
```

## Performance Considerations

- Progress bar updates are rate-limited (4 Hz default)
- Tables only show at configured intervals (25 steps default)
- Console rendering is optimized for terminal performance
- Memory usage is minimal with automatic cleanup

## Best Practices

1. **Use try/finally**: Always call `visualizer.close()` in finally block
2. **Appropriate Intervals**: Match table intervals to your training speed
3. **Custom Metrics**: Use `custom_metrics` for domain-specific information
4. **Expert Utilization**: Enable for MoE models to monitor load balancing
5. **Validation Timing**: Update `val_loss` only when validation runs

## Troubleshooting

### Progress Bar Not Updating

- Check that `start()` was called before `update()`
- Verify terminal supports rich formatting

### Tables Not Showing

- Ensure `show_tables=True` in configuration
- Check that step numbers are multiples of `table_update_interval`

### Performance Issues

- Reduce `refresh_per_second` for slower terminals
- Increase `table_update_interval` for less frequent updates
- Disable unused features via configuration

## Combined Visualization: Rich Terminal + Matplotlib Integration

For the most comprehensive training monitoring experience, you can combine `m.rich_trainer_viz` (terminal UI) with `m.training_viz` (matplotlib plotting) to get both real-time progress monitoring AND scientific visualization.

### Why Use Both Together?

- **Rich Terminal**: Real-time progress bars, live metrics tables, immediate feedback
- **Matplotlib Plots**: Historical trend analysis, publication-ready charts, detailed visualizations
- **Together**: Complete monitoring solution covering both live status and analytical insights

### Complete Dual Visualization Setup

```python
from m.rich_trainer_viz import RichTrainerVisualizer, TrainingSnapshot, VisualizationConfig
from m.training_viz import create_moe_visualizer

class ComprehensiveTrainingVisualizer:
    """
    Combines Rich terminal UI with matplotlib plotting for complete visualization coverage.
    Based on the proven pattern from examples/trajectory_3d/training.py
    """

    def __init__(self, total_steps: int, n_experts: int = 8):
        # 1. Matplotlib-based scientific plotting
        self.plot_viz = create_moe_visualizer(
            update_matplotlib=1,  # Update plots every step
            n_experts=n_experts   # For individual expert tracking
        )

        # 2. Rich-based terminal progress monitoring
        config = VisualizationConfig(
            show_progress_bar=True,
            show_tables=True,
            table_update_interval=25,  # Show detailed tables every 25 steps
            show_expert_utilization=True,
            show_throughput=True,
            show_system_metrics=True
        )
        self.terminal_viz = RichTrainerVisualizer(config)

        self.total_steps = total_steps

    def start(self, description: str = "Training MoE Model"):
        """Initialize both visualization systems."""
        self.terminal_viz.start(total_steps=self.total_steps, description=description)

    def update(self, step: int, metrics: dict):
        """Update both visualization systems with training metrics."""

        # Update matplotlib plots with detailed metrics
        plot_metrics = {
            # Core losses
            "train_loss": metrics.get("train_loss"),
            "val_loss": metrics.get("val_loss"),
            "aux_loss": metrics.get("aux_loss"),

            # Task-specific metrics
            "position_error": metrics.get("position_error"),
            "velocity_error": metrics.get("velocity_error"),
            "accuracy": metrics.get("accuracy"),

            # Expert utilization
            "expert_entropy": metrics.get("expert_entropy"),
            "load_balance": metrics.get("load_balance"),

            # Throughput
            "samples_per_sec": metrics.get("samples_per_sec"),
            "tokens_per_sec": metrics.get("tokens_per_sec"),
        }

        # Filter out None values
        plot_metrics = {k: v for k, v in plot_metrics.items() if v is not None}
        self.plot_viz.update_metrics(plot_metrics)

        # Update Rich terminal display with structured snapshot
        snapshot = TrainingSnapshot(
            step=step,
            train_loss=metrics.get("train_loss", 0.0),
            val_loss=metrics.get("val_loss"),
            learning_rate=metrics.get("learning_rate"),
            aux_loss=metrics.get("aux_loss"),
            expert_utilization=metrics.get("expert_utilization", {}),
            batch_size=metrics.get("batch_size"),
            gpu_memory_mb=metrics.get("gpu_memory_mb"),
            gradient_norm=metrics.get("gradient_norm"),
            custom_metrics=metrics.get("custom_metrics", {})
        )
        self.terminal_viz.update(snapshot)

    def save_plots(self, save_path: str):
        """Save current matplotlib plots to file."""
        self.plot_viz.save_plot(save_path)

    def close(self):
        """Clean up both visualization systems."""
        self.plot_viz.close()
        self.terminal_viz.close()
```

### Training Loop Integration

```python
def train_with_comprehensive_visualization():
    """Complete example showing dual visualization in action."""

    # Initialize comprehensive visualizer
    visualizer = ComprehensiveTrainingVisualizer(
        total_steps=1000,
        n_experts=8
    )

    visualizer.start("Training Advanced MoE Model")

    try:
        for step in range(1000):
            # Your training code here
            train_loss = perform_training_step()

            # Periodic validation
            val_loss = None
            if step % 50 == 0 and step > 0:
                val_loss = perform_validation()

            # Collect comprehensive metrics
            metrics = {
                # Core training metrics
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "aux_loss": model.get_aux_loss(),

                # MoE-specific metrics
                "expert_utilization": model.get_expert_utilization(),
                "expert_entropy": model.get_routing_entropy(),
                "load_balance": model.get_load_balance_factor(),

                # Task-specific metrics
                "position_error": calculate_position_error(),
                "velocity_error": calculate_velocity_error(),

                # System metrics
                "samples_per_sec": calculate_throughput(),
                "gpu_memory_mb": get_gpu_memory_usage(),
                "gradient_norm": get_gradient_norm(),

                # Custom metrics
                "custom_metrics": {
                    "routing_confidence": model.get_routing_confidence(),
                    "expert_specialization": model.get_expert_specialization()
                }
            }

            # Update both visualization systems
            visualizer.update(step, metrics)

            # Periodic plot saving
            if step % 200 == 0 and step > 0:
                visualizer.save_plots(f"training_plots_step_{step}.png")

    finally:
        # Ensure cleanup even if training fails
        visualizer.close()
```

### Real-World Production Pattern

Based on `examples/trajectory_3d/training.py`, here's the production-tested approach:

```python
class ProductionTrainer:
    def __init__(self, model_config, training_config):
        self.model = create_model(model_config)
        self.config = training_config

        # Dual visualization setup (production pattern)
        n_experts = model_config.block.moe.router.n_experts

        # Scientific plotting for analysis
        self.real_time_vis = create_moe_visualizer(
            update_matplotlib=1,
            n_experts=n_experts
        )

        # Terminal UI for monitoring
        viz_config = VisualizationConfig(
            show_progress_bar=True,
            show_tables=True,
            table_update_interval=25,
            show_expert_utilization=True,
            show_throughput=True,
            show_system_metrics=True
        )
        self.rich_visualizer = RichTrainerVisualizer(viz_config)

    def train(self):
        total_steps = self.config.training["max_steps"]
        self.rich_visualizer.start(total_steps, "Production MoE Training")

        try:
            for step in range(total_steps):
                # Training step
                metrics = self._train_step()

                # Update visualizations
                self._update_metrics(step, metrics)

                # Checkpoint saving
                if step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(step)

        finally:
            self._cleanup_visualizers()

    def _update_metrics(self, step: int, metrics: dict):
        """Update both visualization systems."""

        # Update matplotlib plots
        plot_data = self._prepare_plot_metrics(metrics)
        self.real_time_vis.update_metrics(plot_data)

        # Update rich terminal
        snapshot = self._create_training_snapshot(step, metrics)
        self.rich_visualizer.update(snapshot)

    def _cleanup_visualizers(self):
        """Clean up both visualization systems."""
        self.real_time_vis.close()
        self.rich_visualizer.close()
```

### Benefits of Combined Approach

1. **Immediate Feedback**: Rich terminal gives instant visual feedback on progress
2. **Historical Analysis**: Matplotlib plots show trends and patterns over time
3. **Publication Ready**: Matplotlib generates publication-quality figures
4. **Debugging Support**: Rich tables help identify issues in real-time
5. **Comprehensive Coverage**: Terminal UI + scientific plots cover all monitoring needs

### Best Practices for Dual Visualization

1. **Coordinate Updates**: Update both systems in the same training loop iteration
2. **Metric Consistency**: Use the same metric names/values for both systems when possible
3. **Performance Balance**: Rich updates can be more frequent (every step), matplotlib less frequent
4. **Resource Management**: Always use try/finally to ensure cleanup of both systems
5. **Selective Metrics**: Send appropriate metrics to each system (Rich: progress data, Matplotlib: trend data)

### Common Integration Patterns

#### Pattern 1: Research Training

```python
# Frequent matplotlib updates for detailed analysis
plot_viz = create_moe_visualizer(update_matplotlib=1)
terminal_viz = RichTrainerVisualizer()  # Default settings
```

#### Pattern 2: Production Training

```python
# Less frequent plotting, more terminal feedback
plot_viz = create_moe_visualizer(update_matplotlib=10)
config = VisualizationConfig(table_update_interval=5)  # Frequent tables
terminal_viz = RichTrainerVisualizer(config)
```

#### Pattern 3: Long Training Runs

```python
# Very infrequent plotting, moderate terminal updates
plot_viz = create_moe_visualizer(update_matplotlib=100)
config = VisualizationConfig(table_update_interval=50)
terminal_viz = RichTrainerVisualizer(config)
```

---

The `m.rich_trainer_viz` module provides a clean, well-tested interface for beautiful training visualization that enhances the development experience without impacting training performance. When combined with `m.training_viz`, it creates a complete monitoring solution that serves both immediate feedback needs and long-term analytical requirements.
