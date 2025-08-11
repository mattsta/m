"""
Training Visualization Examples

Demonstrates how to use the m.training_viz module for real-time training visualization.
Shows basic usage, factory functions, and custom configurations.
"""

import os
import time

import numpy as np

from m.training_viz import (
    MetricConfig,
    PlotConfig,
    RealTimeTrainingVisualizer,
    TrainingVisualizerConfig,
    create_custom_visualizer,
    create_loss_visualizer,
    create_moe_visualizer,
)


def example_basic_loss_visualization():
    """Example 1: Basic loss visualization using factory function."""
    print("🎯 Example 1: Basic Loss Visualization")

    # Create a simple loss visualizer
    visualizer = create_loss_visualizer(update_interval=10)

    try:
        # Simulate training loop
        print("   Simulating training for 100 steps...")
        for step in range(100):
            # Simulate decreasing loss with some noise
            train_loss = 2.0 * (0.95**step) + np.random.normal(0, 0.05)
            val_loss = train_loss * 1.15 + np.random.normal(0, 0.03)
            learning_rate = 0.001 * (0.99**step)

            metrics = {
                "train_loss": max(0.01, train_loss),  # Clamp to positive
                "val_loss": max(0.01, val_loss),
                "learning_rate": learning_rate,
            }

            visualizer.update_metrics(metrics)

            # Print progress occasionally
            if step % 20 == 0:
                print(
                    f"     Step {step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            # Small delay for demonstration
            time.sleep(0.02)

        # Get summary
        summary = visualizer.get_metrics_summary()
        print(f"   Final summary: Train loss {summary['train_loss']['current']:.4f}")

        # Save plot
        visualizer.save_plot("outputs/basic_loss_example.png")
        print("   📊 Plot saved to outputs/basic_loss_example.png")

    finally:
        visualizer.close()

    print("✅ Basic loss visualization completed!\n")


def example_moe_visualization():
    """Example 2: MoE-specific visualization with expert metrics."""
    print("🔥 Example 2: MoE Training Visualization")

    # Create MoE-specific visualizer
    visualizer = create_moe_visualizer(update_interval=5)

    try:
        print("   Simulating MoE training for 50 steps...")
        for step in range(50):
            # Simulate MoE training metrics
            train_loss = 1.5 * (0.96**step) + np.random.normal(0, 0.02)
            val_loss = train_loss * 1.1 + np.random.normal(0, 0.01)
            aux_loss = 0.1 * np.exp(-step * 0.05) + np.random.normal(0, 0.005)

            # Task-specific metrics (could be position error for trajectory, accuracy for classification, etc.)
            position_error = 2.0 * (0.92**step) + np.random.normal(0, 0.1)
            velocity_error = 1.0 * (0.94**step) + np.random.normal(0, 0.05)

            # Expert utilization metrics
            expert_entropy = 2.0 + np.sin(step * 0.1) * 0.5  # Oscillating entropy
            load_balance = (
                1.0 - np.abs(np.sin(step * 0.05)) * 0.3
            )  # Load balance factor

            # Performance metrics
            samples_per_sec = 120 + np.random.normal(0, 10)

            metrics = {
                "train_loss": max(0.001, train_loss),
                "val_loss": max(0.001, val_loss),
                "aux_loss": max(0.0, aux_loss),
                "position_error": max(0.0, position_error),
                "velocity_error": max(0.0, velocity_error),
                "expert_entropy": expert_entropy,
                "load_balance": load_balance,
                "samples_per_sec": max(0, samples_per_sec),
            }

            visualizer.update_metrics(metrics)

            if step % 10 == 0:
                print(
                    f"     Step {step}: loss={train_loss:.4f}, pos_err={position_error:.4f}, entropy={expert_entropy:.3f}"
                )

            time.sleep(0.03)

        # Save plot
        visualizer.save_plot("outputs/moe_training_example.png")
        print("   📊 Plot saved to outputs/moe_training_example.png")

    finally:
        visualizer.close()

    print("✅ MoE visualization completed!\n")


def example_custom_visualization():
    """Example 3: Custom visualization with specific layout and metrics."""
    print("🎨 Example 3: Custom Visualization Layout")

    # Define custom metrics configuration
    metrics_config = {
        # Loss metrics on first plot
        "train_loss": (0, "blue"),
        "val_loss": (0, "red"),
        "aux_loss": (0, "orange"),
        # Accuracy metrics on second plot
        "train_accuracy": (1, "green"),
        "val_accuracy": (1, "darkgreen"),
        # Learning rate on third plot
        "learning_rate": (2, "purple"),
        "momentum": (2, "brown"),
        # Custom metrics on fourth plot
        "custom_metric_1": (3, "cyan"),
        "custom_metric_2": (3, "magenta"),
    }

    plot_titles = [
        "Loss Curves",
        "Accuracy Metrics",
        "Optimization Parameters",
        "Custom Task Metrics",
    ]

    # Create custom visualizer
    visualizer = create_custom_visualizer(
        metrics_config=metrics_config,
        plot_titles=plot_titles,
        subplot_layout=(2, 2),
        update_interval=8,
        title="Custom Multi-Task Training Visualization",
    )

    try:
        print("   Simulating custom training for 80 steps...")
        for step in range(80):
            # Simulate various training metrics
            train_loss = 1.0 * (0.97**step) + np.random.normal(0, 0.02)
            val_loss = train_loss * 1.05 + np.random.normal(0, 0.01)
            aux_loss = 0.05 * (0.99**step) + np.random.normal(0, 0.002)

            train_acc = min(1.0, 0.5 + step * 0.006 + np.random.normal(0, 0.02))
            val_acc = train_acc - 0.02 + np.random.normal(0, 0.015)

            lr = 0.001 * (0.995**step)
            momentum = 0.9 + np.sin(step * 0.1) * 0.05

            custom_1 = np.sin(step * 0.2) + np.random.normal(0, 0.1)
            custom_2 = np.exp(-step * 0.02) + np.random.normal(0, 0.05)

            metrics = {
                "train_loss": max(0.01, train_loss),
                "val_loss": max(0.01, val_loss),
                "aux_loss": max(0.0, aux_loss),
                "train_accuracy": np.clip(train_acc, 0, 1),
                "val_accuracy": np.clip(val_acc, 0, 1),
                "learning_rate": lr,
                "momentum": momentum,
                "custom_metric_1": custom_1,
                "custom_metric_2": max(0, custom_2),
            }

            visualizer.update_metrics(metrics)

            if step % 15 == 0:
                print(
                    f"     Step {step}: loss={train_loss:.4f}, acc={train_acc:.3f}, lr={lr:.2e}"
                )

            time.sleep(0.025)

        # Save plot
        visualizer.save_plot("outputs/custom_visualization_example.png")
        print("   📊 Plot saved to outputs/custom_visualization_example.png")

    finally:
        visualizer.close()

    print("✅ Custom visualization completed!\n")


def example_advanced_configuration():
    """Example 4: Advanced configuration with detailed customization."""
    print("⚙️ Example 4: Advanced Configuration")

    # Create highly customized configuration
    config = TrainingVisualizerConfig(
        subplot_rows=3,
        subplot_cols=2,
        figure_size=(15, 12),
        title="Advanced Multi-Model Training Dashboard",
        update_interval=5,
        max_points=200,  # Limit history to last 200 points
        # Define plots with custom settings
        plots=[
            PlotConfig(
                title="Primary Loss", ylabel="Loss", yscale="log", show_grid=True
            ),
            PlotConfig(title="Secondary Metrics", ylabel="Score", yscale="linear"),
            PlotConfig(title="Gradient Norms", ylabel="Norm", yscale="log"),
            PlotConfig(title="Learning Dynamics", ylabel="Rate", yscale="linear"),
            PlotConfig(title="Memory Usage", ylabel="GB", yscale="linear"),
            PlotConfig(title="Throughput", ylabel="Samples/sec", yscale="linear"),
        ],
        # Define metrics with precise styling
        metrics=[
            # Primary loss plot (index 0)
            MetricConfig(
                "main_loss",
                0,
                color="darkblue",
                label="Main Loss",
                line_style="-",
                alpha=0.8,
            ),
            MetricConfig(
                "regularization_loss",
                0,
                color="lightblue",
                label="Regularization",
                line_style="--",
                alpha=0.6,
            ),
            # Secondary metrics plot (index 1)
            MetricConfig(
                "precision", 1, color="green", label="Precision", line_style="-"
            ),
            MetricConfig("recall", 1, color="orange", label="Recall", line_style="-"),
            MetricConfig(
                "f1_score", 1, color="red", label="F1 Score", line_style="-", alpha=0.8
            ),
            # Gradient norms plot (index 2)
            MetricConfig(
                "grad_norm", 2, color="purple", label="Gradient Norm", line_style="-"
            ),
            MetricConfig(
                "param_norm", 2, color="brown", label="Parameter Norm", line_style="--"
            ),
            # Learning dynamics plot (index 3)
            MetricConfig(
                "learning_rate", 3, color="black", label="Learning Rate", line_style="-"
            ),
            MetricConfig(
                "weight_decay",
                3,
                color="gray",
                label="Weight Decay",
                line_style=":",
                alpha=0.7,
            ),
            # Memory usage plot (index 4)
            MetricConfig(
                "gpu_memory", 4, color="red", label="GPU Memory", line_style="-"
            ),
            MetricConfig(
                "cpu_memory", 4, color="blue", label="CPU Memory", line_style="--"
            ),
            # Throughput plot (index 5)
            MetricConfig(
                "samples_per_sec", 5, color="cyan", label="Samples/sec", line_style="-"
            ),
            MetricConfig(
                "batches_per_sec",
                5,
                color="magenta",
                label="Batches/sec",
                line_style="--",
            ),
        ],
        use_blit=False,  # Disable blitting for compatibility
        pause_duration=0.01,
    )

    # Create visualizer with advanced config
    visualizer = RealTimeTrainingVisualizer(config)

    try:
        print("   Simulating advanced training scenario for 60 steps...")
        for step in range(60):
            # Simulate comprehensive metrics
            metrics = {
                # Losses
                "main_loss": 2.0 * (0.95**step) + np.random.normal(0, 0.05),
                "regularization_loss": 0.1 * (0.98**step) + np.random.normal(0, 0.01),
                # Classification metrics
                "precision": min(1.0, 0.6 + step * 0.005 + np.random.normal(0, 0.02)),
                "recall": min(1.0, 0.55 + step * 0.006 + np.random.normal(0, 0.025)),
                "f1_score": min(1.0, 0.5 + step * 0.007 + np.random.normal(0, 0.02)),
                # Gradient information
                "grad_norm": 1.0 * (0.93**step) + np.random.normal(0, 0.1),
                "param_norm": 10.0 + np.sin(step * 0.1) * 2.0,
                # Optimization
                "learning_rate": 0.001 * (0.99**step),
                "weight_decay": 0.0001,
                # System metrics
                "gpu_memory": 8.0
                + np.sin(step * 0.05) * 1.5
                + np.random.normal(0, 0.2),
                "cpu_memory": 16.0 + np.random.normal(0, 0.5),
                # Performance
                "samples_per_sec": 150 + np.random.normal(0, 15),
                "batches_per_sec": 5 + np.random.normal(0, 0.5),
            }

            # Ensure all values are positive where needed
            for key in [
                "main_loss",
                "regularization_loss",
                "grad_norm",
                "param_norm",
                "gpu_memory",
                "cpu_memory",
            ]:
                metrics[key] = max(0.001, metrics[key])

            for key in ["precision", "recall", "f1_score"]:
                metrics[key] = np.clip(metrics[key], 0, 1)

            visualizer.update_metrics(metrics)

            if step % 10 == 0:
                print(
                    f"     Step {step}: main_loss={metrics['main_loss']:.4f}, f1={metrics['f1_score']:.3f}"
                )

            time.sleep(0.05)

        # Final summary
        summary = visualizer.get_metrics_summary()
        print(f"   Final F1 Score: {summary['f1_score']['current']:.3f}")
        print(f"   GPU Memory Peak: {summary['gpu_memory']['max']:.1f} GB")

        # Save plot
        visualizer.save_plot("outputs/advanced_configuration_example.png")
        print("   📊 Plot saved to outputs/advanced_configuration_example.png")

    finally:
        visualizer.close()

    print("✅ Advanced configuration completed!\n")


def main():
    """Run all training visualization examples."""
    print("🎨 Training Visualization Examples\n")
    print(
        "This will demonstrate the m.training_viz module with various configurations."
    )
    print(
        "Each example will create a real-time visualization and save plots to outputs/\n"
    )

    # Create outputs directory
    # os already imported at top

    os.makedirs("outputs", exist_ok=True)

    try:
        # Run examples
        example_basic_loss_visualization()
        example_moe_visualization()
        example_custom_visualization()
        example_advanced_configuration()

        print("🎯 All examples completed successfully!")
        print("📊 Check the outputs/ directory for saved visualizations.")

    except KeyboardInterrupt:
        print("\n⚠️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    main()
