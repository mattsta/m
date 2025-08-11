"""
Tests for Real-time Training Visualization Module

Comprehensive testing of the m.training_viz module including configuration,
metrics tracking, plot generation, and factory functions.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from m.training_viz import (
    MetricConfig,
    PlotConfig,
    RealTimeTrainingVisualizer,
    TrainingVisualizerConfig,
    create_custom_visualizer,
    create_loss_visualizer,
    create_moe_visualizer,
)


class TestPlotConfig:
    """Test plot configuration dataclass."""

    def test_plot_config_creation(self):
        """Test basic plot config creation."""
        config = PlotConfig(title="Test Plot", ylabel="Test Y")

        assert config.title == "Test Plot"
        assert config.ylabel == "Test Y"
        assert config.xlabel == "Step"  # Default
        assert config.yscale == "linear"  # Default
        assert config.show_grid is True  # Default
        assert config.show_legend is True  # Default
        assert len(config.color_cycle) > 0  # Default colors

    def test_plot_config_custom_values(self):
        """Test plot config with custom values."""
        custom_colors = ["red", "blue", "green"]
        config = PlotConfig(
            title="Custom Plot",
            ylabel="Custom Y",
            xlabel="Custom X",
            yscale="log",
            show_grid=False,
            show_legend=False,
            color_cycle=custom_colors,
        )

        assert config.title == "Custom Plot"
        assert config.ylabel == "Custom Y"
        assert config.xlabel == "Custom X"
        assert config.yscale == "log"
        assert config.show_grid is False
        assert config.show_legend is False
        assert config.color_cycle == custom_colors


class TestMetricConfig:
    """Test metric configuration dataclass."""

    def test_metric_config_creation(self):
        """Test basic metric config creation."""
        config = MetricConfig(metric_name="loss", plot_index=0)

        assert config.metric_name == "loss"
        assert config.plot_index == 0
        assert config.color is None  # Default
        assert config.label is None  # Default
        assert config.line_style == "-"  # Default
        assert config.alpha == 1.0  # Default

    def test_metric_config_custom_values(self):
        """Test metric config with custom values."""
        config = MetricConfig(
            metric_name="accuracy",
            plot_index=1,
            color="red",
            label="Test Accuracy",
            line_style="--",
            alpha=0.7,
        )

        assert config.metric_name == "accuracy"
        assert config.plot_index == 1
        assert config.color == "red"
        assert config.label == "Test Accuracy"
        assert config.line_style == "--"
        assert config.alpha == 0.7


class TestTrainingVisualizerConfig:
    """Test training visualizer configuration."""

    def test_config_defaults(self):
        """Test configuration with default values."""
        config = TrainingVisualizerConfig()

        assert config.subplot_rows == 2
        assert config.subplot_cols == 2
        assert config.figure_size == (12, 8)
        assert config.title == "Real-time Training Progress"
        assert config.update_interval == 100
        assert config.max_points is None
        assert len(config.plots) == 0
        assert len(config.metrics) == 0
        assert config.use_blit is False
        assert config.pause_duration == 0.01

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        plots = [PlotConfig("Plot 1", "Y1"), PlotConfig("Plot 2", "Y2")]
        metrics = [MetricConfig("loss", 0), MetricConfig("acc", 1)]

        config = TrainingVisualizerConfig(
            subplot_rows=1,
            subplot_cols=2,
            figure_size=(10, 6),
            title="Custom Training",
            update_interval=50,
            max_points=1000,
            plots=plots,
            metrics=metrics,
            use_blit=True,
            pause_duration=0.05,
        )

        assert config.subplot_rows == 1
        assert config.subplot_cols == 2
        assert config.figure_size == (10, 6)
        assert config.title == "Custom Training"
        assert config.update_interval == 50
        assert config.max_points == 1000
        assert len(config.plots) == 2
        assert len(config.metrics) == 2
        assert config.use_blit is True
        assert config.pause_duration == 0.05


class TestRealTimeTrainingVisualizer:
    """Test the main visualizer class."""

    def test_visualizer_creation(self):
        """Test basic visualizer creation."""
        config = TrainingVisualizerConfig()
        visualizer = RealTimeTrainingVisualizer(config)

        assert visualizer.config == config
        assert len(visualizer.metrics_history) == 0
        assert visualizer.step_count == 0
        assert visualizer.fig is None
        assert visualizer.axes is None

    def test_metrics_update(self):
        """Test updating metrics."""
        config = TrainingVisualizerConfig(update_interval=1000)  # Don't trigger plots
        visualizer = RealTimeTrainingVisualizer(config)

        # Update metrics
        metrics = {"loss": 0.5, "accuracy": 0.8}
        visualizer.update_metrics(metrics)

        assert visualizer.step_count == 1
        assert len(visualizer.metrics_history) == 2
        assert visualizer.metrics_history["loss"] == [0.5]
        assert visualizer.metrics_history["accuracy"] == [0.8]

    def test_multiple_metrics_updates(self):
        """Test multiple metric updates."""
        config = TrainingVisualizerConfig(update_interval=1000)  # Don't trigger plots
        visualizer = RealTimeTrainingVisualizer(config)

        # Multiple updates
        for i in range(5):
            metrics = {"loss": 1.0 - i * 0.1, "accuracy": i * 0.2}
            visualizer.update_metrics(metrics)

        assert visualizer.step_count == 5
        assert len(visualizer.metrics_history["loss"]) == 5
        assert len(visualizer.metrics_history["accuracy"]) == 5
        expected_loss = [1.0, 0.9, 0.8, 0.7, 0.6]
        expected_accuracy = [0.0, 0.2, 0.4, 0.6, 0.8]

        assert len(visualizer.metrics_history["loss"]) == 5
        assert len(visualizer.metrics_history["accuracy"]) == 5

        for actual, expected in zip(visualizer.metrics_history["loss"], expected_loss):
            assert abs(actual - expected) < 1e-10

        for actual, expected in zip(
            visualizer.metrics_history["accuracy"], expected_accuracy
        ):
            assert abs(actual - expected) < 1e-10

    def test_max_points_limiting(self):
        """Test that max_points limits history size."""
        config = TrainingVisualizerConfig(max_points=3, update_interval=1000)
        visualizer = RealTimeTrainingVisualizer(config)

        # Add more points than max_points
        for i in range(10):
            visualizer.update_metrics({"loss": float(i)})

        # Should only keep last 3 points
        assert len(visualizer.metrics_history["loss"]) == 3
        assert visualizer.metrics_history["loss"] == [7.0, 8.0, 9.0]

    def test_metrics_summary(self):
        """Test metrics summary generation."""
        config = TrainingVisualizerConfig(update_interval=1000)
        visualizer = RealTimeTrainingVisualizer(config)

        # Add some metrics
        for i in range(10):
            visualizer.update_metrics({"loss": 1.0 - i * 0.1})

        summary = visualizer.get_metrics_summary()

        assert "loss" in summary
        loss_summary = summary["loss"]
        assert abs(loss_summary["current"] - 0.1) < 1e-10  # Last value
        assert abs(loss_summary["min"] - 0.1) < 1e-10
        assert loss_summary["max"] == 1.0
        assert loss_summary["count"] == 10
        assert abs(loss_summary["mean"] - 0.55) < 1e-10  # Average of 1.0 to 0.1

    def test_save_plot(self):
        """Test saving plot to file."""
        config = TrainingVisualizerConfig()
        visualizer = RealTimeTrainingVisualizer(config)

        # Add some data and force plot creation
        visualizer.update_metrics({"loss": 0.5})
        visualizer.force_update()  # Creates the plot

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Save plot
            visualizer.save_plot(tmp_path)
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0  # File is not empty
        finally:
            visualizer.close()
            if tmp_path.exists():
                tmp_path.unlink()

    def test_config_validation_too_many_plots(self):
        """Test validation catches too many plots."""
        plots = [PlotConfig("Plot 1", "Y1"), PlotConfig("Plot 2", "Y2")]
        config = TrainingVisualizerConfig(
            subplot_rows=1,
            subplot_cols=1,  # Only 1 subplot available
            plots=plots,  # But 2 plots configured
        )

        with pytest.raises(ValueError, match="Too many plots configured"):
            RealTimeTrainingVisualizer(config)

    def test_config_validation_invalid_plot_index(self):
        """Test validation catches invalid plot indices."""
        metrics = [MetricConfig("loss", 0), MetricConfig("acc", 5)]  # Index 5 invalid
        config = TrainingVisualizerConfig(
            subplot_rows=2,
            subplot_cols=2,  # Only indices 0-3 valid
            metrics=metrics,
        )

        with pytest.raises(ValueError, match="references invalid plot_index"):
            RealTimeTrainingVisualizer(config)

    def test_close(self):
        """Test proper cleanup."""
        config = TrainingVisualizerConfig()
        visualizer = RealTimeTrainingVisualizer(config)

        # Add some data to create plots
        visualizer.update_metrics({"loss": 0.5})
        visualizer.force_update()

        # Close should work without errors
        visualizer.close()


class TestVisualizerFactories:
    """Test pre-configured visualizer factory functions."""

    def test_create_loss_visualizer(self):
        """Test loss visualizer factory."""
        visualizer = create_loss_visualizer(update_interval=50)

        assert visualizer.config.update_interval == 50
        assert visualizer.config.subplot_rows == 1
        assert visualizer.config.subplot_cols == 2
        assert len(visualizer.config.plots) == 2
        assert (
            len(visualizer.config.metrics) == 3
        )  # train_loss, val_loss, learning_rate

        # Check metric names
        metric_names = {m.metric_name for m in visualizer.config.metrics}
        assert "train_loss" in metric_names
        assert "val_loss" in metric_names
        assert "learning_rate" in metric_names

        visualizer.close()

    def test_create_moe_visualizer(self):
        """Test MoE visualizer factory."""
        visualizer = create_moe_visualizer(update_matplotlib=200)

        assert visualizer.config.update_interval == 200
        assert visualizer.config.subplot_rows == 2
        assert visualizer.config.subplot_cols == 2
        assert len(visualizer.config.plots) == 4
        assert len(visualizer.config.metrics) > 5  # Multiple MoE-specific metrics

        # Check some key metric names
        metric_names = {m.metric_name for m in visualizer.config.metrics}
        assert "train_loss" in metric_names
        assert "val_loss" in metric_names
        assert "aux_loss" in metric_names
        assert "position_error" in metric_names

        visualizer.close()

    def test_create_custom_visualizer(self):
        """Test custom visualizer factory."""
        metrics_config = {
            "loss": (0, "red"),
            "accuracy": (0, "blue"),
            "lr": (1, "green"),
        }
        plot_titles = ["Training Metrics", "Learning Rate"]

        visualizer = create_custom_visualizer(
            metrics_config=metrics_config,
            plot_titles=plot_titles,
            subplot_layout=(1, 2),
            update_interval=75,
            title="Custom Test Visualization",
        )

        assert visualizer.config.update_interval == 75
        assert visualizer.config.subplot_rows == 1
        assert visualizer.config.subplot_cols == 2
        assert visualizer.config.title == "Custom Test Visualization"
        assert len(visualizer.config.plots) == 2
        assert len(visualizer.config.metrics) == 3

        # Verify metrics are configured correctly
        loss_metric = next(
            m for m in visualizer.config.metrics if m.metric_name == "loss"
        )
        assert loss_metric.plot_index == 0
        assert loss_metric.color == "red"

        accuracy_metric = next(
            m for m in visualizer.config.metrics if m.metric_name == "accuracy"
        )
        assert accuracy_metric.plot_index == 0
        assert accuracy_metric.color == "blue"

        lr_metric = next(m for m in visualizer.config.metrics if m.metric_name == "lr")
        assert lr_metric.plot_index == 1
        assert lr_metric.color == "green"

        visualizer.close()


class TestVisualizerIntegration:
    """Integration tests with realistic usage patterns."""

    def test_realistic_training_simulation(self):
        """Test with realistic training data patterns."""
        visualizer = create_loss_visualizer(update_interval=10)

        # Simulate training loop
        initial_loss = 2.5
        for step in range(50):
            # Simulate decreasing loss with noise
            train_loss = initial_loss * (0.95**step) + np.random.normal(0, 0.01)
            val_loss = train_loss * 1.1 + np.random.normal(0, 0.02)
            lr = 0.001 * (0.99**step)

            metrics = {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": float(lr),
            }
            visualizer.update_metrics(metrics)

        # Check that metrics were captured
        assert len(visualizer.metrics_history["train_loss"]) == 50
        assert len(visualizer.metrics_history["val_loss"]) == 50
        assert len(visualizer.metrics_history["learning_rate"]) == 50

        # Check that loss generally decreased
        first_loss = visualizer.metrics_history["train_loss"][0]
        last_loss = visualizer.metrics_history["train_loss"][-1]
        assert last_loss < first_loss  # Should be decreasing

        # Check learning rate decreased
        first_lr = visualizer.metrics_history["learning_rate"][0]
        last_lr = visualizer.metrics_history["learning_rate"][-1]
        assert last_lr < first_lr

        # Get summary
        summary = visualizer.get_metrics_summary()
        assert "train_loss" in summary
        assert "val_loss" in summary
        assert "learning_rate" in summary

        visualizer.close()

    def test_moe_training_simulation(self):
        """Test MoE visualizer with realistic MoE metrics."""
        visualizer = create_moe_visualizer(update_matplotlib=5)

        for step in range(25):
            # Simulate MoE training metrics
            metrics = {
                "train_loss": 1.0 - step * 0.03,
                "val_loss": 1.1 - step * 0.025,
                "aux_loss": 0.1 * np.exp(-step * 0.1),
                "position_error": 2.0 - step * 0.07,
                "expert_entropy": 2.0 + np.sin(step * 0.3) * 0.5,
                "load_balance": 1.0 - np.abs(np.sin(step * 0.2)) * 0.3,
                "samples_per_sec": 100 + np.random.normal(0, 10),
            }
            visualizer.update_metrics(metrics)

        # Verify all metrics were captured
        for metric_name in metrics.keys():
            assert metric_name in visualizer.metrics_history
            assert len(visualizer.metrics_history[metric_name]) == 25

        visualizer.close()
