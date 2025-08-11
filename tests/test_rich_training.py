"""
Tests for rich training visualization components.

Tests the actual current implementation of the rich trainer visualizer system.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from m.rich_trainer_viz import (
    MinimalTrainerVisualizer,
    RichTrainerVisualizer,
    SystemMetrics,
    ThroughputMetrics,
    TrainingSnapshot,
    VisualizationConfig,
    create_default_visualizer,
    snapshot_from_trainer_state,
)


class TestThroughputMetrics:
    """Test the ThroughputMetrics dataclass."""

    def test_default_creation(self):
        """Test creating ThroughputMetrics with default values."""
        metrics = ThroughputMetrics()

        # Check all fields exist and have reasonable defaults
        assert hasattr(metrics, "samples_per_second")
        assert hasattr(metrics, "tokens_per_second")
        assert hasattr(metrics, "steps_per_second")
        assert hasattr(metrics, "batches_per_second")

        # Should be floats or None
        assert metrics.samples_per_second is None or isinstance(
            metrics.samples_per_second, float
        )
        assert metrics.tokens_per_second is None or isinstance(
            metrics.tokens_per_second, float
        )
        assert metrics.steps_per_second is None or isinstance(
            metrics.steps_per_second, float
        )
        assert metrics.batches_per_second is None or isinstance(
            metrics.batches_per_second, float
        )

    def test_with_values(self):
        """Test creating ThroughputMetrics with specific values."""
        metrics = ThroughputMetrics(
            samples_per_second=100.5,
            tokens_per_second=1500.0,
            steps_per_second=10.0,
            batches_per_second=5.0,
        )

        assert metrics.samples_per_second == 100.5
        assert metrics.tokens_per_second == 1500.0
        assert metrics.steps_per_second == 10.0
        assert metrics.batches_per_second == 5.0


class TestSystemMetrics:
    """Test the SystemMetrics dataclass."""

    def test_default_creation(self):
        """Test creating SystemMetrics with default values."""
        metrics = SystemMetrics()

        # Check all expected fields exist
        assert hasattr(metrics, "gpu_memory_used_mb")
        assert hasattr(metrics, "gpu_memory_total_mb")
        assert hasattr(metrics, "gpu_utilization_percent")
        assert hasattr(metrics, "gradient_norm")

        # Should be floats or None
        assert metrics.gpu_memory_used_mb is None or isinstance(
            metrics.gpu_memory_used_mb, float
        )
        assert metrics.gpu_memory_total_mb is None or isinstance(
            metrics.gpu_memory_total_mb, float
        )
        assert metrics.gpu_utilization_percent is None or isinstance(
            metrics.gpu_utilization_percent, float
        )
        assert metrics.gradient_norm is None or isinstance(metrics.gradient_norm, float)

    def test_with_values(self):
        """Test creating SystemMetrics with specific values."""
        metrics = SystemMetrics(
            gpu_memory_used_mb=1024.5,
            gpu_memory_total_mb=8192.0,
            gpu_utilization_percent=75.2,
            gradient_norm=2.34,
        )

        assert metrics.gpu_memory_used_mb == 1024.5
        assert metrics.gpu_memory_total_mb == 8192.0
        assert metrics.gpu_utilization_percent == 75.2
        assert metrics.gradient_norm == 2.34


class TestTrainingSnapshot:
    """Test the TrainingSnapshot dataclass."""

    def test_minimal_creation(self):
        """Test creating TrainingSnapshot with minimal required fields."""
        snapshot = TrainingSnapshot(step=100, train_loss=0.5)

        assert snapshot.step == 100
        assert snapshot.train_loss == 0.5
        assert snapshot.val_loss is None
        assert snapshot.learning_rate is None
        assert snapshot.aux_loss is None
        assert isinstance(snapshot.expert_utilization, dict)
        assert len(snapshot.expert_utilization) == 0

    def test_full_creation(self):
        """Test creating TrainingSnapshot with all fields."""
        expert_util = {0: 0.25, 1: 0.35, 2: 0.40}

        snapshot = TrainingSnapshot(
            step=500,
            train_loss=0.123,
            val_loss=0.145,
            learning_rate=0.001,
            aux_loss=0.05,
            expert_utilization=expert_util,
            batch_size=32,
            sequence_length=128,
            gpu_memory_mb=2048.0,
        )

        assert snapshot.step == 500
        assert snapshot.train_loss == 0.123
        assert snapshot.val_loss == 0.145
        assert snapshot.learning_rate == 0.001
        assert snapshot.aux_loss == 0.05
        assert snapshot.expert_utilization == expert_util
        assert snapshot.batch_size == 32
        assert snapshot.sequence_length == 128
        assert snapshot.gpu_memory_mb == 2048.0

    def test_frozen_dataclass(self):
        """Test that TrainingSnapshot is frozen (immutable)."""
        snapshot = TrainingSnapshot(step=100, train_loss=0.5)

        # Should raise an error when trying to modify
        with pytest.raises(Exception):  # FrozenInstanceError
            snapshot.step = 200


class TestVisualizationConfig:
    """Test the VisualizationConfig dataclass."""

    def test_default_config(self):
        """Test creating VisualizationConfig with defaults."""
        config = VisualizationConfig()

        # Check essential fields exist
        assert hasattr(config, "show_progress_bar")
        assert hasattr(config, "show_throughput")
        assert hasattr(config, "show_system_metrics")
        assert hasattr(config, "show_expert_utilization")

        # Should have reasonable defaults
        assert isinstance(config.show_progress_bar, bool)
        assert isinstance(config.show_throughput, bool)
        assert isinstance(config.show_system_metrics, bool)
        assert isinstance(config.show_expert_utilization, bool)

    def test_custom_config(self):
        """Test creating VisualizationConfig with custom values."""
        config = VisualizationConfig(
            show_progress_bar=False,
            show_throughput=False,
            show_system_metrics=True,
            show_expert_utilization=True,
        )

        assert config.show_progress_bar is False
        assert config.show_throughput is False
        assert config.show_system_metrics is True
        assert config.show_expert_utilization is True


class TestRichTrainerVisualizer:
    """Test the RichTrainerVisualizer class."""

    @patch("m.rich_trainer_viz.Console")
    @patch("m.rich_trainer_viz.Progress")
    def test_initialization(self, mock_progress, mock_console):
        """Test RichTrainerVisualizer initialization."""
        config = VisualizationConfig()
        visualizer = RichTrainerVisualizer(config)

        assert visualizer.config == config
        assert hasattr(visualizer, "console")
        assert hasattr(visualizer, "progress")

    @patch("m.rich_trainer_viz.Console")
    @patch("m.rich_trainer_viz.Progress")
    def test_update_with_snapshot(self, mock_progress, mock_console):
        """Test updating visualizer with training snapshot."""
        config = VisualizationConfig()
        visualizer = RichTrainerVisualizer(config)

        snapshot = TrainingSnapshot(step=100, train_loss=0.5, val_loss=0.6)

        # Should not raise an error
        try:
            visualizer.update(snapshot)
        except Exception as e:
            pytest.fail(f"update() raised an exception: {e}")


class TestMinimalTrainerVisualizer:
    """Test the MinimalTrainerVisualizer class."""

    @patch("m.rich_trainer_viz.Console")
    def test_initialization(self, mock_console):
        """Test MinimalTrainerVisualizer initialization."""
        visualizer = MinimalTrainerVisualizer()

        assert hasattr(visualizer, "console")

    @patch("m.rich_trainer_viz.Console")
    def test_update_with_step_and_loss(self, mock_console):
        """Test updating minimal visualizer with step and loss."""
        visualizer = MinimalTrainerVisualizer()

        # Should not raise an error when updating with step and loss
        try:
            visualizer.update(step=100, loss=0.5)
        except Exception as e:
            pytest.fail(f"update() raised an exception: {e}")


class TestUtilityFunctions:
    """Test utility functions."""

    @patch("m.rich_trainer_viz.Console")
    @patch("m.rich_trainer_viz.Progress")
    def test_create_default_visualizer(self, mock_progress, mock_console):
        """Test create_default_visualizer factory function."""
        visualizer = create_default_visualizer()

        assert isinstance(visualizer, RichTrainerVisualizer)
        assert isinstance(visualizer.config, VisualizationConfig)

    def test_snapshot_from_trainer_state(self):
        """Test snapshot_from_trainer_state utility function."""
        # Call with required positional arguments
        snapshot = snapshot_from_trainer_state(
            step=150,
            epoch=10,
            train_loss=0.25,
            val_loss=0.30,
            learning_rate=0.002,
            expert_stats={0: 0.4, 1: 0.6},
        )

        assert isinstance(snapshot, TrainingSnapshot)
        assert snapshot.step == 150
        assert snapshot.train_loss == 0.25
        assert snapshot.val_loss == 0.30
        assert snapshot.learning_rate == 0.002
        assert snapshot.expert_utilization == {0: 0.4, 1: 0.6}


class TestIntegration:
    """Integration tests for the rich training system."""

    @patch("m.rich_trainer_viz.Console")
    @patch("m.rich_trainer_viz.Progress")
    def test_complete_workflow(self, mock_progress, mock_console):
        """Test complete training visualization workflow."""
        # Create visualizer
        visualizer = create_default_visualizer()

        # Simulate training updates
        for step in range(1, 11):
            snapshot = TrainingSnapshot(
                step=step * 10,
                train_loss=1.0 / step,  # Decreasing loss
                val_loss=1.0 / step + 0.1,
                learning_rate=0.001,
                expert_utilization={0: 0.3, 1: 0.4, 2: 0.3},
            )

            # Should handle updates without errors
            try:
                visualizer.update(snapshot)
            except Exception as e:
                pytest.fail(f"Workflow failed at step {step}: {e}")

        # Test final summary
        try:
            visualizer.print_final_summary()
        except Exception as e:
            pytest.fail(f"Final summary failed: {e}")

    @patch("m.rich_trainer_viz.Console")
    def test_minimal_workflow(self, mock_console):
        """Test minimal visualizer workflow."""
        visualizer = MinimalTrainerVisualizer()

        # Test basic updates
        for step in range(1, 6):
            try:
                visualizer.update(step=step * 50, loss=0.5 / step)
            except Exception as e:
                pytest.fail(f"Minimal workflow failed at step {step}: {e}")
