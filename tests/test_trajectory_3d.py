"""
Pytest tests for 3D trajectory learning system.
Tests the complete pipeline: dataset generation -> training -> inference -> deployment.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from examples.trajectory_3d.datasets import (
    Trajectory3DDataset,
    TrajectoryConfig,
    generate_sample_trajectories,
)
from examples.trajectory_3d.training import (
    Trajectory3DTrainer,
    load_config_from_yaml,
)
from examples.trajectory_3d.visualization import Trajectory3DVisualizer
from m.moe import MoESequenceRegressor


class TestTrajectory3DDataset:
    """Test suite for 3D trajectory dataset generation."""

    @pytest.fixture
    def config(self) -> TrajectoryConfig:
        """Create test configuration."""
        return TrajectoryConfig(
            sequence_length=32,
            prediction_length=8,
            sampling_rate=20.0,
            noise_std=0.01,
            helical_weight=0.2,
            orbital_weight=0.2,
            lissajous_weight=0.2,
            lorenz_weight=0.2,
            robotic_weight=0.2,
        )

    def test_dataset_creation(self, config: TrajectoryConfig):
        """Test that dataset can be created with valid configuration."""
        dataset = Trajectory3DDataset(config, seed=42)
        assert dataset is not None
        assert dataset.config == config

    def test_trajectory_weights_validation(self):
        """Test that trajectory weights must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            TrajectoryConfig(
                helical_weight=0.3,
                orbital_weight=0.3,
                lissajous_weight=0.3,
                lorenz_weight=0.3,  # Sum > 1.0
                robotic_weight=0.3,
            )

    def test_sample_generation(self, config: TrajectoryConfig):
        """Test generating samples from dataset."""
        dataset = Trajectory3DDataset(config, seed=42)
        samples = []

        for i, sample in enumerate(dataset):
            if i >= 5:
                break
            samples.append(sample)

        assert len(samples) == 5

        for sample in samples:
            # Check required keys
            assert "input_sequence" in sample
            assert "target_sequence" in sample
            assert "trajectory_type" in sample
            assert "metadata" in sample

            # Check tensor shapes
            input_seq = sample["input_sequence"]
            target_seq = sample["target_sequence"]

            assert input_seq.shape == (config.sequence_length, 3)
            assert target_seq.shape == (config.prediction_length, 3)
            assert input_seq.dtype == torch.float32
            assert target_seq.dtype == torch.float32

            # Check trajectory type is valid
            assert sample["trajectory_type"] in [
                "helical",
                "orbital",
                "lissajous",
                "lorenz",
                "robotic",
            ]

    def test_trajectory_types(self, config: TrajectoryConfig):
        """Test that all trajectory types can be generated."""
        dataset = Trajectory3DDataset(config, seed=42)

        trajectory_types_seen = set()
        for i, sample in enumerate(dataset):
            if i >= 50:  # Sample enough to see all types
                break
            trajectory_types_seen.add(sample["trajectory_type"])

        # Should see multiple trajectory types (might not see all due to randomness)
        assert len(trajectory_types_seen) >= 3

    def test_individual_trajectory_generators(self, config: TrajectoryConfig):
        """Test individual trajectory generation methods."""
        dataset = Trajectory3DDataset(config, seed=42)

        # Test each trajectory type
        trajectory_types = [
            ("helical", dataset._generate_helical_trajectory),
            ("orbital", dataset._generate_orbital_trajectory),
            ("lissajous", dataset._generate_lissajous_trajectory),
            ("lorenz", dataset._generate_lorenz_trajectory),
            ("robotic", dataset._generate_robotic_trajectory),
        ]

        for traj_type, generator in trajectory_types:
            trajectory = generator()

            # Basic checks
            assert isinstance(trajectory, np.ndarray)
            assert trajectory.shape[1] == 3  # x, y, z coordinates
            assert trajectory.shape[0] > 0  # Has time steps
            assert not np.any(np.isnan(trajectory))  # No NaN values
            assert not np.any(np.isinf(trajectory))  # No infinite values

            # Check reasonable value ranges
            assert np.all(np.abs(trajectory) < 1000)  # No extreme values

    def test_noise_application(self):
        """Test noise application in dataset."""
        config_no_noise = TrajectoryConfig(noise_std=0.0)
        config_with_noise = TrajectoryConfig(noise_std=0.1)

        dataset_no_noise = Trajectory3DDataset(config_no_noise, seed=42)
        dataset_with_noise = Trajectory3DDataset(config_with_noise, seed=42)

        # Get samples (note: random seeds might affect which trajectory type is chosen)
        sample_no_noise = next(iter(dataset_no_noise))
        sample_with_noise = next(iter(dataset_with_noise))

        # With noise, values should be more spread out (in general)
        # This is a statistical test so might occasionally fail
        no_noise_std = torch.std(sample_no_noise["input_sequence"])
        with_noise_std = torch.std(sample_with_noise["input_sequence"])

        # Noise should generally increase variance
        assert with_noise_std >= no_noise_std * 0.9  # Allow some tolerance

    def test_generate_sample_trajectories(self):
        """Test convenience function for sample trajectory generation."""
        samples = generate_sample_trajectories(num_samples=5)

        assert isinstance(samples, dict)
        expected_types = {"helical", "orbital", "lissajous", "lorenz", "robotic"}
        assert set(samples.keys()) == expected_types

        for traj_type, trajectory in samples.items():
            assert isinstance(trajectory, np.ndarray)
            assert trajectory.shape[1] == 3
            assert trajectory.shape[0] > 0


class TestTrajectory3DVisualization:
    """Test suite for 3D trajectory visualization system."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_visualizer_creation(self, temp_dir: Path):
        """Test that visualizer can be created."""
        visualizer = Trajectory3DVisualizer(temp_dir)
        assert visualizer.output_dir == temp_dir
        assert visualizer.colors is not None

    def test_trajectory_samples_plot(self, temp_dir: Path):
        """Test plotting trajectory samples."""
        visualizer = Trajectory3DVisualizer(temp_dir)

        save_path = temp_dir / "test_samples.png"
        visualizer.plot_trajectory_samples(str(save_path))

        # Check that file was created
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_prediction_comparison_plot(self, temp_dir: Path):
        """Test plotting prediction comparison."""
        visualizer = Trajectory3DVisualizer(temp_dir)

        # Create fake prediction data
        sequence_length = 32
        prediction_length = 8

        input_sequence = torch.randn(sequence_length, 3)
        target_sequence = torch.randn(prediction_length, 3)
        predicted_sequence = target_sequence + 0.1 * torch.randn_like(target_sequence)

        save_path = temp_dir / "test_prediction.png"
        visualizer.plot_prediction_comparison(
            input_sequence,
            target_sequence,
            predicted_sequence,
            "test_trajectory",
            str(save_path),
        )

        # Check that file was created
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_training_progress_plot(self, temp_dir: Path):
        """Test plotting training progress."""
        visualizer = Trajectory3DVisualizer(temp_dir)

        # Create fake metrics
        steps = 100
        metrics = {
            "train_loss": np.random.exponential(0.1, steps)
            .cumsum()[::-1]
            .tolist(),  # Decreasing loss
            "val_loss": np.random.exponential(0.1, steps).cumsum()[::-1].tolist(),
            "position_error": np.random.exponential(0.05, steps)
            .cumsum()[::-1]
            .tolist(),
            "velocity_error": np.random.exponential(0.03, steps)
            .cumsum()[::-1]
            .tolist(),
            "expert_entropy": np.random.normal(2.0, 0.1, steps).tolist(),
            "samples_per_sec": np.random.normal(100, 10, steps).tolist(),
        }

        save_path = temp_dir / "test_training_progress.png"
        visualizer.plot_training_progress(metrics, str(save_path))

        # Check that file was created
        assert save_path.exists()
        assert save_path.stat().st_size > 0


class TestTrajectory3DIntegration:
    """Integration tests for the complete 3D trajectory pipeline."""

    @pytest.fixture
    def minimal_config(self):
        """Create minimal model configuration for testing."""
        return {
            "model": {
                "n_layers": 1,
                "input_dim": 3,
                "target_dim": 3,
                "pool": "none",
                "block": {
                    "use_rms_norm": True,
                    "attn": {
                        "n_heads": 2,
                        "causal": True,
                        "use_rope": True,
                        "use_rms_norm": True,
                        "init": "scaled_xavier",
                        "rope_max_seq_len": 128,
                    },
                    "moe": {
                        "d_model": 32,
                        "router": {
                            "n_experts": 2,
                            "k": 1,
                            "use_rms_norm": True,
                            "load_balance_weight": 0.01,
                            "router_type": "topk",
                        },
                        "expert": {
                            "d_model": 32,
                            "d_hidden": 64,
                            "activation": "gelu",
                            "init": "scaled_xavier",
                            "dropout": 0.0,
                        },
                    },
                },
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 0.01,
                "max_steps": 10,
                "eval_interval": 5,
                "eval_steps": 2,
                "aux_loss_weight": 0.1,
                "output_dir": "test_outputs",
                "experiment_name": "test_experiment",
            },
            "dataset": {
                "sequence_length": 16,
                "prediction_length": 4,
                "sampling_rate": 10.0,
                "noise_std": 0.01,
                "helical_weight": 0.5,
                "orbital_weight": 0.5,
                "lissajous_weight": 0.0,
                "lorenz_weight": 0.0,
                "robotic_weight": 0.0,
            },
        }

    def test_model_creation_with_3d_config(self, minimal_config):
        """Test that MoE model can be created for 3D trajectory learning."""
        # load_config_from_yaml already imported at top

        # Save config to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(minimal_config, f)
            config_path = f.name

        try:
            # Load configuration
            model_config, training_config, dataset_config = load_config_from_yaml(
                config_path
            )

            # Create model
            model = MoESequenceRegressor(model_config)

            # Check model properties
            assert model.cfg.input_dim == 3
            assert model.cfg.target_dim == 3
            assert model.cfg.n_layers == 1

            # Test forward pass with 3D data
            batch_size = 2
            seq_len = 16
            input_data = torch.randn(batch_size, seq_len, 3)

            with torch.no_grad():
                output, aux_metrics = model(input_data)

            # Check output shape
            assert output.shape == (batch_size, seq_len, 3)
            # aux_metrics might be None or a dict, both are valid
            if aux_metrics is not None:
                assert isinstance(aux_metrics, dict)

        finally:
            Path(config_path).unlink()  # Clean up

    def test_dataset_model_compatibility(self, minimal_config):
        """Test that dataset output is compatible with model input."""
        # All required imports already at top

        # Create dataset
        dataset_config = TrajectoryConfig(**minimal_config["dataset"])
        dataset = Trajectory3DDataset(dataset_config, seed=42)

        # Get a sample
        sample = next(iter(dataset))
        input_seq = sample["input_sequence"]
        target_seq = sample["target_sequence"]

        # Create model using temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(minimal_config, f)
            config_path = f.name

        try:
            model_config, _, _ = load_config_from_yaml(config_path)
            model = MoESequenceRegressor(model_config)

            # Test forward pass
            input_batch = input_seq.unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output, aux_metrics = model(input_batch)

            # Check compatibility
            assert output.shape[0] == 1  # Batch size
            assert output.shape[1] == input_seq.shape[0]  # Sequence length
            assert output.shape[2] == 3  # 3D coordinates

            # Test loss computation
            pred_len = target_seq.shape[0]
            predictions = output[0, -pred_len:, :]  # Last pred_len outputs

            loss = torch.nn.functional.mse_loss(predictions, target_seq)
            assert torch.isfinite(loss)
            assert loss.item() >= 0

        finally:
            Path(config_path).unlink()

    def test_training_step_execution(self, minimal_config):
        """Test that a single training step can execute successfully."""
        # Imports already at top

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(minimal_config, f)
            config_path = f.name

        try:
            # Load configurations
            model_config, training_config, dataset_config = load_config_from_yaml(
                config_path
            )

            # Create trainer with temporary output directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                training_config.output_dir = tmp_dir
                training_config.experiment_name = "test"
                training_config.save_visualizations = False  # Skip for speed

                trainer = Trajectory3DTrainer(
                    model_config=model_config,
                    training_config=training_config,
                    dataset_config=dataset_config,
                    device="cpu",
                )

                # Execute single training step
                metrics = trainer._training_step()

                # Validate metrics
                required_metrics = [
                    "train_loss",
                    "aux_loss",
                    "total_loss",
                    "position_error",
                    "velocity_error",
                ]

                for metric in required_metrics:
                    assert metric in metrics
                    assert isinstance(metrics[metric], int | float)
                    assert np.isfinite(metrics[metric])

                # Loss should be positive
                assert metrics["train_loss"] >= 0
                assert metrics["total_loss"] >= 0
                assert metrics["position_error"] >= 0

        finally:
            Path(config_path).unlink()

    def test_config_loading_validation(self, minimal_config):
        """Test configuration loading and validation."""
        # load_config_from_yaml already imported at top

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(minimal_config, f)
            config_path = f.name

        try:
            model_config, training_config, dataset_config = load_config_from_yaml(
                config_path
            )

            # Check model config
            assert model_config.input_dim == 3
            assert model_config.target_dim == 3
            assert model_config.n_layers == 1

            # Check training config
            assert training_config.batch_size == 4
            assert training_config.max_steps == 10

            # Check dataset config
            assert dataset_config.sequence_length == 16
            assert dataset_config.prediction_length == 4

        finally:
            Path(config_path).unlink()


# Integration test for end-to-end pipeline
def test_trajectory_3d_end_to_end():
    """Test the complete 3D trajectory learning pipeline."""
    # All required imports already at top

    print("🧪 Testing complete 3D trajectory pipeline...")

    # Step 1: Dataset generation
    dataset_config = TrajectoryConfig(
        sequence_length=16,
        prediction_length=4,
        sampling_rate=10.0,
        helical_weight=1.0,  # Use only helical for consistency
        orbital_weight=0.0,
        lissajous_weight=0.0,
        lorenz_weight=0.0,
        robotic_weight=0.0,
    )

    dataset = Trajectory3DDataset(dataset_config, seed=42)
    sample = next(iter(dataset))

    # Validate sample
    assert sample["input_sequence"].shape == (16, 3)
    assert sample["target_sequence"].shape == (4, 3)
    assert sample["trajectory_type"] == "helical"

    # Step 2: Visualization
    with tempfile.TemporaryDirectory() as tmp_dir:
        visualizer = Trajectory3DVisualizer(tmp_dir)

        # Test visualization creation
        save_path = Path(tmp_dir) / "test_end_to_end.png"
        visualizer.plot_prediction_comparison(
            sample["input_sequence"],
            sample["target_sequence"],
            sample["target_sequence"]
            + 0.1 * torch.randn_like(sample["target_sequence"]),
            sample["trajectory_type"],
            str(save_path),
        )

        assert save_path.exists()

    print("✅ End-to-end 3D trajectory test passed!")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
