"""
3D Trajectory Dataset Generation

Generates diverse 3D trajectory types for sequence learning:
- Helical trajectories (DNA helix, spirals)
- Orbital mechanics (elliptical orbits, planetary motion)
- Lissajous curves (3D harmonic oscillations)
- Lorenz attractor (chaotic dynamics)
- Robotic trajectories (waypoint interpolation)
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
from torch.utils.data import IterableDataset


class TrajectoryMetadata(TypedDict):
    """Metadata for trajectory samples."""

    sequence_length: int
    prediction_length: int
    sampling_rate: float


class TrajectorySample(TypedDict):
    """Sample from trajectory dataset."""

    input_sequence: torch.Tensor
    target_sequence: torch.Tensor
    trajectory_type: str
    metadata: TrajectoryMetadata


@dataclass(slots=True)
class TrajectoryConfig:
    """Configuration for 3D trajectory generation."""

    sequence_length: int = 128  # Input trajectory length
    prediction_length: int = 32  # Future steps to predict
    sampling_rate: float = 100.0  # Samples per second
    noise_std: float = 0.01  # Gaussian noise standard deviation

    # Trajectory type weights (should sum to 1.0)
    helical_weight: float = 0.2
    orbital_weight: float = 0.2
    lissajous_weight: float = 0.2
    lorenz_weight: float = 0.2
    robotic_weight: float = 0.2

    def __post_init__(self):
        """Validate configuration after initialization."""
        total_weight = (
            self.helical_weight
            + self.orbital_weight
            + self.lissajous_weight
            + self.lorenz_weight
            + self.robotic_weight
        )
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Trajectory weights must sum to 1.0, got {total_weight}")


type TrajectoryPoint = tuple[float, float, float]


class Trajectory3DDataset(IterableDataset[TrajectorySample]):
    """
    Infinite dataset of 3D trajectory sequences.

    Generates diverse trajectory types with configurable parameters.
    Each sample contains input sequence and target prediction sequence.
    """

    def __init__(self, config: TrajectoryConfig, seed: int | None = None):
        self.config = config
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Create cumulative weight distribution for trajectory type selection
        self.trajectory_cdf = np.cumsum(
            [
                config.helical_weight,
                config.orbital_weight,
                config.lissajous_weight,
                config.lorenz_weight,
                config.robotic_weight,
            ]
        )

    def __iter__(self) -> Iterator[TrajectorySample]:
        """Generate infinite stream of 3D trajectory sequences."""
        while True:
            # Select trajectory type based on weights
            rand_val = self.rng.random()
            if rand_val < self.trajectory_cdf[0]:
                trajectory_type = "helical"
                trajectory = self._generate_helical_trajectory()
            elif rand_val < self.trajectory_cdf[1]:
                trajectory_type = "orbital"
                trajectory = self._generate_orbital_trajectory()
            elif rand_val < self.trajectory_cdf[2]:
                trajectory_type = "lissajous"
                trajectory = self._generate_lissajous_trajectory()
            elif rand_val < self.trajectory_cdf[3]:
                trajectory_type = "lorenz"
                trajectory = self._generate_lorenz_trajectory()
            else:
                trajectory_type = "robotic"
                trajectory = self._generate_robotic_trajectory()

            # Extract sequence from trajectory
            total_length = self.config.sequence_length + self.config.prediction_length
            if len(trajectory) < total_length:
                continue  # Skip if trajectory too short

            # Random starting point in trajectory
            start_idx = self.rng.randint(0, len(trajectory) - total_length)
            sequence = trajectory[start_idx : start_idx + total_length]

            # Split into input and target
            input_sequence = sequence[: self.config.sequence_length]
            target_sequence = sequence[self.config.sequence_length :]

            # Add noise
            if self.config.noise_std > 0:
                noise = self.np_rng.normal(
                    0, self.config.noise_std, input_sequence.shape
                )
                input_sequence = input_sequence + noise

            # Convert to tensors
            input_tensor = torch.tensor(
                input_sequence, dtype=torch.float32
            )  # [seq_len, 3]
            target_tensor = torch.tensor(
                target_sequence, dtype=torch.float32
            )  # [pred_len, 3]

            yield {
                "input_sequence": input_tensor,
                "target_sequence": target_tensor,
                "trajectory_type": trajectory_type,
                "metadata": {
                    "sequence_length": self.config.sequence_length,
                    "prediction_length": self.config.prediction_length,
                    "sampling_rate": self.config.sampling_rate,
                },
            }

    def _generate_helical_trajectory(self) -> np.ndarray:
        """Generate helical/spiral trajectory (DNA helix, spiral staircases)."""
        # Random helical parameters
        radius = self.rng.uniform(1.0, 3.0)
        pitch = self.rng.uniform(0.2, 1.0)  # Vertical distance per revolution
        turns = self.rng.uniform(2.0, 6.0)  # Number of complete turns
        phase = self.rng.uniform(0, 2 * math.pi)

        # Generate helical trajectory
        total_samples = int(self.config.sampling_rate * turns * 2)  # 2 seconds per turn
        t = np.linspace(0, turns * 2 * math.pi, total_samples)

        x = radius * np.cos(t + phase)
        y = radius * np.sin(t + phase)
        z = pitch * t / (2 * math.pi)  # Linear vertical progression

        return np.column_stack([x, y, z])

    def _generate_orbital_trajectory(self) -> np.ndarray:
        """Generate orbital trajectory (elliptical orbits, planetary motion)."""
        # Random orbital parameters
        semi_major_a = self.rng.uniform(2.0, 5.0)
        eccentricity = self.rng.uniform(0.0, 0.8)
        inclination = self.rng.uniform(0, math.pi / 4)  # Up to 45 degrees
        argument_periapsis = self.rng.uniform(0, 2 * math.pi)

        semi_minor_b = semi_major_a * math.sqrt(1 - eccentricity**2)

        # Generate orbital trajectory
        total_samples = int(self.config.sampling_rate * 8)  # 8 second orbit
        t = np.linspace(0, 2 * math.pi, total_samples)

        # Elliptical orbit in 2D
        x_orbit = semi_major_a * np.cos(t)
        y_orbit = semi_minor_b * np.sin(t)

        # Apply 3D rotation (inclination and argument of periapsis)
        cos_inc = math.cos(inclination)
        sin_inc = math.sin(inclination)
        cos_arg = math.cos(argument_periapsis)
        sin_arg = math.sin(argument_periapsis)

        x = cos_arg * x_orbit - sin_arg * cos_inc * y_orbit
        y = sin_arg * x_orbit + cos_arg * cos_inc * y_orbit
        z = sin_inc * y_orbit

        return np.column_stack([x, y, z])

    def _generate_lissajous_trajectory(self) -> np.ndarray:
        """Generate 3D Lissajous curves (3D harmonic oscillations)."""
        # Random frequency ratios (should create interesting patterns)
        freq_x = self.rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])
        freq_y = self.rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])
        freq_z = self.rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])

        # Random phases
        phase_x = self.rng.uniform(0, 2 * math.pi)
        phase_y = self.rng.uniform(0, 2 * math.pi)
        phase_z = self.rng.uniform(0, 2 * math.pi)

        # Random amplitudes
        amp_x = self.rng.uniform(1.0, 3.0)
        amp_y = self.rng.uniform(1.0, 3.0)
        amp_z = self.rng.uniform(1.0, 3.0)

        # Generate Lissajous trajectory
        duration = 10.0  # 10 seconds to capture full pattern
        total_samples = int(self.config.sampling_rate * duration)
        t = np.linspace(0, duration, total_samples)

        x = amp_x * np.sin(freq_x * t + phase_x)
        y = amp_y * np.sin(freq_y * t + phase_y)
        z = amp_z * np.sin(freq_z * t + phase_z)

        return np.column_stack([x, y, z])

    def _generate_lorenz_trajectory(self) -> np.ndarray:
        """Generate Lorenz attractor trajectory (chaotic dynamics)."""
        # Lorenz system parameters (classic values with slight randomization)
        sigma = self.rng.uniform(9.0, 11.0)  # Narrow range around 10
        rho = self.rng.uniform(26.0, 30.0)  # Narrow range around 28
        beta = self.rng.uniform(2.5, 2.8)  # Narrow range around 8/3

        # Random initial conditions (smaller range to avoid overflow)
        x0 = self.rng.uniform(-10, 10)
        y0 = self.rng.uniform(-10, 10)
        z0 = self.rng.uniform(5, 25)

        # Integrate Lorenz equations with smaller timestep
        dt = 0.5 / self.config.sampling_rate  # Half the normal timestep for stability
        total_samples = int(self.config.sampling_rate * 10)  # 10 seconds

        trajectory = np.zeros((total_samples, 3))
        trajectory[0] = [x0, y0, z0]

        for i in range(1, total_samples):
            x, y, z = trajectory[i - 1]

            # Check for numerical overflow
            if np.abs(x) > 100 or np.abs(y) > 100 or np.abs(z) > 100:
                # Reset to avoid overflow
                x, y, z = x0, y0, z0

            # Lorenz equations
            dx_dt = sigma * (y - x)
            dy_dt = x * (rho - z) - y
            dz_dt = x * y - beta * z

            # Euler integration with stability check
            new_x = x + dx_dt * dt
            new_y = y + dy_dt * dt
            new_z = z + dz_dt * dt

            # Check for overflow and clip values
            new_x = np.clip(new_x, -50, 50)
            new_y = np.clip(new_y, -50, 50)
            new_z = np.clip(new_z, 0, 50)

            trajectory[i] = [new_x, new_y, new_z]

        return trajectory

    def _generate_robotic_trajectory(self) -> np.ndarray:
        """Generate realistic robotic trajectory with waypoint interpolation."""
        # Generate random waypoints in 3D space
        num_waypoints = self.rng.randint(3, 8)
        waypoints_list: list[list[float]] = []

        # First waypoint
        waypoints_list.append(
            [self.rng.uniform(-3, 3), self.rng.uniform(-3, 3), self.rng.uniform(0, 4)]
        )

        # Subsequent waypoints (ensuring reasonable distances)
        for _ in range(num_waypoints - 1):
            prev_point = waypoints_list[-1]
            # New point within reasonable distance from previous
            new_point = [
                prev_point[0] + self.rng.uniform(-2, 2),
                prev_point[1] + self.rng.uniform(-2, 2),
                prev_point[2] + self.rng.uniform(-1, 1),
            ]
            waypoints_list.append(new_point)

        waypoints = np.array(waypoints_list)

        # Interpolate between waypoints with smooth curves
        trajectory_segments = []

        for i in range(len(waypoints) - 1):
            start_point = waypoints[i]
            end_point = waypoints[i + 1]

            # Time for this segment (based on distance)
            distance = np.linalg.norm(end_point - start_point)
            segment_time = max(1.0, float(distance))  # At least 1 second per segment
            segment_samples = int(self.config.sampling_rate * segment_time)

            # Smooth interpolation (cubic spline-like)
            t = np.linspace(0, 1, segment_samples)

            # Add some curvature for realism (simple cubic interpolation)
            # Control points for smooth curve
            if i == 0:
                control_start = start_point
            else:
                control_start = start_point + 0.3 * (start_point - waypoints[i - 1])

            if i == len(waypoints) - 2:
                control_end = end_point
            else:
                control_end = end_point + 0.3 * (waypoints[i + 2] - end_point)

            # Cubic Bézier curve approximation
            segment = np.zeros((segment_samples, 3))
            for j in range(3):  # x, y, z coordinates
                segment[:, j] = (
                    (1 - t) ** 3 * start_point[j]
                    + 3 * (1 - t) ** 2 * t * control_start[j]
                    + 3 * (1 - t) * t**2 * control_end[j]
                    + t**3 * end_point[j]
                )

            trajectory_segments.append(segment)

        # Concatenate all segments
        if trajectory_segments:
            trajectory = np.vstack(trajectory_segments)
        else:
            # Fallback: just return first waypoint repeated
            trajectory = np.tile(waypoints[0], (100, 1))

        return trajectory


def create_trajectory_dataset(
    config: TrajectoryConfig, seed: int | None = None
) -> Trajectory3DDataset:
    """Create a 3D trajectory dataset with specified configuration."""
    return Trajectory3DDataset(config, seed)


# Convenience function for quick testing
def generate_sample_trajectories(num_samples: int = 5) -> dict[str, np.ndarray]:
    """Generate sample trajectories for visualization and testing."""
    config = TrajectoryConfig()
    dataset = Trajectory3DDataset(config, seed=42)

    samples = {}
    trajectory_types = ["helical", "orbital", "lissajous", "lorenz", "robotic"]

    for i, trajectory_type in enumerate(trajectory_types):
        if trajectory_type == "helical":
            trajectory = dataset._generate_helical_trajectory()
        elif trajectory_type == "orbital":
            trajectory = dataset._generate_orbital_trajectory()
        elif trajectory_type == "lissajous":
            trajectory = dataset._generate_lissajous_trajectory()
        elif trajectory_type == "lorenz":
            trajectory = dataset._generate_lorenz_trajectory()
        else:  # robotic
            trajectory = dataset._generate_robotic_trajectory()

        samples[trajectory_type] = trajectory

    return samples


if __name__ == "__main__":
    # Quick test of trajectory generation
    print("🎯 Testing 3D Trajectory Dataset Generation")

    config = TrajectoryConfig(sequence_length=64, prediction_length=16)
    dataset = Trajectory3DDataset(config, seed=42)

    # Generate a few samples
    samples = []
    for i, sample in enumerate(dataset):
        if i >= 5:
            break
        samples.append(sample)
        print(f"Sample {i + 1}: {sample['trajectory_type']}")
        print(f"  Input shape: {sample['input_sequence'].shape}")
        print(f"  Target shape: {sample['target_sequence'].shape}")
        print(
            f"  Position range: ({sample['input_sequence'].min():.2f}, {sample['input_sequence'].max():.2f})"
        )

    print("✅ Trajectory generation working correctly!")
