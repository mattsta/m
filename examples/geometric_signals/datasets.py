"""
Signal generation datasets for geometric learning tasks.
Creates various types of signals for regression learning with MoE transformers.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset


@dataclass(slots=True)
class SignalParams:
    """Parameters for signal generation."""

    frequency: float = 1.0
    amplitude: float = 1.0
    phase: float = 0.0
    noise_std: float = 0.0
    signal_type: str = "sine"


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for signal datasets."""

    sequence_length: int = 128  # Input sequence length
    prediction_length: int = 32  # Number of future steps to predict
    sampling_rate: float = 100.0  # Samples per second
    num_samples: int = 100000  # Total samples per epoch

    # Signal parameter ranges
    frequency_range: tuple[float, float] = (0.1, 10.0)
    amplitude_range: tuple[float, float] = (0.5, 2.0)
    phase_range: tuple[float, float] = (0.0, 2 * math.pi)
    noise_std_range: tuple[float, float] = (0.0, 0.1)


class SineWaveDataset(IterableDataset):
    """Dataset generating sine waves with varying parameters."""

    def __init__(self, config: DatasetConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.total_length = config.sequence_length + config.prediction_length

    def _generate_signal(self, params: SignalParams) -> torch.Tensor:
        """Generate a single signal with given parameters."""
        t = (
            torch.arange(self.total_length, dtype=torch.float32)
            / self.config.sampling_rate
        )

        if params.signal_type == "sine":
            signal = params.amplitude * torch.sin(
                2 * math.pi * params.frequency * t + params.phase
            )
        elif params.signal_type == "cosine":
            signal = params.amplitude * torch.cos(
                2 * math.pi * params.frequency * t + params.phase
            )
        elif params.signal_type == "square":
            signal = params.amplitude * torch.sign(
                torch.sin(2 * math.pi * params.frequency * t + params.phase)
            )
        elif params.signal_type == "sawtooth":
            # Sawtooth wave
            signal = params.amplitude * (
                2 * (t * params.frequency + params.phase / (2 * math.pi)) % 1 - 1
            )
        else:
            raise ValueError(f"Unknown signal type: {params.signal_type}")

        # Add noise
        if params.noise_std > 0:
            noise = torch.normal(0, params.noise_std, signal.shape)
            signal = signal + noise

        return signal

    def _sample_params(self, rng: random.Random) -> SignalParams:
        """Sample random signal parameters."""
        return SignalParams(
            frequency=rng.uniform(*self.config.frequency_range),
            amplitude=rng.uniform(*self.config.amplitude_range),
            phase=rng.uniform(*self.config.phase_range),
            noise_std=rng.uniform(*self.config.noise_std_range),
            signal_type="sine",  # Default to sine
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Generate infinite stream of signal samples."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Multi-worker setup - ensure different seeds per worker
            rng = random.Random(self.seed + worker_info.id)
        else:
            rng = random.Random(self.seed)

        while True:
            params = self._sample_params(rng)
            signal = self._generate_signal(params)

            # Split into input and target
            input_seq = signal[: self.config.sequence_length].unsqueeze(
                -1
            )  # [seq_len, 1]
            target_seq = signal[self.config.sequence_length :].unsqueeze(
                -1
            )  # [pred_len, 1]

            yield input_seq, target_seq

    def __len__(self) -> int:
        """Return nominal dataset size per epoch."""
        return self.config.num_samples


class CompositeSignalDataset(IterableDataset):
    """Dataset generating composite signals (sum of multiple sine waves)."""

    def __init__(self, config: DatasetConfig, max_components: int = 4, seed: int = 42):
        self.config = config
        self.max_components = max_components
        self.seed = seed
        self.total_length = config.sequence_length + config.prediction_length

    def _generate_composite_signal(
        self, components: list[SignalParams]
    ) -> torch.Tensor:
        """Generate composite signal from multiple components."""
        t = (
            torch.arange(self.total_length, dtype=torch.float32)
            / self.config.sampling_rate
        )
        signal = torch.zeros_like(t)

        for params in components:
            if params.signal_type == "sine":
                component = params.amplitude * torch.sin(
                    2 * math.pi * params.frequency * t + params.phase
                )
            elif params.signal_type == "cosine":
                component = params.amplitude * torch.cos(
                    2 * math.pi * params.frequency * t + params.phase
                )
            else:
                component = params.amplitude * torch.sin(
                    2 * math.pi * params.frequency * t + params.phase
                )

            signal += component

        # Add composite noise
        if len(components) > 0:
            avg_noise = sum(c.noise_std for c in components) / len(components)
            if avg_noise > 0:
                noise = torch.normal(0, avg_noise, signal.shape)
                signal += noise

        return signal

    def _sample_components(self, rng: random.Random) -> list[SignalParams]:
        """Sample random composite signal components."""
        num_components = rng.randint(1, self.max_components)
        components = []

        for _ in range(num_components):
            params = SignalParams(
                frequency=rng.uniform(*self.config.frequency_range),
                amplitude=rng.uniform(0.2, 1.0),  # Smaller amplitudes for composites
                phase=rng.uniform(*self.config.phase_range),
                noise_std=rng.uniform(0.0, 0.05),  # Less noise per component
                signal_type=rng.choice(["sine", "cosine"]),
            )
            components.append(params)

        return components

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Generate infinite stream of composite signal samples."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rng = random.Random(self.seed + worker_info.id)
        else:
            rng = random.Random(self.seed)

        while True:
            components = self._sample_components(rng)
            signal = self._generate_composite_signal(components)

            # Split into input and target
            input_seq = signal[: self.config.sequence_length].unsqueeze(-1)
            target_seq = signal[self.config.sequence_length :].unsqueeze(-1)

            yield input_seq, target_seq

    def __len__(self) -> int:
        return self.config.num_samples


class GeometricShapeDataset(IterableDataset):
    """Dataset generating signals from geometric shapes projected to 1D."""

    def __init__(self, config: DatasetConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        self.total_length = config.sequence_length + config.prediction_length

    def _generate_circle_signal(
        self, radius: float, speed: float, phase: float, noise_std: float
    ) -> torch.Tensor:
        """Generate 1D projection of circular motion."""
        t = (
            torch.arange(self.total_length, dtype=torch.float32)
            / self.config.sampling_rate
        )
        # Project circle onto x-axis
        signal = radius * torch.cos(2 * math.pi * speed * t + phase)

        if noise_std > 0:
            noise = torch.normal(0, noise_std, signal.shape)
            signal += noise

        return signal

    def _generate_spiral_signal(
        self, growth_rate: float, speed: float, phase: float, noise_std: float
    ) -> torch.Tensor:
        """Generate 1D projection of spiral motion."""
        t = (
            torch.arange(self.total_length, dtype=torch.float32)
            / self.config.sampling_rate
        )
        radius = growth_rate * t
        # Project spiral onto x-axis
        signal = radius * torch.cos(2 * math.pi * speed * t + phase)

        if noise_std > 0:
            noise = torch.normal(0, noise_std, signal.shape)
            signal += noise

        return signal

    def _generate_lissajous_signal(
        self, freq_x: float, freq_y: float, phase_diff: float, noise_std: float
    ) -> torch.Tensor:
        """Generate 1D projection of Lissajous curve."""
        t = (
            torch.arange(self.total_length, dtype=torch.float32)
            / self.config.sampling_rate
        )
        # Take x-component of Lissajous curve
        signal = torch.sin(2 * math.pi * freq_x * t) + 0.5 * torch.sin(
            2 * math.pi * freq_y * t + phase_diff
        )

        if noise_std > 0:
            noise = torch.normal(0, noise_std, signal.shape)
            signal += noise

        return signal

    def _sample_shape_params(self, rng: random.Random) -> tuple[str, dict]:
        """Sample random geometric shape parameters."""
        shape_type = rng.choice(["circle", "spiral", "lissajous"])

        if shape_type == "circle":
            params = {
                "radius": rng.uniform(0.5, 2.0),
                "speed": rng.uniform(0.1, 5.0),
                "phase": rng.uniform(0, 2 * math.pi),
                "noise_std": rng.uniform(0, 0.1),
            }
        elif shape_type == "spiral":
            params = {
                "growth_rate": rng.uniform(0.01, 0.1),
                "speed": rng.uniform(0.1, 3.0),
                "phase": rng.uniform(0, 2 * math.pi),
                "noise_std": rng.uniform(0, 0.1),
            }
        else:  # lissajous
            params = {
                "freq_x": rng.uniform(0.5, 4.0),
                "freq_y": rng.uniform(0.5, 4.0),
                "phase_diff": rng.uniform(0, 2 * math.pi),
                "noise_std": rng.uniform(0, 0.1),
            }

        return shape_type, params

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Generate infinite stream of geometric shape signals."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rng = random.Random(self.seed + worker_info.id)
        else:
            rng = random.Random(self.seed)

        while True:
            shape_type, params = self._sample_shape_params(rng)

            if shape_type == "circle":
                signal = self._generate_circle_signal(**params)
            elif shape_type == "spiral":
                signal = self._generate_spiral_signal(**params)
            else:  # lissajous
                signal = self._generate_lissajous_signal(**params)

            # Split into input and target
            input_seq = signal[: self.config.sequence_length].unsqueeze(-1)
            target_seq = signal[self.config.sequence_length :].unsqueeze(-1)

            yield input_seq, target_seq

    def __len__(self) -> int:
        return self.config.num_samples


def create_mixed_dataset(config: DatasetConfig, seed: int = 42) -> IterableDataset:
    """Create a mixed dataset combining all signal types."""

    class MixedSignalDataset(IterableDataset):
        def __init__(self):
            self.datasets = [
                SineWaveDataset(config, seed),
                CompositeSignalDataset(config, seed=seed + 1000),
                GeometricShapeDataset(config, seed=seed + 2000),
            ]
            self.weights = [0.4, 0.4, 0.2]  # Sine, Composite, Geometric

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                rng = random.Random(seed + worker_info.id + 3000)
            else:
                rng = random.Random(seed + 3000)

            # Create iterators for each dataset
            iterators = [iter(dataset) for dataset in self.datasets]

            while True:
                # Choose dataset based on weights
                dataset_idx = rng.choices(
                    range(len(self.datasets)), weights=self.weights
                )[0]
                yield next(iterators[dataset_idx])

        def __len__(self):
            return config.num_samples

    return MixedSignalDataset()


# Convenience function for creating datasets
def create_signal_dataset(
    dataset_type: str,
    sequence_length: int = 128,
    prediction_length: int = 32,
    num_samples: int = 100000,
    seed: int = 42,
) -> IterableDataset:
    """Create a signal dataset of specified type."""

    config = DatasetConfig(
        sequence_length=sequence_length,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )

    if dataset_type == "sine":
        return SineWaveDataset(config, seed)
    elif dataset_type == "composite":
        return CompositeSignalDataset(config, seed=seed)
    elif dataset_type == "geometric":
        return GeometricShapeDataset(config, seed)
    elif dataset_type == "mixed":
        return create_mixed_dataset(config, seed)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
