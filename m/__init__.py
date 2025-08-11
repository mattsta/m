"""
M - MoE Platform Core Package

Provides core functionality for Mixture of Experts training and inference.
"""

# Core MoE functionality
from .moe import (
    AttentionConfig,
    BlockConfig,
    ExpertConfig,
    ModelConfig,
    MoEConfig,
    MoESequenceRegressor,
    RouterConfig,
)

# Rich CLI training visualization
from .rich_trainer_viz import (
    MinimalTrainerVisualizer,
    RichTrainerVisualizer,
    SystemMetrics,
    ThroughputMetrics,
    TrainingSnapshot,
    VisualizationConfig,
    create_default_visualizer,
    snapshot_from_trainer_state,
)

# Training visualization
from .training_viz import (
    MetricConfig,
    PlotConfig,
    RealTimeTrainingVisualizer,
    TrainingVisualizerConfig,
    create_custom_visualizer,
    create_loss_visualizer,
    create_moe_visualizer,
)

__all__ = [
    # Core MoE functionality
    "ModelConfig",
    "MoESequenceRegressor",
    "AttentionConfig",
    "BlockConfig",
    "ExpertConfig",
    "MoEConfig",
    "RouterConfig",
    # Training visualization
    "RealTimeTrainingVisualizer",
    "TrainingVisualizerConfig",
    "PlotConfig",
    "MetricConfig",
    "create_loss_visualizer",
    "create_moe_visualizer",
    "create_custom_visualizer",
    # Rich CLI training visualization
    "RichTrainerVisualizer",
    "MinimalTrainerVisualizer",
    "VisualizationConfig",
    "TrainingSnapshot",
    "ThroughputMetrics",
    "SystemMetrics",
    "create_default_visualizer",
    "snapshot_from_trainer_state",
]
