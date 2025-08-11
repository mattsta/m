# 3D Trajectory Learning with MoE Transformers

This example demonstrates how to use MoE (Mixture of Experts) transformers for 3D trajectory prediction tasks. The system learns to predict future 3D positions from trajectory sequences including helical, orbital, chaotic, and robotic motion patterns.

## Features

- **3D Trajectory Generation**: Creates diverse 3D motion patterns (helical, orbital, Lissajous, Lorenz, robotic)
- **Real-time Monitoring**: Live training visualization with 3D plots and expert utilization
- **Multiple Architectures**: Compare different MoE configurations (small, large, quick test)
- **Comprehensive Evaluation**: Performance analysis across trajectory types with 3D-specific metrics
- **Expert Analysis**: Visualization of expert specialization in 3D space and motion types
- **Production Integration**: Seamless integration with existing inference server infrastructure
- **3D Visualization**: Interactive plots, animations, and comparative analysis dashboards

## Quick Start

### 1. Generate Demo Dataset

```bash
# Create demo trajectory examples to visualize the task
uv run trajectory-3d-compare demo
```

This creates example 3D trajectories and saves visualizations to understand the prediction task.

### 2. Train a Small Model

```bash
# Train with the small MoE configuration
uv run trajectory-3d-train examples/trajectory_3d/configs/small_3d_moe.yaml --name my_experiment

# Or without custom name (uses default from config)
uv run trajectory-3d-train examples/trajectory_3d/configs/small_3d_moe.yaml
```

### 3. Evaluate the Trained Model

```bash
# Evaluate the trained model
uv run trajectory-3d-eval --latest

# Or specify model and config explicitly
uv run trajectory-3d-eval outputs/trajectory_3d/trajectory_experiment/best_model.pt
```

### 4. Run Full Comparison Study

```bash
# Compare all configurations
uv run trajectory-3d-compare compare
```

This trains and evaluates models with all available configurations and generates a comparison report.

## CLI Commands Reference

All commands use `uv run` with the CLI entrypoints defined in `pyproject.toml`:

### Training Commands

```bash
# Train with configuration
uv run trajectory-3d-train <config_path> [--name experiment_name] [--device auto]

# Train with all configurations and compare
uv run trajectory-3d-compare compare

# Generate demo trajectories
uv run trajectory-3d-demo
uv run trajectory-3d-demo --animate  # Include animations
```

### Evaluation Commands

```bash
# Evaluate trained model
uv run trajectory-3d-eval <model_path> [config_path]
uv run trajectory-3d-eval --latest                    # Use latest model
uv run trajectory-3d-eval --interactive               # Select model interactively

# Run comparative evaluation
uv run trajectory-3d-compare evaluate --model <model_path> --config <config_path>
```

### Model Management

```bash
# List all trained models
uv run trajectory-3d-models list

# Get latest model path
uv run trajectory-3d-models latest

# Show detailed model information
uv run trajectory-3d-models details [experiment_name]

# Interactive model selection
uv run trajectory-3d-models interactive
```

### Inference & Deployment

```bash
# Demo core inference platform integration
uv run trajectory-3d-inference-demo --latest

# Start interactive inference server
uv run trajectory-3d-inference-demo --serve --port 8080 --device auto

# Create optimized production deployment
uv run trajectory-3d-deploy <model_path> <deployment_name>
uv run trajectory-3d-deploy --latest [deployment_name]

# Test deployment
cd outputs/trajectory_3d/deployments/my_deployment
python start_server.py --port 8080
```

## Trajectory Types

The system learns five types of 3D trajectories:

### Helical Trajectories

- DNA helix patterns, spiral staircases
- Parameters: radius (1.0-3.0), pitch (0.2-1.0), turns (2.0-6.0)
- Applications: protein folding, mechanical systems

### Orbital Mechanics

- Elliptical orbits, planetary motion
- Parameters: semi-major axis (2.0-5.0), eccentricity (0.0-0.8), inclination
- Applications: aerospace, celestial mechanics

### Lissajous Curves

- 3D harmonic oscillations with frequency ratios
- Parameters: frequency ratios (1.0-3.0), phases, amplitudes
- Applications: vibration analysis, signal processing

### Lorenz Attractor

- Chaotic dynamics with butterfly effect
- Parameters: sigma (9-11), rho (26-30), beta (2.5-2.8)
- Applications: chaos theory, weather modeling

### Robotic Trajectories

- Smooth waypoint interpolation for pick-and-place
- Parameters: 3-8 waypoints, cubic Bézier smoothing
- Applications: robotics, motion planning, automation

## Configuration Files

### `quick_test_3d.yaml`

- 2 layers, 4 experts, 64 dimensions
- 1000 steps for rapid testing
- Good for setup verification

### `small_3d_moe.yaml`

- 4 layers, 8 experts, 256 dimensions
- 20K steps for balanced experimentation
- Good for development and testing

### `large_3d_moe.yaml`

- 8 layers, 32 experts, 1024 dimensions
- 50K steps for high-capacity learning
- Good for research and production

## Training Configuration

Key training parameters:

```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  max_steps: 20000
  gradient_clip_norm: 1.0
  eval_interval: 500

dataset:
  sequence_length: 128 # Input trajectory length
  prediction_length: 32 # Future steps to predict
  sampling_rate: 100.0 # Samples per second
```

## Real-time Monitoring

During training, the system provides:

- **Loss Curves**: Training and validation loss over time
- **3D Error Metrics**: Position, velocity, and acceleration errors
- **Expert Utilization**: Which experts are being used for different trajectory types
- **3D Trajectory Predictions**: Live examples of model predictions vs targets in 3D space
- **Training Throughput**: Samples and tokens processed per second

## Evaluation Metrics

The evaluation system computes:

### 3D Trajectory Metrics

- **Position Error**: Mean Euclidean distance between predicted and actual 3D positions
- **Velocity Error**: Error in 3D velocity vectors (consecutive position differences)
- **Acceleration Error**: Error in acceleration/curvature of 3D paths
- **Trajectory Deviation**: Cumulative path length difference
- **Endpoint Error**: Error at the final predicted position

### Temporal Consistency Metrics

- **Temporal Consistency**: Smoothness of predicted motion over time
- **Smoothness Metric**: Inverse of jerk (third derivative) magnitude
- **Energy Conservation**: How well predicted trajectories conserve kinetic energy

### Standard Regression Metrics

- **MSE Loss**: Mean squared error across all 3D coordinates
- **MAE Loss**: Mean absolute error across all 3D coordinates
- **R² Score**: Coefficient of determination for 3D trajectory fitting

### Expert Analysis

- **Expert Utilization**: How evenly experts are used
- **Trajectory Specialization**: Which experts specialize on which trajectory types
- **Spatial Specialization**: Which experts handle which 3D regions
- **Routing Entropy**: Measure of routing diversity

## Platform Integration Philosophy

### Core Platform Usage

This example demonstrates **proper platform usage** for multi-dimensional data:

- **Use `m.moe` directly**: No custom MoE implementations for 3D data
- **Natural 3D support**: Existing platform handles 3D coordinates seamlessly
- **3D trajectories as "tokens"**: Natural extension of 1D signal processing
- **Standard checkpoints**: Compatible with core platform loaders

### Key Discovery

**The existing `InferenceServer` naturally supports 3D trajectories!**

```python
# This just works - no modification needed for 3D data
server = InferenceServer(model_path="trajectory_3d_model.pt", device="auto")

request = {
    "tokens": [[1.0, 2.0, 3.0], [1.1, 2.1, 2.9], [1.2, 2.3, 2.7]],  # 3D points
    "max_new_tokens": 16,                    # Predict 16 more 3D positions
    "temperature": 1.0
}

response = await server.handle_request(request)
predictions = response["generated_tokens"]  # More 3D coordinates
```

## Output Structure

```
outputs/trajectory_3d/
├── experiment_name/
│   ├── best_model.pt              # Best model checkpoint
│   ├── final_model.pt             # Final model state
│   ├── training_progress.png      # Training visualizations
│   ├── training_summary.yaml      # Training metadata
│   ├── visualizations/
│   │   └── prediction_step_*.png  # 3D prediction plots during training
│   └── evaluation/
│       ├── prediction_*.png       # 3D trajectory prediction comparisons
│       ├── metrics_comparison.png # Performance comparison across types
│       ├── trajectory_analysis_dashboard.png # Comprehensive analysis
│       └── evaluation_summary.txt # Detailed text summary
├── comparison_study/
│   └── comparison_report.txt       # Multi-config comparison
└── demo/
    ├── demo_trajectories.png       # Sample trajectory types
    └── demo_prediction_*.png       # Mock prediction examples
```

## Deep Dive Architecture

### File Structure and Responsibilities

```
examples/trajectory_3d/
├── __init__.py                        # Package initialization (minimal)
├── datasets.py                        # 3D trajectory generation (5 types)
├── visualization.py                   # 3D plotting, animations, real-time monitoring
├── training.py                        # Core training pipeline with 3D metrics
├── evaluation.py                      # Model evaluation and 3D performance analysis
├── run_experiments.py                 # Orchestrates multi-configuration experiments
├── model_discovery.py                 # Dynamic model finding and management utilities
├── inference_demo.py                  # Demonstrates core m.inference_server directly
├── deploy.py                          # Simple deployment using core platform features
│
└── configs/
    ├── quick_test_3d.yaml             # Quick testing configuration
    ├── small_3d_moe.yaml              # Small MoE configuration
    └── large_3d_moe.yaml              # Large MoE configuration
```

### Core System Components

#### 1. **Data Pipeline** (`datasets.py`)

- **Purpose**: Generate diverse 3D trajectory types for training and evaluation
- **Capabilities**: Helical, orbital, Lissajous, Lorenz, robotic trajectories
- **Integration**: PyTorch IterableDataset with infinite streaming
- **Status**: ✅ Stable and well-tested

#### 2. **Training System** (`training.py`)

- **Purpose**: Complete training pipeline with real-time 3D monitoring
- **Key Features**: 3D-specific metrics, expert utilization tracking, live visualization
- **Integration**: Uses core `m.moe` infrastructure directly
- **Status**: ✅ Production-ready with comprehensive 3D metrics

#### 3. **Evaluation Framework** (`evaluation.py`)

- **Purpose**: Comprehensive model performance analysis for 3D trajectories
- **Metrics**: 3D position/velocity/acceleration errors, trajectory analysis, expert utilization
- **Output**: Detailed reports with 3D visualizations and comparative dashboards
- **Status**: ✅ Comprehensive evaluation suite

#### 4. **Visualization Engine** (`visualization.py`)

- **Purpose**: 3D plotting, animations, and real-time training monitoring
- **Features**: Interactive 3D plots, trajectory animations, prediction comparisons
- **Integration**: Matplotlib 3D with save/display modes
- **Status**: ✅ Full-featured 3D visualization system

#### 5. **Experiment Orchestration** (`run_experiments.py`)

- **Purpose**: Multi-configuration training and comparison for 3D trajectory learning
- **Features**: Parallel training, comparative analysis, report generation
- **Integration**: Coordinates training.py and evaluation.py
- **Status**: ✅ Stable experiment runner

#### 6. **Model Discovery** (`model_discovery.py`)

- **Purpose**: Dynamic model finding, metadata extraction, and interactive selection
- **Capabilities**: Auto-discovery, architecture analysis, interactive selection UI
- **Integration**: Used by all inference and deployment tools
- **Status**: ✅ Robust discovery system

#### 7. **Direct Integration Demo** (`inference_demo.py`)

- **Purpose**: Demonstrate core `InferenceServer` with 3D trajectories
- **Key Insight**: 3D coordinates work as "tokens" without modification
- **Features**: Production server setup, client examples, batch processing
- **Status**: ✅ Clean, direct platform usage

#### 8. **Simple Deployment** (`deploy.py`)

- **Purpose**: Production model deployment using core platform features
- **Features**: Model optimization, metadata generation, Docker files, startup scripts
- **Philosophy**: If advanced deployment features needed, add to core platform
- **Status**: ✅ Minimal, focused deployment utilities

### Key Architecture Insights

The critical discoveries that make 3D trajectory learning work seamlessly:

**1. Natural Multi-Dimensional Support**

- The existing `MoESequenceRegressor` naturally supports 3D coordinates
- Input/output dimensions scale from `[batch, seq, 1]` to `[batch, seq, 3]` without changes
- Expert routing works identically across dimensions
- All optimization and caching features work automatically

**2. Zero-Overhead Scaling**

- No custom architectures or adapters needed for 3D data
- Standard checkpoint format works directly
- All advanced features (attention, position encoding) scale naturally
- Multi-dimensional prediction requires no special handling

**3. Expert Specialization in 3D**

- Experts naturally specialize on spatial regions and motion types
- Routing algorithms work identically for 3D coordinate prediction
- Load balancing maintains effectiveness across higher dimensions

### Core Architecture Details

The system uses the main MoE implementation from `m/moe.py` with:

- **Input Dimension**: 3 (x, y, z coordinates per time step)
- **Output Dimension**: 3 (3D position predictions per time step)
- **Sequence Modeling**: Causal attention with RoPE position encoding
- **Modern Components**: RMSNorm, scaled initialization
- **Expert Routing**: TopK routing algorithms optimized for 3D trajectories

### Data Flow

1. **Training**: `datasets.py` → `training.py` → `m.moe` → checkpoints
2. **Inference**: Checkpoint → `m.inference_server` (core) → 3D trajectory predictions
3. **Evaluation**: `evaluation.py` → comprehensive 3D analysis + visualizations
4. **Comparison**: `run_experiments.py` → multi-config reports

## Customization

### Adding New Trajectory Types

Extend `datasets.py` to add new 3D trajectory generation logic:

```python
def _generate_custom_trajectory(self) -> np.ndarray:
    # Your custom 3D trajectory generation logic
    # Return: np.ndarray of shape [time_steps, 3]
    pass
```

### Modifying Training

Create new config files in `configs/` directory with different:

- Model architectures (layers, dimensions, experts)
- Training hyperparameters
- Dataset parameters
- 3D trajectory type weightings

### Custom Evaluation

Extend `evaluation.py` to add 3D-specific metrics:

```python
def _compute_custom_3d_metric(self, targets, predictions):
    # Your custom 3D trajectory evaluation logic
    return metric_value
```

## Performance Tips

1. **Start Small**: Use `quick_test_3d.yaml` for initial experiments
2. **Monitor 3D Metrics**: Watch position/velocity errors across trajectory types
3. **Check 3D Visualizations**: Examine prediction plots to verify trajectory learning
4. **Expert Utilization**: Ensure balanced usage across different motion types
5. **Temporal Consistency**: Look for smooth, physically plausible predictions

## Research Applications

This example is ideal for:

- **MoE Architecture Research**: Compare routing algorithms for multi-dimensional data
- **3D Motion Learning**: Extend to real-world trajectory datasets
- **Spatial Expert Specialization**: Study how experts specialize in 3D space
- **Physics-Informed Learning**: Add physical constraints to trajectory prediction
- **Multi-Modal Learning**: Combine vision + 3D trajectory prediction

## Troubleshooting

### Common Issues

**Training Loss Not Decreasing**

- Check learning rate (try 0.0001-0.01 range)
- Verify gradient clipping (1.0 is usually good)
- Monitor expert utilization across trajectory types

**3D Predictions Look Random**

- Increase model capacity (more layers/experts)
- Extend training steps
- Check trajectory generation (should be smooth 3D curves)

**Expert Utilization Unbalanced**

- Increase load balancing weight
- Check trajectory type distribution in dataset
- Reduce number of experts if too many

**Out of Memory with 3D Data**

- Reduce batch size (3D requires 3x memory vs 1D)
- Reduce model dimensions
- Use gradient accumulation

## Next Steps

After running these experiments, consider:

1. **Real 3D Data**: Replace synthetic trajectories with real-world motion capture
2. **Higher Dimensions**: Extend to 4D+ (position + velocity + acceleration)
3. **Longer Contexts**: Test with longer 3D sequence lengths
4. **Physics Integration**: Add physical constraints and laws of motion
5. **Multi-Modal**: Combine vision with 3D trajectory prediction

## Citation

If you use this example in research, please cite:

```bibtex
@misc{trajectory_3d_moe,
  title={3D Trajectory Learning with Mixture of Experts Transformers},
  author={Your Name},
  year={2024},
  howpublished={\\url{https://github.com/your-repo/m}}
}
```
