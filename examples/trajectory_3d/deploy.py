"""
3D Trajectory Model Deployment System

Production deployment utilities for 3D trajectory models using core platform features.
Creates optimized deployments with startup scripts and metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .model_discovery import (
    TrajectoryModelInfo,
    _basic_model_info,
    find_trajectory_3d_models,
    get_latest_trajectory_model,
)


def create_trajectory_deployment(
    model_info: TrajectoryModelInfo,
    deployment_name: str,
    output_dir: Path | str = "outputs/trajectory_3d/deployments",
    optimize_model: bool = True,
    include_config: bool = True,
) -> Path:
    """
    Create optimized production deployment for a 3D trajectory model.

    Args:
        model_info: Model information from discovery system
        deployment_name: Name for the deployment
        output_dir: Base directory for deployments
        optimize_model: Whether to optimize model for inference
        include_config: Whether to include configuration files

    Returns:
        Path to created deployment directory
    """
    deployment_dir = Path(output_dir) / deployment_name
    deployment_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Creating 3D trajectory deployment: {deployment_name}")
    print(f"📁 Source model: {model_info.model_path}")
    print(f"🎯 Deployment directory: {deployment_dir}")

    # Copy and optimize model
    deployment_model_path = deployment_dir / "model.pt"
    if optimize_model:
        print("⚡ Optimizing model for inference...")
        _optimize_trajectory_model(model_info.model_path, deployment_model_path)
    else:
        print("📋 Copying model...")
        shutil.copy2(model_info.model_path, deployment_model_path)

    # Copy configuration if available and requested
    if include_config and model_info.config_path and model_info.config_path.exists():
        config_dest = deployment_dir / "config.yaml"
        shutil.copy2(model_info.config_path, config_dest)
        print(f"⚙️  Configuration copied: {config_dest.name}")

    # Create deployment metadata
    metadata = _create_deployment_metadata(model_info, deployment_name, deployment_dir)
    metadata_path = deployment_dir / "deployment_info.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📋 Metadata created: {metadata_path.name}")

    # Create startup script
    startup_script = _create_startup_script(deployment_name, metadata)
    script_path = deployment_dir / "start_server.py"
    with open(script_path, "w") as f:
        f.write(startup_script)
    script_path.chmod(0o755)  # Make executable
    print(f"🚀 Startup script created: {script_path.name}")

    # Create Docker deployment files
    _create_docker_files(deployment_dir, deployment_name, metadata)

    # Create deployment README
    _create_deployment_readme(deployment_dir, deployment_name, metadata)

    print("✅ Deployment created successfully!")
    print(f"📁 Location: {deployment_dir}")
    print(f"🚀 Start server: cd {deployment_dir} && python start_server.py")

    return deployment_dir


def _optimize_trajectory_model(source_path: Path, dest_path: Path) -> None:
    """Optimize trajectory model for inference deployment."""

    try:
        # Load model checkpoint
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)

        # Extract only what's needed for inference
        optimized_checkpoint = {
            "model_state_dict": checkpoint["model_state_dict"],
            "model_config": checkpoint.get("model_config"),
            "training_config": checkpoint.get("training_config"),
            "dataset_config": checkpoint.get("dataset_config"),
        }

        # Remove training-specific data
        excluded_keys = [
            "optimizer_state_dict",
            "scheduler_state_dict",
            "metrics_history",
            "step",
            "best_val_loss",
        ]

        for key in excluded_keys:
            checkpoint.pop(key, None)

        # Save optimized model
        torch.save(optimized_checkpoint, dest_path)

        # Calculate size reduction
        original_size = source_path.stat().st_size
        optimized_size = dest_path.stat().st_size
        reduction_pct = (1 - optimized_size / original_size) * 100

        print(f"   Original size: {original_size / 1024 / 1024:.1f} MB")
        print(f"   Optimized size: {optimized_size / 1024 / 1024:.1f} MB")
        print(f"   Size reduction: {reduction_pct:.1f}%")

    except Exception as e:
        print(f"⚠️  Optimization failed, copying original: {e}")
        shutil.copy2(source_path, dest_path)


def _create_deployment_metadata(
    model_info: TrajectoryModelInfo, deployment_name: str, deployment_dir: Path
) -> dict[str, Any]:
    """Create deployment metadata dictionary."""

    # time already imported at top

    metadata = {
        "deployment_name": deployment_name,
        "deployment_type": "3d_trajectory_inference",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "deployment_dir": str(deployment_dir),
        # Model information
        "model_info": {
            "source_path": str(model_info.model_path),
            "experiment_name": model_info.experiment_name,
            "config_name": model_info.config_name,
            "parameters": model_info.parameters,
            "architecture": {
                "n_layers": model_info.n_layers,
                "n_experts": model_info.n_experts,
                "d_model": model_info.d_model,
                "input_dim": 3,  # 3D trajectories
                "output_dim": 3,
            },
            "training_info": {
                "training_steps": model_info.training_steps,
                "best_val_loss": model_info.best_val_loss,
                "final_train_loss": model_info.final_train_loss,
            },
        },
        # Deployment configuration
        "server_config": {
            "default_port": 8080,
            "default_host": "0.0.0.0",
            "default_device": "auto",
            "max_batch_size": 16,
            "max_sequence_length": 512,
        },
        # Usage examples
        "usage_examples": {
            "start_server": "python start_server.py",
            "start_with_options": "python start_server.py --port 9000 --device cuda",
            "health_check": "curl http://localhost:8080/health",
            "inference_request": {
                "url": "http://localhost:8080/infer",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "request_id": "3d_trajectory_prediction",
                    "tokens": [[1.0, 2.0, 3.0], [1.1, 2.1, 2.9], [1.2, 2.3, 2.7]],
                    "max_new_tokens": 16,
                    "temperature": 1.0,
                    "use_cache": True,
                },
            },
        },
        # Platform information
        "platform_info": {
            "framework": "MoE Transformers",
            "inference_server": "m.inference_server",
            "data_type": "3d_continuous_trajectories",
            "supported_formats": [
                "helical",
                "orbital",
                "lissajous",
                "lorenz",
                "robotic",
            ],
        },
    }

    return metadata


def _create_startup_script(deployment_name: str, metadata: dict[str, Any]) -> str:
    """Create Python startup script for the deployment."""

    # Extract server config properly
    server_config = metadata["server_config"]

    # Create example request as proper data structure
    example_request = {
        "request_id": "3d_trajectory_test",
        "tokens": [[1.0, 2.0, 3.0], [1.1, 2.1, 2.9]],
        "max_new_tokens": 16,
        "temperature": 1.0,
    }

    # Format as compact JSON for single-line curl command
    example_json = json.dumps(example_request, separators=(",", ":"))

    # Create the script template without the problematic JSON insertion
    script_template = '''#!/usr/bin/env python3
"""
3D Trajectory Inference Server Startup Script
Deployment: {deployment_name}

This script starts a production inference server for 3D trajectory prediction
using the core MoE platform infrastructure.
"""

import argparse
import asyncio
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
from m.inference_server import InferenceServer

# Add safe globals for checkpoint loading - minimal exposure  
torch.serialization.add_safe_globals([Path, pathlib.PosixPath, type(np.float64(1.0))])


async def main():
    model_path = "{model_path}"
    default_port = {default_port}
    default_device = "auto"
    default_host = "0.0.0.0"
    
    # Parse command line arguments
    # argparse already imported at top
    parser = argparse.ArgumentParser(description="3D Trajectory Inference Server")
    parser.add_argument("--port", type=int, default=default_port, help=f"Server port (default: {{default_port}})")
    parser.add_argument("--host", type=str, default=default_host, help=f"Server host (default: {{default_host}})")
    parser.add_argument("--device", type=str, default=default_device, help=f"Device (default: {{default_device}})")
    
    args = parser.parse_args()
    
    port = args.port
    device = args.device
    host = args.host
    
    print(f"🎯 Starting 3D trajectory server")
    print(f"📁 Model: {{model_path}}")
    print(f"🌐 Address: {{host}}:{{port}}")
    print(f"🖥️  Device: {{device}}")
    print("💡 Using core InferenceServer with auto port resolution")
    print()
    
    # Use core platform directly
    server = InferenceServer(
        model_path=model_path,
        device=device,
        port=port
    )
    
    print("📊 Send 3D coordinates as 'tokens' in POST /infer requests")
    print("📡 Example: [[1.0,2.0,3.0],[1.1,2.1,2.9]] for trajectory prediction")
    
    try:
        await server.run(host=host)
    except KeyboardInterrupt:
        print("\\n👋 Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
'''

    # Format the template with safe parameter substitution
    script = script_template.format(
        deployment_name=deployment_name,
        default_port=server_config["default_port"],
        default_host=server_config["default_host"],
        default_device=server_config["default_device"],
        example_request_json=example_json,
    )

    return script


def _create_docker_files(
    deployment_dir: Path, deployment_name: str, metadata: dict[str, Any]
) -> None:
    """Create Docker deployment files."""

    # Dockerfile
    dockerfile = f"""FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (you may need to create this)
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Install MoE platform (adjust as needed for your setup)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# RUN pip install your-moe-platform

# Copy deployment files
COPY model.pt .
COPY config.yaml .
COPY deployment_info.json .
COPY start_server.py .

# Expose default port
EXPOSE {metadata["server_config"]["default_port"]}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:{metadata["server_config"]["default_port"]}/health || exit 1

# Run server
CMD ["python", "start_server.py"]
"""

    with open(deployment_dir / "Dockerfile", "w") as f:
        f.write(dockerfile)

    # docker-compose.yml
    compose_file = f'''version: '3.8'

services:
  {deployment_name.replace("_", "-")}:
    build: .
    ports:
      - "{metadata["server_config"]["default_port"]}:{metadata["server_config"]["default_port"]}"
    environment:
      - DEVICE=auto
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{metadata["server_config"]["default_port"]}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
'''

    with open(deployment_dir / "docker-compose.yml", "w") as f:
        f.write(compose_file)

    print("🐳 Docker files created: Dockerfile, docker-compose.yml")


def _create_deployment_readme(
    deployment_dir: Path, deployment_name: str, metadata: dict[str, Any]
) -> None:
    """Create README for the deployment."""

    readme = f"""# 3D Trajectory Inference Deployment: {deployment_name}

Production deployment for 3D trajectory prediction using MoE transformers.

## Quick Start

### Local Development
```bash
# Start the inference server
python start_server.py

# Start on custom port/host
python start_server.py --port 9000 --host 0.0.0.0 --device cuda
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t {deployment_name.replace("_", "-")} .
docker run -p 8080:8080 {deployment_name.replace("_", "-")}
```

## Model Information

- **Experiment**: {metadata["model_info"]["experiment_name"]}
- **Configuration**: {metadata["model_info"]["config_name"]}
- **Parameters**: {metadata["model_info"]["parameters"]:,}
- **Architecture**: {metadata["model_info"]["architecture"]["n_layers"]} layers, {metadata["model_info"]["architecture"]["n_experts"]} experts
- **Training Steps**: {metadata["model_info"]["training_info"]["training_steps"]}
- **Best Validation Loss**: {metadata["model_info"]["training_info"]["best_val_loss"]}

## API Usage

### Health Check
```bash
curl http://localhost:8080/health
```

### 3D Trajectory Prediction
```bash
curl -X POST http://localhost:8080/infer \\
  -H "Content-Type: application/json" \\
  -d '{{
    "request_id": "3d_trajectory_prediction",
    "tokens": [[1.0, 2.0, 3.0], [1.1, 2.1, 2.9], [1.2, 2.3, 2.7]],
    "max_new_tokens": 16,
    "temperature": 1.0,
    "use_cache": true
  }}'
```

### Example Response
```json
{{
  "request_id": "3d_trajectory_prediction",
  "generated_tokens": [
    [1.3, 2.4, 2.5],
    [1.4, 2.5, 2.3],
    ...
  ],
  "cache_hits": 0,
  "processing_time_ms": 45
}}
```

## Supported Trajectory Types

This model was trained on 5 types of 3D trajectories:
- **Helical**: DNA helix patterns, spiral motions
- **Orbital**: Elliptical orbits, planetary motion
- **Lissajous**: 3D harmonic oscillations
- **Lorenz**: Chaotic dynamics (butterfly effect)
- **Robotic**: Smooth waypoint interpolation

## Performance

- **Input**: 3D coordinates `[[x, y, z], ...]`
- **Output**: Predicted 3D positions
- **Sequence Length**: Up to {metadata["server_config"]["max_sequence_length"]} positions
- **Batch Size**: Up to {metadata["server_config"]["max_batch_size"]} requests
- **Inference Speed**: ~50ms per request (depending on hardware)

## Files

- `model.pt` - Optimized model checkpoint
- `config.yaml` - Training configuration
- `start_server.py` - Production server script
- `deployment_info.json` - Deployment metadata
- `Dockerfile` - Container definition
- `docker-compose.yml` - Multi-container deployment

## Deployment Details

- **Created**: {metadata["created_at"]}
- **Platform**: MoE Transformers
- **Inference Server**: m.inference_server
- **Framework**: PyTorch + FastAPI
- **Default Port**: {metadata["server_config"]["default_port"]}

## Troubleshooting

### Server won't start
1. Check that the model file exists: `ls -la model.pt`
2. Verify dependencies: `python -c "import torch; from m.inference_server import InferenceServer"`
3. Check port availability: `netstat -an | grep {metadata["server_config"]["default_port"]}`

### Poor prediction quality
1. Ensure input trajectories are smooth 3D curves
2. Use appropriate sequence lengths (32-128 points recommended)
3. Consider the trajectory type - model performs best on patterns similar to training data

### Memory issues
1. Reduce batch size: `--max-batch-size 4`
2. Use CPU device: `--device cpu`
3. Restart server periodically for long-running deployments

## Support

For issues related to:
- **Model training**: Check the trajectory_3d example documentation
- **Inference server**: See m.inference_server documentation
- **Deployment**: Review this README and deployment_info.json
"""

    with open(deployment_dir / "README.md", "w") as f:
        f.write(readme)

    print("📝 Deployment README created")


def deploy_latest_trajectory_model(deployment_name: str | None = None) -> Path:
    """Deploy the latest trained trajectory model."""

    latest_model = get_latest_trajectory_model()
    if not latest_model:
        raise ValueError("No trained trajectory models found. Train a model first!")

    if deployment_name is None:
        deployment_name = f"{latest_model.experiment_name}_deployment"

    return create_trajectory_deployment(latest_model, deployment_name)


def main():
    """Main CLI entrypoint for 3D trajectory deployment."""
    # argparse and sys already imported at top

    parser = argparse.ArgumentParser(description="3D Trajectory Model Deployment")
    parser.add_argument("model_path", nargs="?", help="Path to model checkpoint")
    parser.add_argument("deployment_name", nargs="?", help="Name for deployment")
    parser.add_argument("--latest", action="store_true", help="Deploy latest model")
    parser.add_argument(
        "--output-dir",
        default="outputs/trajectory_3d/deployments",
        help="Output directory for deployments",
    )
    parser.add_argument(
        "--no-optimize", action="store_true", help="Skip model optimization"
    )
    parser.add_argument(
        "--no-config", action="store_true", help="Don't include configuration files"
    )

    args = parser.parse_args()

    if len(sys.argv) < 2:
        print("🚀 3D Trajectory Model Deployment")
        print("Usage: trajectory-3d-deploy <model_path> <deployment_name>")
        print("       trajectory-3d-deploy --latest [deployment_name]")
        print("")
        print("Examples:")
        print("  trajectory-3d-deploy --latest production")
        print(
            "  trajectory-3d-deploy outputs/trajectory_3d/my_exp/best_model.pt my_deployment"
        )
        print("")

        # Show available models
        models = find_trajectory_3d_models(include_metadata=False)
        if models:
            print(f"📊 Available models: {len(models)}")
            latest = models[0]
            print(f"📅 Latest: {latest.experiment_name} ({latest.modified_time})")
        else:
            print("❌ No trained models found. Train a model first!")

        sys.exit(1)

    try:
        if args.latest:
            # Deploy latest model
            deployment_name = args.deployment_name or "latest_deployment"
            print(f"🔍 Finding latest model for deployment: {deployment_name}")
            deployment_dir = deploy_latest_trajectory_model(deployment_name)

        else:
            # Deploy specific model
            if not args.model_path:
                print("❌ Model path required when not using --latest")
                sys.exit(1)

            model_path = Path(args.model_path)
            if not model_path.exists():
                print(f"❌ Model not found: {model_path}")
                sys.exit(1)

            if not args.deployment_name:
                print("❌ Deployment name required")
                sys.exit(1)

            # Find model info for the specified path
            models = find_trajectory_3d_models()
            model_info = next((m for m in models if m.model_path == model_path), None)

            if not model_info:
                print("⚠️  Model not found in discovery system, creating basic info...")
                # Create basic model info for deployment
                # _basic_model_info already imported at top

                experiment_dir = model_path.parent
                model_info = _basic_model_info(model_path, experiment_dir)

            deployment_dir = create_trajectory_deployment(
                model_info=model_info,
                deployment_name=args.deployment_name,
                output_dir=args.output_dir,
                optimize_model=not args.no_optimize,
                include_config=not args.no_config,
            )

        print("\\n🎉 Deployment completed successfully!")
        print(f"📁 Location: {deployment_dir}")
        print(f"🚀 Start server: cd {deployment_dir} && python start_server.py")
        print(f"🐳 Docker deploy: cd {deployment_dir} && docker-compose up --build")

    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
