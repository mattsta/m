"""
Model Discovery and Management for 3D Trajectory Learning

Dynamic model finding, metadata extraction, and interactive selection utilities
for managing trained trajectory models across experiments and configurations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass(slots=True, frozen=True)
class TrajectoryModelInfo:
    """Container for trajectory model metadata and information."""

    model_path: Path
    config_path: Path | None
    experiment_name: str
    config_name: str

    # Model architecture info
    parameters: int
    n_layers: int
    n_experts: int
    d_model: int

    # Training info
    training_steps: int | None
    best_val_loss: float | None
    final_train_loss: float | None

    # File info
    model_size_mb: float
    created_time: str
    modified_time: str


def find_trajectory_3d_models(
    base_dir: Path | str = "outputs/trajectory_3d", include_metadata: bool = True
) -> list[TrajectoryModelInfo]:
    """
    Find all trained 3D trajectory models with optional metadata extraction.

    Args:
        base_dir: Base directory to search for models
        include_metadata: Whether to extract detailed model metadata

    Returns:
        List of TrajectoryModelInfo objects, sorted by modification time (newest first)
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    models = []

    # Search through experiment directories
    for experiment_dir in base_path.iterdir():
        if not experiment_dir.is_dir():
            continue

        # Look for model files
        for model_name in ["best_model.pt", "final_model.pt"]:
            model_path = experiment_dir / model_name
            if not model_path.exists():
                continue

            try:
                if include_metadata:
                    model_info = _extract_model_metadata(model_path, experiment_dir)
                else:
                    model_info = _basic_model_info(model_path, experiment_dir)

                if model_info:
                    models.append(model_info)

            except Exception as e:
                print(f"Warning: Could not process {model_path}: {e}")
                continue

    # Sort by modification time (newest first)
    models.sort(key=lambda x: x.model_path.stat().st_mtime, reverse=True)

    return models


def _extract_model_metadata(
    model_path: Path, experiment_dir: Path
) -> TrajectoryModelInfo | None:
    """Extract detailed metadata from a trajectory model."""
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Extract model architecture info
        if "model_config" in checkpoint:
            model_config = checkpoint["model_config"]
            n_layers = model_config.n_layers
            n_experts = model_config.block.moe.router.n_experts
            d_model = model_config.block.moe.d_model
        else:
            # Fallback: analyze model state dict
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            n_layers = _count_layers_from_state_dict(state_dict)
            n_experts = _count_experts_from_state_dict(state_dict)
            d_model = _extract_d_model_from_state_dict(state_dict)

        # Count parameters
        if "model_state_dict" in checkpoint:
            parameters = sum(p.numel() for p in checkpoint["model_state_dict"].values())
        else:
            parameters = sum(
                p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor)
            )

        # Extract training info
        training_steps = checkpoint.get("step")
        best_val_loss = checkpoint.get("best_val_loss")

        # Try to get final train loss from metrics history
        final_train_loss = None
        if "metrics_history" in checkpoint and checkpoint["metrics_history"].get(
            "train_loss"
        ):
            final_train_loss = checkpoint["metrics_history"]["train_loss"][-1]

        # Find associated config
        config_path = _find_associated_config(experiment_dir)
        config_name = config_path.name if config_path else "unknown"

        # File info
        stat = model_path.stat()
        model_size_mb = stat.st_size / (1024 * 1024)
        created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime))
        modified_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
        )

        return TrajectoryModelInfo(
            model_path=model_path,
            config_path=config_path,
            experiment_name=experiment_dir.name,
            config_name=config_name,
            parameters=parameters,
            n_layers=n_layers,
            n_experts=n_experts,
            d_model=d_model,
            training_steps=training_steps,
            best_val_loss=best_val_loss,
            final_train_loss=final_train_loss,
            model_size_mb=model_size_mb,
            created_time=created_time,
            modified_time=modified_time,
        )

    except Exception as e:
        print(f"Warning: Could not extract metadata from {model_path}: {e}")
        return _basic_model_info(model_path, experiment_dir)


def _basic_model_info(model_path: Path, experiment_dir: Path) -> TrajectoryModelInfo:
    """Create basic model info without deep metadata extraction."""
    config_path = _find_associated_config(experiment_dir)
    config_name = config_path.name if config_path else "unknown"

    stat = model_path.stat()
    model_size_mb = stat.st_size / (1024 * 1024)
    created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime))
    modified_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))

    return TrajectoryModelInfo(
        model_path=model_path,
        config_path=config_path,
        experiment_name=experiment_dir.name,
        config_name=config_name,
        parameters=0,  # Unknown
        n_layers=0,  # Unknown
        n_experts=0,  # Unknown
        d_model=0,  # Unknown
        training_steps=None,
        best_val_loss=None,
        final_train_loss=None,
        model_size_mb=model_size_mb,
        created_time=created_time,
        modified_time=modified_time,
    )


def _find_associated_config(experiment_dir: Path) -> Path | None:
    """Find the configuration file associated with an experiment."""

    # Check for training summary which might contain config info
    summary_path = experiment_dir / "training_summary.yaml"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = yaml.safe_load(f)
                if (
                    "training_config" in summary
                    and "output_dir" in summary["training_config"]
                ):
                    # This might help us infer the config, but for now just look in configs dir
                    pass
        except Exception:
            pass

    # Look for config files in the trajectory_3d configs directory
    config_dir = Path(__file__).parent / "configs"
    if config_dir.exists():
        # Try to match config names with experiment names
        for config_file in config_dir.glob("*.yaml"):
            if config_file.stem in experiment_dir.name:
                return config_file

        # Default to quick_test_3d.yaml if nothing matches
        quick_test = config_dir / "quick_test_3d.yaml"
        if quick_test.exists():
            return quick_test

    return None


def _count_layers_from_state_dict(state_dict: dict) -> int:
    """Count number of layers from model state dict."""
    layer_keys = [k for k in state_dict.keys() if "layers." in k]
    if not layer_keys:
        return 0

    # Extract layer numbers
    layer_numbers = set()
    for key in layer_keys:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_numbers.add(int(parts[i + 1]))
                except ValueError:
                    continue

    return len(layer_numbers)


def _count_experts_from_state_dict(state_dict: dict) -> int:
    """Count number of experts from model state dict."""
    expert_keys = [k for k in state_dict.keys() if "expert." in k and "weight" in k]
    if not expert_keys:
        return 0

    # Extract expert numbers
    expert_numbers = set()
    for key in expert_keys:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part == "expert" and i + 1 < len(parts):
                try:
                    expert_numbers.add(int(parts[i + 1]))
                except ValueError:
                    continue

    return len(expert_numbers)


def _extract_d_model_from_state_dict(state_dict: dict) -> int:
    """Extract d_model dimension from model state dict."""
    # Look for embedding or projection weights
    for key, tensor in state_dict.items():
        if ("embed" in key or "proj" in key) and tensor.dim() >= 2:
            return tensor.shape[-1]

    # Fallback: look for any reasonable-sized weight tensor
    for key, tensor in state_dict.items():
        if tensor.dim() >= 2 and "weight" in key:
            dims = tensor.shape
            for dim in dims:
                if 64 <= dim <= 4096:  # Reasonable d_model range
                    return dim

    return 0  # Unknown


def get_latest_trajectory_model() -> TrajectoryModelInfo | None:
    """Get the most recently modified trajectory model."""
    models = find_trajectory_3d_models(
        include_metadata=False
    )  # Faster without full metadata
    return models[0] if models else None


def select_trajectory_model_interactively() -> TrajectoryModelInfo | None:
    """Interactively select a trajectory model from available options."""
    models = find_trajectory_3d_models()

    if not models:
        print("❌ No trained trajectory models found.")
        return None

    print("📊 Available 3D Trajectory Models:")
    print("-" * 80)

    for i, model in enumerate(models, 1):
        print(f"{i:2d}. {model.experiment_name} ({model.model_path.name})")
        print(f"    📁 {model.model_path}")
        print(f"    ⚙️  Config: {model.config_name}")
        if model.parameters > 0:
            print(
                f"    🧠 {model.parameters:,} parameters ({model.n_layers} layers, {model.n_experts} experts)"
            )
        if model.best_val_loss is not None:
            print(f"    📈 Best val loss: {model.best_val_loss:.6f}")
        print(f"    📅 Modified: {model.modified_time}")
        print()

    while True:
        try:
            choice = input(f"Select model [1-{len(models)}] or 'q' to quit: ").strip()
            if choice.lower() in ("q", "quit", "exit"):
                return None

            index = int(choice) - 1
            if 0 <= index < len(models):
                selected = models[index]
                print(
                    f"✅ Selected: {selected.experiment_name} ({selected.model_path.name})"
                )
                return selected
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(models)}")

        except ValueError:
            print("❌ Invalid input. Please enter a number or 'q'")
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return None


def print_model_details(model: TrajectoryModelInfo) -> None:
    """Print detailed information about a trajectory model."""
    print("=" * 60)
    print(f"3D Trajectory Model: {model.experiment_name}")
    print("=" * 60)
    print(f"📁 Model Path: {model.model_path}")
    print(f"⚙️  Config: {model.config_name}")
    if model.config_path:
        print(f"📋 Config Path: {model.config_path}")

    print("\n🏗️  Architecture:")
    if model.parameters > 0:
        print(f"   Parameters: {model.parameters:,}")
        print(f"   Layers: {model.n_layers}")
        print(f"   Experts: {model.n_experts}")
        print(f"   Model Dimension: {model.d_model}")
    else:
        print("   (Architecture details not available)")

    print("\n📊 Training:")
    if model.training_steps is not None:
        print(f"   Training Steps: {model.training_steps:,}")
    if model.best_val_loss is not None:
        print(f"   Best Validation Loss: {model.best_val_loss:.6f}")
    if model.final_train_loss is not None:
        print(f"   Final Training Loss: {model.final_train_loss:.6f}")

    print("\n📁 File Info:")
    print(f"   Size: {model.model_size_mb:.2f} MB")
    print(f"   Created: {model.created_time}")
    print(f"   Modified: {model.modified_time}")


def main():
    """Main CLI entrypoint for trajectory model discovery and management."""
    import sys

    if len(sys.argv) < 2:
        print("🎯 3D Trajectory Model Discovery")
        print("Usage: trajectory-3d-models <command>")
        print("")
        print("Commands:")
        print("  list                     # List all available models")
        print("  latest                   # Show latest model")
        print("  details                  # Show detailed model information")
        print("  interactive              # Interactive model selection")
        print("")

        # Show quick model count
        models = find_trajectory_3d_models(include_metadata=False)
        if models:
            print(f"📊 Found {len(models)} trained trajectory models")
            latest = models[0]
            print(f"📅 Latest: {latest.experiment_name} ({latest.modified_time})")
        else:
            print("❌ No trained models found. Train a model first!")

        sys.exit(1)

    command = sys.argv[1].lower()

    if command in ("list", "ls"):
        models = find_trajectory_3d_models()
        if not models:
            print("❌ No trained trajectory models found")
            sys.exit(1)

        print(f"📊 Found {len(models)} 3D Trajectory Models:")
        print("-" * 90)
        print(
            f"{'Experiment':<25} {'Model':<15} {'Parameters':<12} {'Val Loss':<10} {'Modified'}"
        )
        print("-" * 90)

        for model in models:
            params_str = f"{model.parameters:,}" if model.parameters > 0 else "Unknown"
            val_loss_str = (
                f"{model.best_val_loss:.4f}"
                if model.best_val_loss is not None
                else "N/A"
            )

            print(
                f"{model.experiment_name:<25} {model.model_path.name:<15} {params_str:<12} {val_loss_str:<10} {model.modified_time}"
            )

    elif command == "latest":
        latest = get_latest_trajectory_model()
        if not latest:
            print("❌ No trained models found")
            sys.exit(1)

        print("📅 Latest 3D Trajectory Model:")
        print_model_details(latest)
        print(f"\n✅ Model path: {latest.model_path}")

    elif command == "details":
        models = find_trajectory_3d_models()
        if not models:
            print("❌ No trained models found")
            sys.exit(1)

        if len(sys.argv) > 2:
            # Show details for specific model by experiment name
            experiment_name = sys.argv[2]
            model = next(
                (m for m in models if m.experiment_name == experiment_name), None
            )
            if not model:
                print(f"❌ Model not found: {experiment_name}")
                available = [m.experiment_name for m in models]
                print(f"Available models: {', '.join(available)}")
                sys.exit(1)
        else:
            # Show details for latest model
            model = models[0]

        print_model_details(model)

    elif command == "interactive":
        selected = select_trajectory_model_interactively()
        if selected:
            print_model_details(selected)
            print(f"\n✅ Selected model path: {selected.model_path}")
        else:
            print("❌ No model selected")
            sys.exit(1)

    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: list, latest, details, interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
