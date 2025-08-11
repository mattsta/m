"""
Dynamic model discovery utilities for geometric signals.
Automatically finds trained models without hardcoded paths.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def find_geometric_signals_models(limit: int | None = None) -> list[Path]:
    """Find all available geometric signal model checkpoints.

    Args:
        limit: Maximum number of models to return (None for all)

    Returns:
        List of model checkpoint paths sorted by modification time (newest first)
    """
    # Look for outputs directory from multiple possible locations
    current_dir = Path.cwd()
    script_dir = Path(__file__).parent

    possible_output_dirs = [
        Path("outputs/geometric_signals"),  # From repo root
        Path("../../outputs/geometric_signals"),  # From examples/geometric_signals/
        Path(
            "../../../outputs/geometric_signals"
        ),  # From examples/geometric_signals/subdir/
        script_dir / "../../outputs/geometric_signals",  # Relative to this file
        Path(
            "/Users/matt/repos/m/outputs/geometric_signals"
        ),  # Absolute fallback (temporary)
        # Find repo root by looking for specific files
        current_dir / "outputs/geometric_signals",
        current_dir.parent / "outputs/geometric_signals",
        current_dir.parent.parent / "outputs/geometric_signals",
    ]

    outputs_dir = None
    for dir_path in possible_output_dirs:
        if dir_path.exists() and dir_path.is_dir():
            outputs_dir = dir_path.resolve()
            break

    if outputs_dir is None:
        return []

    models = []

    # Search all experiment subdirectories
    for experiment_dir in outputs_dir.iterdir():
        if not experiment_dir.is_dir():
            continue

        # Look for common checkpoint names (in priority order)
        checkpoint_names = [
            "final_model.pt",  # Final trained model
            "best_model.pt",  # Best validation model
            "latest.pt",  # Latest checkpoint
        ]

        for checkpoint_name in checkpoint_names:
            checkpoint_path = experiment_dir / checkpoint_name
            if checkpoint_path.exists():
                try:
                    # Verify it's a valid PyTorch checkpoint
                    checkpoint = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=False
                    )
                    if "model_state_dict" in checkpoint and "config" in checkpoint:
                        models.append(checkpoint_path)
                        break  # Only take the first valid checkpoint per experiment
                except Exception:
                    # Skip invalid checkpoints
                    continue

    # Sort by modification time (newest first)
    models.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if limit is not None:
        models = models[:limit]

    return models


def get_model_info(checkpoint_path: Path) -> dict[str, Any]:
    """Get information about a model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary with model information
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Basic file info
        stat = checkpoint_path.stat()

        info = {
            "path": checkpoint_path,
            "experiment_name": checkpoint_path.parent.name,
            "checkpoint_name": checkpoint_path.name,
            "size_mb": stat.st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
        }

        # Model configuration info
        if "config" in checkpoint:
            config = checkpoint["config"]
            if hasattr(config, "model") and hasattr(config.model, "get"):
                model_config = config.model
                info.update(
                    {
                        "n_layers": model_config.get("n_layers", "unknown"),
                        "input_dim": model_config.get("input_dim", "unknown"),
                        "target_dim": model_config.get("target_dim", "unknown"),
                        "pool_mode": model_config.get("pool", "unknown"),
                    }
                )

                # MoE specific info
                if "block" in model_config and "moe" in model_config["block"]:
                    moe_config = model_config["block"]["moe"]
                    if "router" in moe_config:
                        router_config = moe_config["router"]
                        info.update(
                            {
                                "n_experts": router_config.get("n_experts", "unknown"),
                                "router_type": router_config.get(
                                    "router_type", "unknown"
                                ),
                            }
                        )

                    info["d_model"] = moe_config.get("d_model", "unknown")

        # Training info
        if "step" in checkpoint:
            info["training_step"] = checkpoint["step"]
        if "epoch" in checkpoint:
            info["training_epoch"] = checkpoint["epoch"]
        if "best_val_loss" in checkpoint:
            info["best_val_loss"] = checkpoint["best_val_loss"]

        return info

    except Exception as e:
        return {"path": checkpoint_path, "error": str(e), "corrupted": True}


def list_available_models(show_details: bool = False) -> None:
    """List all available models with optional details.

    Args:
        show_details: Whether to show detailed model information
    """
    models = find_geometric_signals_models()

    if not models:
        print("❌ No trained models found!")
        print("   Train models using the training pipeline first")
        return

    print(f"📁 Found {len(models)} trained geometric signal models:")
    print()

    for i, model_path in enumerate(models, 1):
        if show_details:
            info = get_model_info(model_path)
            print(f"{i:2d}. {model_path}")
            print(f"     Experiment: {info.get('experiment_name', 'unknown')}")
            print(f"     Size: {info.get('size_mb', 0):.1f} MB")
            print(f"     Modified: {info.get('modified', 'unknown')}")
            if "n_experts" in info:
                print(
                    f"     Architecture: {info.get('n_layers', '?')} layers, {info.get('n_experts', '?')} experts"
                )
            if "training_step" in info:
                print(
                    f"     Training: Step {info.get('training_step', '?')}, Loss {info.get('best_val_loss', '?'):.4f}"
                )
            print()
        else:
            print(f"{i:2d}. {model_path}")


def get_latest_model() -> Path | None:
    """Get the most recently trained model.

    Returns:
        Path to the latest model, or None if no models found
    """
    models = find_geometric_signals_models(limit=1)
    return models[0] if models else None


def select_model_interactively() -> Path | None:
    """Interactively select a model from available options.

    Returns:
        Selected model path, or None if cancelled
    """
    models = find_geometric_signals_models()

    if not models:
        print("❌ No trained models available for selection")
        return None

    if len(models) == 1:
        print(f"🎯 Only one model available: {models[0]}")
        return models[0]

    print(f"📁 Select from {len(models)} available models:")
    print()

    for i, model_path in enumerate(models, 1):
        info = get_model_info(model_path)
        print(f"{i:2d}. {model_path.parent.name}/{model_path.name}")
        print(f"     Modified: {info.get('modified', 'unknown')}")
        if "n_experts" in info:
            print(
                f"     {info.get('n_experts', '?')} experts, {info.get('training_step', '?')} steps"
            )

    print()

    while True:
        try:
            choice = (
                input(f"Select model (1-{len(models)}, or 'q' to quit): ")
                .strip()
                .lower()
            )

            if choice == "q" or choice == "quit":
                return None

            index = int(choice) - 1
            if 0 <= index < len(models):
                selected = models[index]
                print(f"✅ Selected: {selected}")
                return selected
            else:
                print(f"❌ Invalid selection. Please choose 1-{len(models)}")

        except (ValueError, KeyboardInterrupt):
            print("\n👋 Selection cancelled")
            return None


def main():
    """Main CLI entrypoint for model discovery."""
    import sys

    if "--help" in sys.argv or "-h" in sys.argv:
        print("🌊 Geometric Signals Model Discovery")
        print("Usage: signals-models [OPTIONS]")
        print("")
        print("Options:")
        print("  --details       Show detailed model information")
        print("  --interactive   Interactively select a model")
        print("  --latest        Show only the latest model")
        print("  --help, -h      Show this help message")
        print("")
        print("Examples:")
        print("  signals-models")
        print("  signals-models --details")
        print("  signals-models --interactive")
        return

    if "--latest" in sys.argv:
        latest = get_latest_model()
        if latest:
            print(f"🎯 Latest model: {latest}")
            info = get_model_info(latest)
            print(f"   Experiment: {info.get('experiment_name', 'unknown')}")
            print(f"   Modified: {info.get('modified', 'unknown')}")
            print(f"   Size: {info.get('size_mb', 0):.1f} MB")
        else:
            print("❌ No models found")
        return

    if "--details" in sys.argv:
        list_available_models(show_details=True)
    else:
        list_available_models(show_details=False)

    if "--interactive" in sys.argv:
        selected = select_model_interactively()
        if selected:
            print(f"✅ Selected model: {selected}")


if __name__ == "__main__":
    main()
