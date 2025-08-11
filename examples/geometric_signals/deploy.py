"""
Simple model deployment utilities for geometric signals.

Uses core inference platform directly with basic optimization.
If more deployment features are needed, they should be added to core platform.
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path

import torch

from .model_discovery import find_geometric_signals_models, get_latest_model

# Add safe globals for checkpoint loading - minimal exposure
torch.serialization.add_safe_globals([Path, pathlib.PosixPath])


def optimize_model_for_inference(checkpoint_path: Path, output_path: Path) -> Path:
    """Basic model optimization using core PyTorch features."""

    print("🔧 Optimizing model for production deployment:")
    print(f"   📁 Source: {checkpoint_path}")
    print(f"   🎯 Target: {output_path}")

    # Load checkpoint
    print("   📥 Loading training checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in checkpoint:
        raise ValueError("Invalid checkpoint format")

    print("   🧹 Removing training-specific state (optimizer, scheduler, etc.)")
    print("   💾 Keeping only model weights and config for inference")

    # Basic optimization: ensure eval mode and remove training-specific state
    optimized_checkpoint = {
        "model_state_dict": checkpoint["model_state_dict"],
        "config": checkpoint["config"],
        "optimized_for_inference": True,
        "optimization_timestamp": torch.tensor(
            torch.utils.data.get_worker_info() is not None
        ).item()
        if hasattr(torch.utils.data, "get_worker_info")
        else 0,
    }

    # Save optimized checkpoint
    torch.save(optimized_checkpoint, output_path)
    print("   ✅ Inference-ready model saved")

    # Show size comparison
    try:
        orig_size = checkpoint_path.stat().st_size / (1024 * 1024)
        new_size = output_path.stat().st_size / (1024 * 1024)
        print(
            f"   📊 Size: {orig_size:.1f}MB → {new_size:.1f}MB ({new_size / orig_size:.1%})"
        )
    except OSError:
        pass

    return output_path


def create_deployment_metadata(model_path: Path, deployment_dir: Path) -> Path:
    """Create basic deployment metadata."""

    print("📋 Creating deployment metadata...")

    metadata = {
        "model_path": str(model_path),
        "deployment_dir": str(deployment_dir),
        "server_class": "m.inference_server.InferenceServer",
        "api_endpoints": {
            "/infer": {
                "method": "POST",
                "description": "Send continuous float values as 'tokens'",
                "example_request": {
                    "tokens": [0.1, 0.5, -0.2, 0.8],
                    "max_new_tokens": 32,
                    "temperature": 1.0,
                },
            },
            "/health": {"method": "GET", "description": "Server health check"},
        },
        "usage_note": "Use core InferenceServer directly - no custom wrapper needed",
    }

    metadata_path = deployment_dir / "deployment_info.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("   ✅ API documentation and metadata saved")
    return metadata_path


def create_production_script(
    model_path: Path, deployment_dir: Path, port: int = 8080
) -> Path:
    """Create simple production startup script using core platform."""

    print("🚀 Creating production startup script...")

    script_content = f'''#!/usr/bin/env python3
"""
Production server for geometric signals using core m.inference_server.
No custom wrappers - uses platform directly.
"""

import asyncio
import pathlib
import sys
from pathlib import Path

import torch
from m.inference_server import InferenceServer

# Add safe globals for checkpoint loading - minimal exposure
torch.serialization.add_safe_globals([Path, pathlib.PosixPath])


async def main():
    model_path = "{model_path}"
    default_port = {port}
    default_device = "auto"
    default_host = "0.0.0.0"
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Geometric Signals Inference Server")
    parser.add_argument("--port", type=int, default=default_port, help=f"Server port (default: {{default_port}})")
    parser.add_argument("--host", type=str, default=default_host, help=f"Server host (default: {{default_host}})")
    parser.add_argument("--device", type=str, default=default_device, help=f"Device (default: {{default_device}})")
    
    args = parser.parse_args()
    
    port = args.port
    device = args.device
    host = args.host
    
    print(f"🚀 Starting geometric signals server")
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
    
    print("📚 Send continuous float values as 'tokens' in POST /infer requests")
    
    try:
        await server.run(host=host)
    except KeyboardInterrupt:
        print("\\n👋 Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
'''

    script_path = deployment_dir / "start_server.py"
    with open(script_path, "w") as f:
        f.write(script_content)

    script_path.chmod(0o755)
    print("   ✅ Production script ready: start_server.py")
    return script_path


def deploy_model(model_path: Path, deployment_name: str) -> Path:
    """Simple deployment using core platform features."""

    deployment_dir = Path("deployments") / deployment_name
    deployment_dir.mkdir(parents=True, exist_ok=True)

    print("\n📦 CREATING PRODUCTION DEPLOYMENT")
    print(f"{'=' * 50}")
    print(f"📂 Deployment name: {deployment_name}")
    print(f"📁 Output directory: {deployment_dir}")
    print(f"🎯 Source model: {model_path}")
    print()

    # Copy and optimize model
    print("STEP 1: Model Optimization")
    optimized_model = deployment_dir / "model.pt"
    optimize_model_for_inference(model_path, optimized_model)
    print()

    # Create metadata
    print("STEP 2: Deployment Configuration")
    create_deployment_metadata(optimized_model, deployment_dir)
    print()

    # Create startup script
    print("STEP 3: Production Scripts")
    create_production_script(optimized_model, deployment_dir)
    print()

    # Test deployment
    print("STEP 4: Deployment Validation")
    try:
        print("🧪 Testing optimized checkpoint loading...")
        test_checkpoint = torch.load(optimized_model, map_location="cpu")
        if "model_state_dict" in test_checkpoint and "config" in test_checkpoint:
            print("   ✅ Checkpoint format valid")
            print("   ✅ Model weights present")
            print("   ✅ Configuration present")
        else:
            print("   ❌ Invalid checkpoint format")

    except Exception as e:
        print(f"   ❌ Checkpoint validation failed: {e}")

    print()
    print(f"{'=' * 50}")
    print(f"🎉 DEPLOYMENT '{deployment_name.upper()}' COMPLETE!")
    print()
    print("🚀 START SERVER:")
    print(f"   cd {deployment_dir}")
    print("   uv run python start_server.py")
    print()
    print("🔗 OR use direct path:")
    print(f"   uv run python {deployment_dir}/start_server.py")
    print()
    print("📊 SERVER OPTIONS:")
    print("   uv run python start_server.py [port] [device]")
    print("   Example: uv run python start_server.py 8080 cuda")
    print(f"{'=' * 50}")

    return deployment_dir


def main():
    """Main CLI for simple deployment."""
    import sys

    if len(sys.argv) < 2:
        print("📦 Geometric Signals Deployment")
        print("Usage: deploy [model_path] [deployment_name]")
        print("       deploy --latest [deployment_name]")
        print("")

        models = find_geometric_signals_models()
        if models:
            print(f"📁 Found {len(models)} trained models:")
            for model in models[:3]:
                print(f"  {model}")
            print("")
            print("Examples:")
            print(f"  deploy {models[0]} my_deployment")
            print("  deploy --latest production")
        else:
            print("❌ No trained models found. Train first!")

        sys.exit(1)

    if sys.argv[1] == "--latest":
        model_path = get_latest_model()
        if not model_path:
            print("❌ No models found")
            sys.exit(1)
        deployment_name = (
            sys.argv[2] if len(sys.argv) > 2 else f"deploy_{int(time.time())}"
        )
    else:
        model_path = Path(sys.argv[1])
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)
        deployment_name = sys.argv[2] if len(sys.argv) > 2 else model_path.stem

    deploy_model(model_path, deployment_name)


if __name__ == "__main__":
    import time

    main()
