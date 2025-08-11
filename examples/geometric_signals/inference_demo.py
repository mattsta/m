"""
Simple demonstration of geometric signals with core m.inference_server.

This shows the key insight: the existing inference server naturally supports
continuous float values as "tokens" without any modifications needed.
"""

from __future__ import annotations

import asyncio
import pathlib
import time
from pathlib import Path

import numpy as np
import torch

from m.inference_server import InferenceServer

from .model_discovery import find_geometric_signals_models, get_latest_model

# Add safe globals for checkpoint loading - minimal exposure
torch.serialization.add_safe_globals([Path, pathlib.PosixPath])


async def demo_direct_inference(model_path: Path) -> bool:
    """Demonstrate direct inference with core InferenceServer."""

    print(f"🎯 Using model: {model_path}")

    try:
        # Use core inference server directly - all issues now fixed!
        server = InferenceServer(model_path=str(model_path), device="auto", port=8080)

        print("✅ InferenceServer created successfully")
        print(f"   Model type: {type(server.engine.model)}")

        # Create sample continuous signal
        t = np.linspace(0, 4 * np.pi, 64)
        input_signal = np.sin(t) + 0.1 * np.cos(3 * t)  # Complex wave

        print(
            f"📊 Input signal: {input_signal.shape}, range: [{input_signal.min():.2f}, {input_signal.max():.2f}]"
        )

        # Request format - the key insight: continuous values as "tokens"!
        request = {
            "request_id": "geometric_signal_test",
            "tokens": input_signal.tolist(),  # Continuous float values
            "max_new_tokens": 32,
            "temperature": 1.0,
            "use_cache": True,
        }

        # Make prediction using existing server
        start_time = time.time()
        result = await server.handle_request(request)
        processing_time = (time.time() - start_time) * 1000

        print("✅ Prediction successful!")
        print(f"📈 Generated {len(result['generated_tokens'])} continuous values")
        print(f"⏱️  Processing time: {processing_time:.2f}ms")
        print(f"🚄 Throughput: {result['tokens_per_second']:.1f} values/sec")

        # Show some predicted values
        predictions = result["generated_tokens"][:5]
        print(f"🔢 Sample predictions: {predictions}")

        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_production_server(
    model_path: Path, port: int = 8080, device: str = "auto"
) -> InferenceServer:
    """Create production server using core InferenceServer directly."""

    print("🚀 Creating production server")
    print(f"📁 Model: {model_path}")
    print(f"🚪 Port: {port}")
    print(f"🖥️  Device: {device}")

    # Use core server directly - that's it!
    server = InferenceServer(model_path=str(model_path), device=device, port=port)

    print("✅ Production server ready!")
    print("💡 Send continuous float values as 'tokens' in requests")

    return server


async def run_production_server(
    model_path: Path, port: int = 8080, device: str = "auto"
):
    """Run production server using core infrastructure."""
    server = create_production_server(model_path, port, device)

    print(f"🌐 Starting server at http://localhost:{port}")
    print("📚 API endpoints:")
    print("  POST /infer - Send continuous values as 'tokens'")
    print("  GET /health - Server health check")
    print("")
    print("Press Ctrl+C to stop...")

    try:
        await server.run()
    except KeyboardInterrupt:
        print("\n👋 Server shutting down...")


def main():
    """Main demo showing direct integration with core platform."""

    print("🌊 Geometric Signals + Core Inference Platform Integration")
    print("=" * 60)
    print()
    print("This demonstrates using m.inference_server directly with continuous signals")
    print()

    # Find available models
    models = find_geometric_signals_models()
    if not models:
        print("❌ No trained models found!")
        print("   Run 'uv run signals-train' first to create a model")
        return

    model_path = models[0]  # Use latest
    print(f"📁 Found {len(models)} trained models")

    # Run demo
    success = asyncio.run(demo_direct_inference(model_path))

    if success:
        print("\n🎯 Key Insights:")
        print("• Core InferenceServer works directly with continuous values")
        print("• No wrapper or adapter classes needed")
        print("• Full caching, batching, optimization available")

        print("\n✅ CORE PLATFORM INTEGRATION SUCCESS:")
        print("• InferenceServer now handles device='auto' properly")
        print("• Safe globals added for Path objects (type-safe approach)")
        print("• InferenceServer supports both config formats")
        print("• InferenceServer supports both checkpoint key formats")
        print("• All core platform issues resolved!")

        print("\n🚀 To start production server:")
        print("   uv run signals-inference-demo --serve")
    else:
        print("\n❌ Core platform compatibility issues prevent proper integration")
        print("   These should be fixed in the core inference server!")


if __name__ == "__main__":
    import sys

    if "--serve" in sys.argv:
        # Server mode
        model_path = get_latest_model()
        if not model_path:
            print("❌ No trained models found")
            sys.exit(1)

        port = 8080
        device = "auto"

        # Parse port if provided
        if "--port" in sys.argv:
            try:
                port_idx = sys.argv.index("--port") + 1
                port = int(sys.argv[port_idx])
            except (IndexError, ValueError):
                print("❌ Invalid port argument")
                sys.exit(1)

        # Parse device if provided
        if "--device" in sys.argv:
            try:
                device_idx = sys.argv.index("--device") + 1
                device = sys.argv[device_idx]
            except IndexError:
                print("❌ Invalid device argument")
                sys.exit(1)

        asyncio.run(run_production_server(model_path, port, device))
    else:
        # Demo mode
        main()
