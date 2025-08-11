"""
3D Trajectory Inference Demo using Core Platform

Demonstrates direct integration with the core InferenceServer for 3D trajectory prediction.
Shows how the existing platform naturally supports multi-dimensional continuous data.
"""

from __future__ import annotations

import asyncio
import pathlib
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import numpy as np
import torch

# Add safe globals for checkpoint loading - minimal exposure
torch.serialization.add_safe_globals([Path, pathlib.PosixPath, type(np.float64(1.0))])

from m.inference_server import InferenceServer

from .datasets import generate_sample_trajectories
from .model_discovery import get_latest_trajectory_model
from .visualization import Trajectory3DVisualizer


@dataclass(slots=True)
class ResponseMetadata:
    """Metadata from inference response."""

    cache_hits: int
    request_id: str


@dataclass(slots=True)
class DirectInferenceResult:
    """Result from direct inference demo."""

    input_3d: np.ndarray
    predicted_3d: np.ndarray
    actual_3d: np.ndarray
    inference_time: float
    response_metadata: ResponseMetadata


@dataclass(slots=True)
class BatchInferenceResult:
    """Result from batch inference demo."""

    batch_size: int
    total_time: float
    avg_time_per_request: float
    total_cache_hits: int


class Trajectory3DInferenceDemo:
    """
    Demonstration of 3D trajectory inference using the core InferenceServer.

    Shows how the existing platform naturally handles 3D coordinates without modifications.
    """

    def __init__(self, model_path: Path, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.server: InferenceServer | None = None

        print("🎯 3D Trajectory Inference Demo")
        print(f"📁 Model: {model_path}")
        print(f"🖥️  Device: {device}")

    async def initialize_server(self) -> None:
        """Initialize the InferenceServer with the 3D trajectory model."""
        print("\n🚀 Initializing InferenceServer...")

        try:
            self.server = InferenceServer(
                model_path=str(self.model_path), device=self.device
            )
            print("✅ InferenceServer initialized successfully")
            print("📊 Model loaded successfully")

        except Exception as e:
            print(f"❌ Failed to initialize server: {e}")
            raise

    async def demo_direct_inference(self) -> DirectInferenceResult:
        """Demonstrate direct inference with 3D trajectory data."""
        print("\n🎯 Direct 3D Trajectory Inference Demo")

        # Generate sample 3D trajectory data
        print("📊 Generating sample 3D trajectory data...")
        trajectories = generate_sample_trajectories()

        # Use helical trajectory as example
        helical_trajectory = trajectories["helical"]

        # Take first 64 points as input context
        input_length = 64
        input_3d = helical_trajectory[:input_length]  # [64, 3]

        print(f"📍 Input trajectory shape: {input_3d.shape}")
        print(
            f"📍 Input range: X[{input_3d[:, 0].min():.2f}, {input_3d[:, 0].max():.2f}], "
            f"Y[{input_3d[:, 1].min():.2f}, {input_3d[:, 1].max():.2f}], "
            f"Z[{input_3d[:, 2].min():.2f}, {input_3d[:, 2].max():.2f}]"
        )

        # Convert to list format for API
        input_tokens = input_3d.tolist()

        # Prepare inference request
        request = {
            "request_id": "3d_trajectory_prediction",
            "tokens": input_tokens,  # List of [x, y, z] coordinates
            "max_new_tokens": 32,  # Predict 32 more 3D positions
            "temperature": 1.0,
            "use_cache": True,
        }

        print(
            f"🔮 Requesting prediction of {request['max_new_tokens']} future 3D positions..."
        )

        # Perform inference
        start_time = time.time()
        assert self.server is not None, "Server not initialized"
        response = await self.server.handle_request(request)
        inference_time = time.time() - start_time

        print(f"✅ Inference completed in {inference_time:.3f}s")

        # Extract predictions
        predicted_tokens = response["generated_tokens"]
        predicted_3d = np.array(predicted_tokens)  # [32, 3]

        print(f"🎯 Predicted trajectory shape: {predicted_3d.shape}")
        print(
            f"🎯 Prediction range: X[{predicted_3d[:, 0].min():.2f}, {predicted_3d[:, 0].max():.2f}], "
            f"Y[{predicted_3d[:, 1].min():.2f}, {predicted_3d[:, 1].max():.2f}], "
            f"Z[{predicted_3d[:, 2].min():.2f}, {predicted_3d[:, 2].max():.2f}]"
        )

        # Show response metadata
        print(f"📈 Cache hits: {response.get('cache_hits', 0)}")
        print(f"🆔 Request ID: {response.get('request_id', 'N/A')}")

        # Create visualization
        print("🎨 Creating prediction visualization...")
        visualizer = Trajectory3DVisualizer("outputs/trajectory_3d/inference_demo")

        # Convert to torch tensors for visualization
        input_tensor = torch.tensor(input_3d, dtype=torch.float32)

        # For comparison, use the actual continuation from the trajectory
        actual_continuation = helical_trajectory[
            input_length : input_length + 32
        ]  # [32, 3]
        actual_tensor = torch.tensor(actual_continuation, dtype=torch.float32)

        predicted_tensor = torch.tensor(predicted_3d, dtype=torch.float32)

        visualizer.plot_prediction_comparison(
            input_tensor,
            actual_tensor,
            predicted_tensor,
            "helical_inference_demo",
            "outputs/trajectory_3d/inference_demo/direct_inference_comparison.png",
        )

        print("✅ Visualization saved to outputs/trajectory_3d/inference_demo/")

        return DirectInferenceResult(
            input_3d=input_3d,
            predicted_3d=predicted_3d,
            actual_3d=actual_continuation,
            inference_time=inference_time,
            response_metadata=ResponseMetadata(
                cache_hits=response.get("cache_hits", 0),
                request_id=response.get("request_id", "N/A"),
            ),
        )

    async def demo_batch_inference(self) -> BatchInferenceResult:
        """Demonstrate batch inference with multiple trajectory types."""
        print("\n📦 Batch 3D Trajectory Inference Demo")

        # Generate multiple trajectory samples
        trajectories = generate_sample_trajectories()

        batch_requests = []
        trajectory_types = []

        # Create batch of different trajectory types
        for i, (traj_type, trajectory) in enumerate(trajectories.items()):
            if i >= 4:  # Limit batch size
                break

            input_3d = trajectory[:48]  # Shorter input for batch demo
            input_tokens = input_3d.tolist()

            request = {
                "request_id": f"batch_3d_{traj_type}_{i}",
                "tokens": input_tokens,
                "max_new_tokens": 16,  # Shorter predictions for batch
                "temperature": 1.0,
                "use_cache": True,
            }

            batch_requests.append(request)
            trajectory_types.append(traj_type)

        print(
            f"🔮 Processing batch of {len(batch_requests)} 3D trajectory predictions..."
        )

        # Process batch
        start_time = time.time()
        batch_responses = []

        for request in batch_requests:
            assert self.server is not None, "Server not initialized"
            response = await self.server.handle_request(request)
            batch_responses.append(response)

        batch_time = time.time() - start_time

        print(f"✅ Batch inference completed in {batch_time:.3f}s")
        print(
            f"📊 Average time per prediction: {batch_time / len(batch_requests):.3f}s"
        )

        # Analyze results
        total_cache_hits = sum(r.get("cache_hits", 0) for r in batch_responses)
        print(f"📈 Total cache hits: {total_cache_hits}")

        # Show sample results
        for i, (response, traj_type) in enumerate(
            zip(batch_responses, trajectory_types)
        ):
            predicted_3d = np.array(response["generated_tokens"])
            print(f"  {traj_type}: predicted {predicted_3d.shape[0]} 3D positions")

        return BatchInferenceResult(
            batch_size=len(batch_requests),
            total_time=batch_time,
            avg_time_per_request=batch_time / len(batch_requests),
            total_cache_hits=total_cache_hits,
        )

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.server:
            print("🧹 Cleaning up InferenceServer...")
            # The InferenceServer doesn't have explicit cleanup, but we can clear references
            self.server = None


async def demo_production_server(
    model_path: Path,
    port: int = 8080,
    host: str = "localhost",
    device: str = "auto",
    duration_seconds: int = 30,
) -> None:
    """
    Demo production server setup with HTTP API for 3D trajectory inference.

    Args:
        model_path: Path to trained 3D trajectory model
        port: Port to run server on
        host: Host to bind to
        device: Device for inference
        duration_seconds: How long to run the demo server
    """
    print("🌐 Production 3D Trajectory Server Demo")
    print(f"📁 Model: {model_path}")
    print(f"🌍 Server: http://{host}:{port}")
    print(f"⏱️  Duration: {duration_seconds}s")

    # NOTE: create_fastapi_app not available in current inference_server implementation
    print(
        "❌ Production server demo not available - create_fastapi_app not implemented"
    )
    print("🎯 Use direct InferenceServer instead for programmatic access")


async def demo_http_client(base_url: str) -> None:
    """Demo HTTP client for 3D trajectory API."""
    print(f"📡 Testing HTTP API at {base_url}")

    async with aiohttp.ClientSession() as session:
        # Health check
        try:
            async with session.get(f"{base_url}/health") as response:
                health_data = await response.json()
                print(f"✅ Health check: {health_data}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return

        # Generate sample 3D trajectory for API test
        trajectories = generate_sample_trajectories()
        orbital_trajectory = trajectories["orbital"]
        input_3d = orbital_trajectory[:32].tolist()  # Convert to list for JSON

        # Test inference endpoint
        inference_request = {
            "request_id": "http_api_test",
            "tokens": input_3d,
            "max_new_tokens": 16,
            "temperature": 1.0,
            "use_cache": True,
        }

        try:
            async with session.post(
                f"{base_url}/infer",
                json=inference_request,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    predicted_3d = np.array(result["generated_tokens"])
                    print(
                        f"✅ API inference: predicted {predicted_3d.shape[0]} 3D positions"
                    )
                    print(
                        f"📊 Response metadata: cache_hits={result.get('cache_hits', 0)}"
                    )
                else:
                    error_text = await response.text()
                    print(f"❌ API inference failed ({response.status}): {error_text}")

        except Exception as e:
            print(f"❌ API request failed: {e}")


def main():
    """Main CLI entrypoint for 3D trajectory inference demos."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="3D Trajectory Inference Demo")
    parser.add_argument(
        "--model", type=Path, help="Path to trained model (use --latest for automatic)"
    )
    parser.add_argument(
        "--latest", action="store_true", help="Use latest trained model"
    )
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--serve", action="store_true", help="Start production server demo"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for server demo")
    parser.add_argument("--host", default="localhost", help="Host for server demo")
    parser.add_argument(
        "--duration", type=int, default=30, help="Server demo duration (seconds)"
    )

    args = parser.parse_args()

    # Determine model path
    if args.latest:
        print("🔍 Finding latest 3D trajectory model...")
        latest_model = get_latest_trajectory_model()
        if not latest_model:
            print("❌ No trained models found. Train a model first!")
            print(
                "Example: uv run trajectory-3d-train examples/trajectory_3d/configs/quick_test_3d.yaml"
            )
            sys.exit(1)
        model_path = latest_model.model_path
        print(f"📁 Using latest model: {model_path}")
    elif args.model:
        model_path = args.model
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)
    else:
        print("❌ Must specify --model or --latest")
        print("Examples:")
        print("  trajectory-3d-inference-demo --latest")
        print(
            "  trajectory-3d-inference-demo --model outputs/trajectory_3d/my_experiment/best_model.pt"
        )
        print("  trajectory-3d-inference-demo --latest --serve --port 8080")
        sys.exit(1)

    async def run_demo():
        if args.serve:
            # Production server demo
            await demo_production_server(
                model_path=model_path,
                port=args.port,
                host=args.host,
                device=args.device,
                duration_seconds=args.duration,
            )
        else:
            # Direct inference demo
            demo = Trajectory3DInferenceDemo(model_path, args.device)

            try:
                await demo.initialize_server()

                # Run direct inference demo
                direct_results = await demo.demo_direct_inference()

                # Run batch inference demo
                batch_results = await demo.demo_batch_inference()

                print("\n📊 Demo Summary:")
                print(
                    f"  Direct inference time: {direct_results['inference_time']:.3f}s"
                )
                print(
                    f"  Batch average time: {batch_results['avg_time_per_request']:.3f}s"
                )
                print(
                    f"  Cache utilization: {direct_results['response_metadata']['cache_hits']} + {batch_results['total_cache_hits']} hits"
                )

                print("\n✅ 3D Trajectory inference demo completed successfully!")
                print(
                    "🎨 Check outputs/trajectory_3d/inference_demo/ for visualizations"
                )

            except Exception as e:
                print(f"❌ Demo failed: {e}")
                sys.exit(1)
            finally:
                await demo.cleanup()

    # Run the async demo
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
