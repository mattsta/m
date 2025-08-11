"""
Pytest tests for geometric signals inference integration.
Tests the complete pipeline: training -> inference -> deployment.
"""

import asyncio
from pathlib import Path

import aiohttp
import numpy as np
import pytest

from examples.geometric_signals.model_discovery import find_geometric_signals_models
from m.inference_server import InferenceServer


class TestGeometricSignalsInference:
    """Test suite for geometric signals inference integration."""

    @pytest.fixture(scope="class")
    def trained_model_path(self) -> Path:
        """Get a trained geometric signals model for testing."""
        models = find_geometric_signals_models()
        if not models:
            pytest.skip(
                "No trained geometric signals models found. Run training first."
            )
        return models[0]  # Use most recent model

    @pytest.fixture
    def test_signal(self) -> np.ndarray:
        """Generate test continuous signal data."""
        t = np.linspace(0, 4 * np.pi, 64)
        return np.sin(t) + 0.1 * np.cos(3 * t)  # Complex wave

    def test_inference_server_creation(self, trained_model_path: Path):
        """Test that InferenceServer can be created with geometric signals model."""
        server = InferenceServer(
            model_path=str(trained_model_path),
            device="cpu",  # Use CPU for consistent testing
            port=9000,  # Use high port to avoid conflicts
        )

        assert server is not None
        assert server.engine is not None
        assert server.engine.model is not None

    def test_direct_inference(self, trained_model_path: Path, test_signal: np.ndarray):
        """Test direct inference with continuous signal data."""
        server = InferenceServer(
            model_path=str(trained_model_path), device="cpu", port=9001
        )

        # Create inference request
        request_data = {
            "request_id": "test_continuous_signal",
            "tokens": test_signal.tolist(),  # Continuous values as tokens
            "max_new_tokens": 16,
            "temperature": 1.0,
            "use_cache": True,
        }

        # Run inference
        response = asyncio.run(server.handle_request(request_data))

        # Verify response structure
        assert "request_id" in response
        assert "generated_tokens" in response
        assert "generation_time" in response
        assert "tokens_per_second" in response

        # Verify response content
        assert response["request_id"] == "test_continuous_signal"
        assert len(response["generated_tokens"]) > 0
        assert response["generation_time"] > 0
        assert response["tokens_per_second"] > 0

    @pytest.mark.asyncio
    async def test_server_startup_and_health(self, trained_model_path: Path):
        """Test server startup and health check endpoint."""
        server = InferenceServer(
            model_path=str(trained_model_path),
            device="cpu",
            port=9002,  # Use unique port
        )

        # Start server in background task
        server_task = asyncio.create_task(server.run(host="127.0.0.1"))

        # Wait for server to start
        await asyncio.sleep(2)

        try:
            # Test health endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{server.port}/health"
                ) as response:
                    assert response.status == 200
                    health_data = await response.json()
                    assert health_data["status"] == "healthy"
                    assert "stats" in health_data

        finally:
            # Cleanup: cancel server task
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_inference_api_endpoint(
        self, trained_model_path: Path, test_signal: np.ndarray
    ):
        """Test the /infer API endpoint with continuous signal data."""
        server = InferenceServer(
            model_path=str(trained_model_path), device="cpu", port=9003
        )

        # Start server
        server_task = asyncio.create_task(server.run(host="127.0.0.1"))
        await asyncio.sleep(2)

        try:
            # Prepare inference request
            request_data = {
                "request_id": "api_test",
                "tokens": test_signal.tolist(),
                "max_new_tokens": 8,
                "temperature": 1.0,
            }

            # Send inference request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://127.0.0.1:{server.port}/infer", json=request_data
                ) as response:
                    assert response.status == 200
                    result = await response.json()

                    # Verify inference result
                    assert result["request_id"] == "api_test"
                    assert "generated_tokens" in result
                    assert len(result["generated_tokens"]) > 0
                    assert "generation_time" in result
                    assert "tokens_per_second" in result

        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_port_conflict_handling(self, trained_model_path: Path):
        """Test that server handles port conflicts gracefully."""
        # Create first server on port 9004
        server1 = InferenceServer(
            model_path=str(trained_model_path), device="cpu", port=9004
        )

        # Create second server with same port - should find alternative
        server2 = InferenceServer(
            model_path=str(trained_model_path), device="cpu", port=9004
        )

        # Verify port conflict resolution
        start_task1 = asyncio.create_task(server1.run(host="127.0.0.1"))
        await asyncio.sleep(1)  # Let first server start

        start_task2 = asyncio.create_task(server2.run(host="127.0.0.1"))
        await asyncio.sleep(1)  # Let second server find alternative port

        try:
            # Should have different ports
            assert server1.port != server2.port
            assert server2.port > server1.port

        finally:
            start_task1.cancel()
            start_task2.cancel()
            try:
                await start_task1
            except asyncio.CancelledError:
                pass
            try:
                await start_task2
            except asyncio.CancelledError:
                pass

    def test_continuous_values_compatibility(self, trained_model_path: Path):
        """Test that various continuous value formats work correctly."""
        server = InferenceServer(
            model_path=str(trained_model_path), device="cpu", port=9005
        )

        test_cases = [
            # Different signal types
            np.sin(np.linspace(0, 2 * np.pi, 32)),  # Simple sine
            np.random.randn(32),  # Random noise
            np.linspace(-1, 1, 32),  # Linear ramp
            np.ones(32) * 0.5,  # Constant signal
        ]

        for i, signal in enumerate(test_cases):
            request_data = {
                "request_id": f"compatibility_test_{i}",
                "tokens": signal.tolist(),
                "max_new_tokens": 4,
                "temperature": 1.0,
            }

            response = asyncio.run(server.handle_request(request_data))

            # Verify all signal types work
            assert response["request_id"] == f"compatibility_test_{i}"
            assert len(response["generated_tokens"]) > 0

    def test_model_discovery_integration(self):
        """Test that model discovery works correctly with inference."""
        models = find_geometric_signals_models()

        if not models:
            pytest.skip("No trained models found")

        # Test that we can create servers for all discovered models
        for model_path in models[:3]:  # Test first 3 models
            try:
                server = InferenceServer(
                    model_path=str(model_path),
                    device="cpu",
                    port=9006 + hash(str(model_path)) % 100,  # Unique port per model
                )
                assert server is not None

                # Quick functionality test
                test_signal = np.sin(np.linspace(0, np.pi, 16))
                request_data = {
                    "request_id": f"model_test_{model_path.stem}",
                    "tokens": test_signal.tolist(),
                    "max_new_tokens": 2,
                    "temperature": 1.0,
                }

                response = asyncio.run(server.handle_request(request_data))
                assert len(response["generated_tokens"]) > 0

            except Exception as e:
                pytest.fail(f"Model {model_path} failed inference test: {e}")


# Integration test for full pipeline
def test_geometric_signals_end_to_end():
    """Test the complete geometric signals pipeline."""
    models = find_geometric_signals_models()

    if not models:
        pytest.skip("No trained geometric signals models found")

    model_path = models[0]

    # Test 1: Model loading
    server = InferenceServer(model_path=str(model_path), device="cpu", port=9100)
    assert server.engine is not None

    # Test 2: Continuous signal inference
    t = np.linspace(0, 4 * np.pi, 48)
    test_signal = np.sin(t) + 0.2 * np.sin(3 * t)

    request_data = {
        "request_id": "end_to_end_test",
        "tokens": test_signal.tolist(),
        "max_new_tokens": 12,
        "temperature": 1.0,
        "use_cache": True,
    }

    response = asyncio.run(server.handle_request(request_data))

    # Test 3: Response validation
    assert response["request_id"] == "end_to_end_test"
    generated_tokens = response["generated_tokens"]
    # Handle both flat list and nested list (batch dimension) formats
    if isinstance(generated_tokens[0], list):
        generated_tokens = generated_tokens[0]  # Remove batch dimension
    assert len(generated_tokens) == 12
    assert response["generation_time"] > 0
    assert response["tokens_per_second"] > 0
    assert response["cache_hit_rate"] >= 0

    print("✅ End-to-end test passed!")
    print(f"   Model: {model_path.name}")
    print(f"   Generated: {len(generated_tokens)} values")
    print(f"   Time: {response['generation_time']:.3f}s")
    print(f"   Throughput: {response['tokens_per_second']:.1f} values/sec")
