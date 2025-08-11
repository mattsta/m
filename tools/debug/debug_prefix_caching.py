#!/usr/bin/env python3
"""Test script to verify prefix caching functionality."""

import torch

from m.inference_server import (
    InferenceEngine,
    InferenceRequest,
)
from m.moe import ModelConfig, MoESequenceRegressor

# Create a small model for testing
model_config = ModelConfig(
    n_layers=2,
    input_dim=64,
    target_dim=1,
)
model = MoESequenceRegressor(model_config).eval()

# Create inference engine
engine = InferenceEngine(
    model,
    device="cpu",  # Use CPU for testing
    kv_cache_size=10,
    kv_cache_max_tokens=1000,
)

print("Testing Prefix Caching Implementation")
print("=" * 50)

# Test 1: Basic prefix reuse
print("\nTest 1: Basic Prefix Reuse")
print("-" * 30)

# Create tokens with common prefix
common_prefix = torch.randint(0, 100, (1, 10))
suffix1 = torch.randint(100, 150, (1, 5))
suffix2 = torch.randint(150, 200, (1, 5))

tokens1 = torch.cat([common_prefix, suffix1], dim=1)
tokens2 = torch.cat([common_prefix, suffix2], dim=1)

print(f"Tokens1 shape: {tokens1.shape}")
print(f"Tokens2 shape: {tokens2.shape}")
print(f"Common prefix length: {common_prefix.shape[1]}")

# First request - should cache the KV states
request1 = InferenceRequest(
    request_id="req1",
    tokens=tokens1,
    max_new_tokens=2,
    use_cache=True,
    reuse_prefix=True,
)

print("\nProcessing request 1...")
response1 = engine.generate(request1)
print(f"Request 1 complete. Cache entries: {len(engine.kv_cache.cache)}")

# Second request - should reuse prefix from first request
request2 = InferenceRequest(
    request_id="req2",
    tokens=tokens2,
    max_new_tokens=2,
    use_cache=True,
    reuse_prefix=True,
)

print("\nProcessing request 2...")
response2 = engine.generate(request2)
print(f"Request 2 complete. Prefix reused tokens: {response2.prefix_reused_tokens}")
print(f"Cache hits: {engine.cache_hits}")

# Test 2: Partial prefix matching
print("\n\nTest 2: Partial Prefix Matching")
print("-" * 30)

# Reset engine
engine.clear_cache()

# Create longer sequences with partial overlap
long_prefix = torch.randint(0, 100, (1, 20))
tokens3 = long_prefix.clone()
tokens4 = torch.cat([long_prefix[:, :15], torch.randint(200, 250, (1, 10))], dim=1)

request3 = InferenceRequest(
    request_id="req3",
    tokens=tokens3,
    max_new_tokens=2,
    use_cache=True,
    reuse_prefix=True,
)

print(f"\nTokens3 length: {tokens3.shape[1]}")
print("Processing request 3...")
response3 = engine.generate(request3)

request4 = InferenceRequest(
    request_id="req4",
    tokens=tokens4,
    max_new_tokens=2,
    use_cache=True,
    reuse_prefix=True,
)

print(f"\nTokens4 length: {tokens4.shape[1]}")
print("Expected prefix match: 15 tokens")
print("Processing request 4...")
response4 = engine.generate(request4)
print(f"Actual prefix reused: {response4.prefix_reused_tokens} tokens")

# Test 3: No prefix match
print("\n\nTest 3: No Prefix Match")
print("-" * 30)

tokens5 = torch.randint(300, 400, (1, 15))
request5 = InferenceRequest(
    request_id="req5",
    tokens=tokens5,
    max_new_tokens=2,
    use_cache=True,
    reuse_prefix=True,
)

print("Processing request 5 (completely different tokens)...")
response5 = engine.generate(request5)
print(f"Prefix reused: {response5.prefix_reused_tokens} tokens (should be 0)")

# Summary
print("\n" + "=" * 50)
print("Summary:")
print(f"Total requests: {engine.total_requests}")
print(f"Cache hits: {engine.cache_hits}")
print(f"Cache hit rate: {engine.cache_hits / engine.total_requests:.2%}")
print(f"Total prefix tokens reused: {engine.total_prefix_tokens_reused}")
print(f"KV cache entries: {len(engine.kv_cache.cache)}")
print(f"KV cache total tokens: {engine.kv_cache.total_tokens}")

# Verify that prefix caching actually works
assert response2.prefix_reused_tokens > 0, "Prefix should have been reused in test 1"
assert response4.prefix_reused_tokens > 0, (
    "Partial prefix should have been reused in test 2"
)
assert response5.prefix_reused_tokens == 0, "No prefix should be reused in test 3"

print("\n✅ All prefix caching tests passed!")
