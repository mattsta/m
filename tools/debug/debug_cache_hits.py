#!/usr/bin/env python3
"""Debug cache hits counter."""

import torch

from m.inference_server import InferenceEngine, InferenceRequest
from m.moe import ModelConfig, MoESequenceRegressor

model = MoESequenceRegressor(ModelConfig(n_layers=1, input_dim=64)).eval()
engine = InferenceEngine(model, device="cpu")

# Test multiple requests with same prefix
prefix = torch.randint(0, 100, (1, 10))

for i in range(3):
    tokens = torch.cat(
        [prefix, torch.randint(100 + i * 10, 110 + i * 10, (1, 5))], dim=1
    )
    request = InferenceRequest(
        request_id=f"req{i}",
        tokens=tokens,
        max_new_tokens=1,
        use_cache=True,
        reuse_prefix=True,
    )

    print(f"\nRequest {i}:")
    print(f"  Cache entries before: {len(engine.kv_cache.cache)}")
    print(f"  Cache hits before: {engine.cache_hits}")

    response = engine.generate(request)

    print(f"  Cache entries after: {len(engine.kv_cache.cache)}")
    print(f"  Cache hits after: {engine.cache_hits}")
    print(f"  Prefix reused: {response.prefix_reused_tokens}")

print("\nFinal stats:")
print(f"  Total requests: {engine.total_requests}")
print(f"  Total cache hits: {engine.cache_hits}")
print(f"  Total prefix tokens reused: {engine.total_prefix_tokens_reused}")
