"""
Comprehensive tests for prefix caching functionality.
"""

import pytest
import torch

from m.inference_server import (
    InferenceEngine,
    InferenceRequest,
    KVCache,
    PrefixCache,
)
from m.moe import ModelConfig, MoESequenceRegressor


class TestPrefixCaching:
    """Test prefix caching and KV state reuse."""

    @pytest.fixture
    def engine(self):
        """Create inference engine with small model."""
        model_config = ModelConfig(
            n_layers=2,
            input_dim=64,
            target_dim=1,
        )
        model = MoESequenceRegressor(model_config).eval()

        return InferenceEngine(
            model,
            device="cpu",  # Use CPU for consistent testing
            kv_cache_size=10,
            kv_cache_max_tokens=1000,
        )

    def test_exact_prefix_reuse(self, engine):
        """Test reusing exact prefix from previous request."""
        # Create tokens with common prefix
        common_prefix = torch.randint(0, 100, (1, 10))
        suffix1 = torch.randint(100, 150, (1, 5))
        suffix2 = torch.randint(150, 200, (1, 5))

        tokens1 = torch.cat([common_prefix, suffix1], dim=1)
        tokens2 = torch.cat([common_prefix, suffix2], dim=1)

        # First request - establishes cache
        request1 = InferenceRequest(
            request_id="req1",
            tokens=tokens1,
            max_new_tokens=2,
            use_cache=True,
            reuse_prefix=True,
        )
        response1 = engine.generate(request1)
        assert response1.generation_time > 0  # Verify first request succeeded

        # Second request - should reuse prefix
        request2 = InferenceRequest(
            request_id="req2",
            tokens=tokens2,
            max_new_tokens=2,
            use_cache=True,
            reuse_prefix=True,
        )
        response2 = engine.generate(request2)

        # Verify prefix was reused
        assert response2.prefix_reused_tokens == 10
        assert engine.cache_hits == 1
        assert engine.total_prefix_tokens_reused == 10

    def test_partial_prefix_matching(self, engine):
        """Test matching partial prefixes of different lengths."""
        # Create sequences with partial overlap
        long_sequence = torch.randint(0, 100, (1, 20))

        # First request with full sequence
        request1 = InferenceRequest(
            request_id="req1",
            tokens=long_sequence,
            max_new_tokens=1,
            use_cache=True,
            reuse_prefix=True,
        )
        response1 = engine.generate(request1)
        assert response1.generation_time > 0  # Verify first request succeeded

        # Second request with partial prefix (first 15 tokens) + different suffix
        partial_tokens = torch.cat(
            [long_sequence[:, :15], torch.randint(200, 250, (1, 10))], dim=1
        )

        request2 = InferenceRequest(
            request_id="req2",
            tokens=partial_tokens,
            max_new_tokens=1,
            use_cache=True,
            reuse_prefix=True,
        )
        response2 = engine.generate(request2)

        # Should reuse first 15 tokens
        assert response2.prefix_reused_tokens == 15

    def test_no_prefix_match(self, engine):
        """Test behavior when no prefix matches."""
        # First request
        tokens1 = torch.randint(0, 100, (1, 15))
        request1 = InferenceRequest(
            request_id="req1",
            tokens=tokens1,
            max_new_tokens=1,
            use_cache=True,
            reuse_prefix=True,
        )
        response1 = engine.generate(request1)
        assert response1.generation_time > 0  # Verify first request succeeded

        # Second request with completely different tokens
        tokens2 = torch.randint(200, 300, (1, 15))
        request2 = InferenceRequest(
            request_id="req2",
            tokens=tokens2,
            max_new_tokens=1,
            use_cache=True,
            reuse_prefix=True,
        )
        response2 = engine.generate(request2)

        # No prefix should be reused
        assert response2.prefix_reused_tokens == 0

    def test_multiple_prefix_reuse(self, engine):
        """Test reusing same prefix across multiple requests."""
        # Common prefix
        prefix = torch.randint(0, 100, (1, 10))

        # First request establishes cache
        tokens1 = torch.cat([prefix, torch.randint(100, 110, (1, 5))], dim=1)
        request1 = InferenceRequest(
            request_id="req1",
            tokens=tokens1,
            max_new_tokens=1,
            use_cache=True,
            reuse_prefix=True,
        )
        response1 = engine.generate(request1)
        assert response1.generation_time > 0  # Verify first request succeeded

        # Multiple subsequent requests should all reuse the prefix
        total_reused = 0
        for i in range(3):
            tokens = torch.cat(
                [prefix, torch.randint(200 + i * 10, 210 + i * 10, (1, 5))], dim=1
            )
            request = InferenceRequest(
                request_id=f"req{i + 2}",
                tokens=tokens,
                max_new_tokens=1,
                use_cache=True,
                reuse_prefix=True,
            )
            response = engine.generate(request)
            assert response.prefix_reused_tokens == 10
            total_reused += response.prefix_reused_tokens

        # Check cumulative stats
        assert engine.cache_hits == 3
        assert engine.total_prefix_tokens_reused == total_reused

    def test_prefix_cache_with_generation(self, engine):
        """Test that prefix caching works during token generation."""
        # Common context
        context = torch.randint(0, 100, (1, 20))

        # First generation
        request1 = InferenceRequest(
            request_id="req1",
            tokens=context.clone(),
            max_new_tokens=5,
            use_cache=True,
            reuse_prefix=True,
        )
        response1 = engine.generate(request1)

        # Second generation with same context
        request2 = InferenceRequest(
            request_id="req2",
            tokens=context.clone(),
            max_new_tokens=5,
            use_cache=True,
            reuse_prefix=True,
        )
        response2 = engine.generate(request2)

        # Should reuse the entire context
        assert response2.prefix_reused_tokens == 20
        # Both should generate tokens (may differ due to sampling)
        assert response1.generated_tokens.shape[1] > 0
        assert response2.generated_tokens.shape[1] > 0

    def test_cache_eviction_and_prefix(self, engine):
        """Test prefix matching still works with cache eviction."""
        # Set a small cache
        engine.kv_cache.max_entries = 3

        # Fill cache with entries
        for i in range(5):
            tokens = torch.randint(i * 100, (i + 1) * 100, (1, 10))
            request = InferenceRequest(
                request_id=f"req{i}",
                tokens=tokens,
                max_new_tokens=1,
                use_cache=True,
                reuse_prefix=True,
            )
            engine.generate(request)

        # Cache should only have last 3 entries
        assert len(engine.kv_cache.cache) <= 3

        # Try to reuse prefix from an evicted entry (req0)
        tokens_like_req0 = torch.randint(0, 100, (1, 10))
        request = InferenceRequest(
            request_id="req_test",
            tokens=tokens_like_req0,
            max_new_tokens=1,
            use_cache=True,
            reuse_prefix=True,
        )
        response = engine.generate(request)

        # Should not find a match (entry was evicted)
        assert response.prefix_reused_tokens == 0

    def test_prefix_hash_consistency(self):
        """Test that prefix hashing is consistent."""
        prefix_cache = PrefixCache()

        tokens = torch.tensor([[1, 2, 3, 4, 5]])

        # Same tokens should produce same hash
        hash1 = prefix_cache.compute_prefix_hash(tokens, 3)
        hash2 = prefix_cache.compute_prefix_hash(tokens, 3)
        assert hash1 == hash2

        # Different lengths should produce different hashes
        hash3 = prefix_cache.compute_prefix_hash(tokens, 4)
        assert hash1 != hash3

        # Different tokens should produce different hashes
        tokens2 = torch.tensor([[1, 2, 4, 4, 5]])
        hash4 = prefix_cache.compute_prefix_hash(tokens2, 3)
        assert hash1 != hash4

    def test_kv_state_merging(self, engine):
        """Test that KV states are properly merged when extending from prefix."""
        # Initial sequence
        tokens1 = torch.randint(0, 100, (1, 10))
        request1 = InferenceRequest(
            request_id="req1",
            tokens=tokens1,
            max_new_tokens=0,  # Just process, don't generate
            use_cache=True,
            reuse_prefix=True,
        )
        response1 = engine.generate(request1)
        assert response1.generation_time > 0  # Verify first request succeeded

        # Extended sequence (original + more tokens)
        extension = torch.randint(100, 150, (1, 5))
        tokens2 = torch.cat([tokens1, extension], dim=1)

        request2 = InferenceRequest(
            request_id="req2",
            tokens=tokens2,
            max_new_tokens=0,
            use_cache=True,
            reuse_prefix=True,
        )
        response2 = engine.generate(request2)

        # Should have reused the prefix
        assert response2.prefix_reused_tokens == 10

        # Check that the cache entry for req2 has the full sequence
        cache_entry = engine.kv_cache.get("req2")
        assert cache_entry is not None
        # KV states should have length 15 (10 reused + 5 new)
        if cache_entry.key_states:
            assert cache_entry.key_states[0].shape[2] == 15


class TestKVCacheManagement:
    """Test KV cache management and eviction."""

    def test_lru_eviction(self):
        """Test that LRU eviction works correctly."""
        cache = KVCache(max_entries=3, max_total_tokens=100, device="cpu")

        # Add entries
        for i in range(5):
            keys = [torch.randn(1, 4, 10, 64)]
            values = [torch.randn(1, 4, 10, 64)]
            cache.put(f"req{i}", keys, values, tokens=torch.randint(0, 100, (1, 10)))

        # Should only have last 3 entries
        assert len(cache.cache) == 3
        assert "req0" not in cache.cache
        assert "req1" not in cache.cache
        assert "req4" in cache.cache

    def test_token_limit_eviction(self):
        """Test eviction based on total token limit."""
        cache = KVCache(max_entries=10, max_total_tokens=50, device="cpu")

        # Add entries that exceed token limit
        for i in range(6):
            keys = [torch.randn(1, 4, 10, 64)]
            values = [torch.randn(1, 4, 10, 64)]
            cache.put(f"req{i}", keys, values)

        # Total tokens should not exceed limit
        assert cache.total_tokens <= 50
        # Older entries should be evicted
        assert len(cache.cache) <= 5
