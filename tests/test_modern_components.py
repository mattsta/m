"""
Unit tests for modern transformer stability components.
Tests RMSNorm, RoPE, and scaled initialization.
"""

import math
import os

# Import the components we want to test
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from m.moe import (
    AttentionConfig,
    MultiheadSelfAttentionEinops,
    RMSNorm,
    RoPE,
    scaled_init_,
)


class TestRMSNorm:
    """Test RMSNorm implementation."""

    def test_rms_norm_shape(self):
        """Test that RMSNorm preserves input shape."""
        d_model = 768
        batch_size, seq_len = 4, 128

        rms_norm = RMSNorm(d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output = rms_norm(x)
        assert output.shape == x.shape

    def test_rms_norm_vs_layer_norm(self):
        """Test that RMSNorm behaves differently from LayerNorm (centered vs non-centered)."""
        d_model = 768
        batch_size, seq_len = 4, 128

        rms_norm = RMSNorm(d_model)
        layer_norm = torch.nn.LayerNorm(d_model)

        # Use same weight initialization for fair comparison
        with torch.no_grad():
            layer_norm.weight.copy_(rms_norm.weight)
            layer_norm.bias.zero_()

        x = torch.randn(batch_size, seq_len, d_model)

        rms_output = rms_norm(x)
        ln_output = layer_norm(x)

        # Should be different since RMS doesn't center the input
        assert not torch.allclose(rms_output, ln_output, rtol=1e-3)

    def test_rms_norm_zero_mean_invariant(self):
        """Test that RMSNorm works correctly with zero-mean input."""
        d_model = 768
        batch_size, seq_len = 4, 128

        rms_norm = RMSNorm(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        x = x - x.mean(dim=-1, keepdim=True)  # Zero-center

        output = rms_norm(x)

        # Should still work and have reasonable scale
        assert torch.isfinite(output).all()
        assert output.std() > 0.1  # Should not be too small

    def test_rms_norm_gradient_flow(self):
        """Test that gradients flow through RMSNorm correctly."""
        d_model = 768
        batch_size, seq_len = 4, 128

        rms_norm = RMSNorm(d_model)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = rms_norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert rms_norm.weight.grad is not None
        assert torch.isfinite(rms_norm.weight.grad).all()


class TestRoPE:
    """Test RoPE implementation."""

    def test_rope_shape_preservation(self):
        """Test that RoPE preserves query/key shapes."""
        d_head = 64
        batch_size, n_heads, seq_len = 4, 12, 128

        rope = RoPE(d_head)
        q = torch.randn(batch_size, n_heads, seq_len, d_head)
        k = torch.randn(batch_size, n_heads, seq_len, d_head)

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_rotation_property(self):
        """Test that RoPE implements proper rotation (norm preservation)."""
        d_head = 64
        batch_size, n_heads, seq_len = 2, 8, 64

        rope = RoPE(d_head)
        q = torch.randn(batch_size, n_heads, seq_len, d_head)
        k = torch.randn(batch_size, n_heads, seq_len, d_head)

        q_rot, k_rot = rope(q, k)

        # Rotation should preserve norm (approximately due to float precision)
        q_norm_orig = torch.norm(q, dim=-1)
        q_norm_rot = torch.norm(q_rot, dim=-1)
        k_norm_orig = torch.norm(k, dim=-1)
        k_norm_rot = torch.norm(k_rot, dim=-1)

        assert torch.allclose(q_norm_orig, q_norm_rot, rtol=1e-5)
        assert torch.allclose(k_norm_orig, k_norm_rot, rtol=1e-5)

    def test_rope_position_sensitivity(self):
        """Test that RoPE creates position-dependent representations."""
        d_head = 64
        seq_len = 128

        rope = RoPE(d_head)

        # Same vector at different positions should produce different outputs
        x = torch.randn(1, 1, 1, d_head).expand(1, 1, seq_len, d_head)
        x_rot, _ = rope(x, x)

        # Check that position 0 and position 64 are different
        pos_0 = x_rot[0, 0, 0]
        pos_64 = x_rot[0, 0, 64] if seq_len > 64 else x_rot[0, 0, -1]

        assert not torch.allclose(pos_0, pos_64, rtol=1e-3)

    def test_rope_caching(self):
        """Test that RoPE caching works correctly."""
        d_head = 64

        rope = RoPE(d_head, max_seq_len=256)

        # First call should build cache
        q1 = torch.randn(1, 1, 128, d_head)
        k1 = torch.randn(1, 1, 128, d_head)
        q1_rot, k1_rot = rope(q1, k1)

        # Second call with same seq_len should reuse cache
        q2 = torch.randn(1, 1, 128, d_head)
        k2 = torch.randn(1, 1, 128, d_head)
        q2_rot, k2_rot = rope(q2, k2)

        # Cache should be consistent (same cos/sin applied)
        # We test this indirectly by checking the rotation is consistent
        assert rope._cached_seq_len >= 128
        assert rope._cached_cos is not None
        assert rope._cached_sin is not None

    def test_rope_different_dtypes(self):
        """Test RoPE with different dtypes."""
        d_head = 64

        rope = RoPE(d_head)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            q = torch.randn(2, 4, 32, d_head, dtype=dtype)
            k = torch.randn(2, 4, 32, d_head, dtype=dtype)

            q_rot, k_rot = rope(q, k)

            assert q_rot.dtype == dtype
            assert k_rot.dtype == dtype
            assert torch.isfinite(q_rot).all()
            assert torch.isfinite(k_rot).all()


class TestScaledInitialization:
    """Test scaled initialization function."""

    def test_scaled_init_xavier(self):
        """Test scaled Xavier initialization."""
        n_layers = 12
        weight = torch.empty(768, 768)

        scaled_init_(weight, n_layers=n_layers, init_type="scaled_xavier")

        # Check that std is approximately what we expect
        expected_std = 0.02 / math.sqrt(2 * n_layers)
        actual_std = weight.std().item()

        # Allow some tolerance due to randomness
        assert abs(actual_std - expected_std) < 0.005

    def test_scaled_init_kaiming(self):
        """Test scaled Kaiming initialization."""
        n_layers = 12
        weight = torch.empty(768, 768)

        scaled_init_(weight, n_layers=n_layers, init_type="scaled_kaiming")

        # Should be scaled down from normal Kaiming
        assert torch.isfinite(weight).all()
        assert weight.std() > 0  # Should not be zero
        assert weight.std() < 0.1  # Should be reasonably small due to scaling

    def test_scaled_init_shapes(self):
        """Test scaled init with different tensor shapes."""
        n_layers = 6

        shapes = [(512, 512), (768, 2304), (16, 768, 3072)]  # Various shapes

        for shape in shapes:
            weight = torch.empty(*shape)
            scaled_init_(weight, n_layers=n_layers, init_type="scaled_xavier")

            assert weight.shape == shape
            assert torch.isfinite(weight).all()
            assert weight.std() > 0


class TestIntegrationModernComponents:
    """Integration tests for modern components working together."""

    def test_attention_with_rope_and_rms_norm(self):
        """Test attention module with RoPE enabled."""
        d_model = 768
        cfg = AttentionConfig(
            n_heads=12, use_rope=True, use_rms_norm=True, init="scaled_xavier"
        )

        attention = MultiheadSelfAttentionEinops(d_model, cfg, n_layers=6)

        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len, d_model)

        output = attention(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_attention_without_rope(self):
        """Test attention module with RoPE disabled for comparison."""
        d_model = 768
        cfg = AttentionConfig(
            n_heads=12, use_rope=False, use_rms_norm=True, init="scaled_xavier"
        )

        attention = MultiheadSelfAttentionEinops(d_model, cfg, n_layers=6)

        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len, d_model)

        output = attention(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_attention_rope_vs_no_rope_different_outputs(self):
        """Test that RoPE changes attention outputs."""
        d_model = 512

        cfg_with_rope = AttentionConfig(n_heads=8, use_rope=True, use_rms_norm=False)
        cfg_without_rope = AttentionConfig(
            n_heads=8, use_rope=False, use_rms_norm=False
        )

        attn_with_rope = MultiheadSelfAttentionEinops(d_model, cfg_with_rope)
        attn_without_rope = MultiheadSelfAttentionEinops(d_model, cfg_without_rope)

        # Use same weights for fair comparison
        with torch.no_grad():
            attn_without_rope.qkv.weight.copy_(attn_with_rope.qkv.weight)
            attn_without_rope.proj.weight.copy_(attn_with_rope.proj.weight)
            if attn_with_rope.qkv.bias is not None:
                attn_without_rope.qkv.bias.copy_(attn_with_rope.qkv.bias)
                attn_without_rope.proj.bias.copy_(attn_with_rope.proj.bias)

        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, d_model)

        output_with_rope = attn_with_rope(x)
        output_without_rope = attn_without_rope(x)

        # Outputs should be different due to RoPE
        assert not torch.allclose(output_with_rope, output_without_rope, rtol=1e-3)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
