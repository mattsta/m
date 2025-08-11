"""
Integration tests for MoE transformer with modern stability components.
Tests TransformerBlock and full MoE model with RoPE, RMSNorm, and scaled initialization.
"""

import os

# Import the components we want to test
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from m.moe import (
    AttentionConfig,
    BlockConfig,
    ExpertConfig,
    ModelConfig,
    MoEConfig,
    MoESequenceRegressor,
    RouterConfig,
    TransformerBlock,
    build_moe,
)


class TestTransformerBlockIntegration:
    """Test TransformerBlock with modern components."""

    def test_transformer_block_modern_config(self):
        """Test TransformerBlock with modern stability features enabled."""
        d_model = 512

        # Configure modern features
        attn_cfg = AttentionConfig(
            n_heads=8, use_rope=True, use_rms_norm=True, init="scaled_xavier"
        )

        moe_cfg = MoEConfig(
            d_model=d_model,
            router=RouterConfig(n_experts=8, k=2, use_rms_norm=True),
            expert=ExpertConfig(d_model=d_model, init="scaled_xavier"),
        )

        block_cfg = BlockConfig(attn=attn_cfg, moe=moe_cfg, use_rms_norm=True)

        block = TransformerBlock(block_cfg, n_layers=6)

        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len, d_model)

        output, aux_loss, metrics = block(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        assert torch.isfinite(aux_loss).all()
        assert aux_loss.numel() == 1  # scalar

    def test_transformer_block_legacy_vs_modern(self):
        """Compare legacy vs modern transformer block configurations."""
        d_model = 512

        # Legacy config (LayerNorm, no RoPE, Xavier init)
        legacy_attn = AttentionConfig(
            n_heads=8, use_rope=False, use_rms_norm=False, init="xavier_uniform"
        )
        legacy_moe = MoEConfig(
            d_model=d_model,
            router=RouterConfig(n_experts=4, k=2, use_rms_norm=False),
            expert=ExpertConfig(d_model=d_model, init="xavier_uniform"),
        )
        legacy_block_cfg = BlockConfig(
            attn=legacy_attn, moe=legacy_moe, use_rms_norm=False
        )

        # Modern config (RMSNorm, RoPE, scaled init)
        modern_attn = AttentionConfig(
            n_heads=8, use_rope=True, use_rms_norm=True, init="scaled_xavier"
        )
        modern_moe = MoEConfig(
            d_model=d_model,
            router=RouterConfig(n_experts=4, k=2, use_rms_norm=True),
            expert=ExpertConfig(d_model=d_model, init="scaled_xavier"),
        )
        modern_block_cfg = BlockConfig(
            attn=modern_attn, moe=modern_moe, use_rms_norm=True
        )

        legacy_block = TransformerBlock(legacy_block_cfg, n_layers=6)
        modern_block = TransformerBlock(modern_block_cfg, n_layers=6)

        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, d_model)

        legacy_output, legacy_aux, legacy_metrics = legacy_block(x)
        modern_output, modern_aux, modern_metrics = modern_block(x)

        # Both should work but produce different outputs
        assert legacy_output.shape == modern_output.shape == x.shape
        assert torch.isfinite(legacy_output).all()
        assert torch.isfinite(modern_output).all()
        assert not torch.allclose(legacy_output, modern_output, rtol=1e-2)

    def test_transformer_block_gradient_flow(self):
        """Test that gradients flow through the modern transformer block."""
        d_model = 256

        block_cfg = BlockConfig(
            attn=AttentionConfig(n_heads=4, use_rope=True, use_rms_norm=True),
            moe=MoEConfig(
                d_model=d_model,
                router=RouterConfig(n_experts=4, k=2),
                expert=ExpertConfig(d_model=d_model),
            ),
            use_rms_norm=True,
        )

        block = TransformerBlock(block_cfg, n_layers=4)

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output, aux_loss, metrics = block(x)
        total_loss = output.sum() + aux_loss
        total_loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check that model parameters have gradients
        has_grad = []
        for param in block.parameters():
            if param.requires_grad:
                has_grad.append(
                    param.grad is not None and torch.isfinite(param.grad).all()
                )

        assert all(has_grad), "Some parameters don't have finite gradients"


class TestMoEModelIntegration:
    """Test full MoE model with modern components."""

    def test_moe_sequence_regressor_modern(self):
        """Test full MoE model with modern stability features."""
        d_model = 512

        model_cfg = ModelConfig(
            block=BlockConfig(
                attn=AttentionConfig(
                    n_heads=8, use_rope=True, use_rms_norm=True, init="scaled_xavier"
                ),
                moe=MoEConfig(
                    d_model=d_model,
                    router=RouterConfig(n_experts=8, k=2, use_rms_norm=True),
                    expert=ExpertConfig(d_model=d_model, init="scaled_xavier"),
                ),
                use_rms_norm=True,
            ),
            n_layers=4,
            input_dim=d_model,
            target_dim=1,
        )

        model = MoESequenceRegressor(model_cfg)

        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, d_model)
        targets = torch.randn(batch_size, 1)  # pooled target

        logits, metrics = model(x, targets)

        assert logits.shape == targets.shape
        assert torch.isfinite(logits).all()

        # Check metrics are returned and aux_total exists
        assert isinstance(metrics, dict)
        assert "aux_total" in metrics
        assert torch.isfinite(metrics["aux_total"]).all()

    def test_moe_model_different_sequence_lengths(self):
        """Test model with different sequence lengths (RoPE extrapolation)."""
        d_model = 256

        model_cfg = ModelConfig(
            block=BlockConfig(
                attn=AttentionConfig(
                    n_heads=4,
                    use_rope=True,
                    rope_max_seq_len=128,  # Cache for 128, test longer
                ),
                moe=MoEConfig(
                    d_model=d_model,
                    router=RouterConfig(n_experts=4, k=2),
                    expert=ExpertConfig(d_model=d_model),
                ),
            ),
            n_layers=2,
            input_dim=d_model,
        )

        model = MoESequenceRegressor(model_cfg)
        model.eval()

        batch_size = 2

        with torch.no_grad():
            # Test with sequence length within cache
            x_short = torch.randn(batch_size, 64, d_model)
            out_short, _ = model(x_short)

            # Test with sequence length beyond cache (extrapolation)
            x_long = torch.randn(batch_size, 192, d_model)
            out_long, _ = model(x_long)

        assert out_short.shape[0] == batch_size
        assert out_long.shape[0] == batch_size
        assert torch.isfinite(out_short).all()
        assert torch.isfinite(out_long).all()

    def test_moe_model_training_step(self):
        """Test a complete training step with the modern MoE model."""
        d_model = 256

        model_cfg = ModelConfig(
            block=BlockConfig(
                attn=AttentionConfig(n_heads=4, use_rope=True),
                moe=MoEConfig(
                    d_model=d_model,
                    router=RouterConfig(n_experts=4, k=2, load_balance_weight=1e-2),
                    expert=ExpertConfig(d_model=d_model),
                ),
            ),
            n_layers=2,
            input_dim=d_model,
            target_dim=1,
        )

        model = MoESequenceRegressor(model_cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, d_model)
        targets = torch.randn(batch_size, 1)

        # Forward pass
        logits, metrics = model(x, targets)

        # Compute loss manually
        main_loss = torch.nn.functional.mse_loss(logits, targets)
        aux_loss = metrics.get("aux_total", 0.0)
        loss = main_loss + 0.01 * aux_loss  # aux_weight = 0.01

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that loss is reasonable and gradients are finite
        assert torch.isfinite(loss).all()
        assert loss.item() > 0  # Should be positive (MSE + aux losses)

        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all()

    def test_build_moe_function(self):
        """Test the build_moe function with modern components."""
        d_model = 512
        moe_cfg = MoEConfig(
            d_model=d_model,
            router=RouterConfig(n_experts=8, k=2, use_rms_norm=True),
            expert=ExpertConfig(d_model=d_model, init="scaled_xavier"),
            dtype="fp32",
        )

        moe_block = build_moe(moe_cfg, n_layers=6)

        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, 512)

        output, metrics = moe_block(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        assert "aux_total" in metrics
        assert torch.isfinite(metrics["aux_total"]).all()


class TestConfigurationCompatibility:
    """Test that configurations work correctly with new features."""

    def test_config_defaults_modern(self):
        """Test that default configurations enable modern features."""
        # Default configs should enable modern features
        attn_cfg = AttentionConfig()
        assert attn_cfg.use_rope
        assert attn_cfg.use_rms_norm
        assert attn_cfg.init == "scaled_xavier"

        router_cfg = RouterConfig()
        assert router_cfg.use_rms_norm

        expert_cfg = ExpertConfig()
        assert expert_cfg.init == "scaled_xavier"

        block_cfg = BlockConfig()
        assert block_cfg.use_rms_norm

    def test_config_backwards_compatibility(self):
        """Test that we can disable modern features for backwards compatibility."""
        # Test that we can disable all modern features
        attn_cfg = AttentionConfig(
            use_rope=False, use_rms_norm=False, init="xavier_uniform"
        )

        router_cfg = RouterConfig(use_rms_norm=False)
        expert_cfg = ExpertConfig(init="xavier_uniform")
        block_cfg = BlockConfig(use_rms_norm=False)

        moe_cfg = MoEConfig(router=router_cfg, expert=expert_cfg)

        block_cfg.attn = attn_cfg
        block_cfg.moe = moe_cfg

        model_cfg = ModelConfig(
            block=block_cfg,
            n_layers=2,
        )

        # Should still work with legacy configuration
        model = MoESequenceRegressor(model_cfg)

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, 768)

        output, loss = model(x)

        assert output.shape[0] == batch_size
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
