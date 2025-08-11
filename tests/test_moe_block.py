"""
Integration tests for the complete MoE block.
"""

import pytest
import torch

from m.moe import (
    ExpertConfig,
    MoEConfig,
    RouterConfig,
    build_moe,
)


class TestMoEBlock:
    """Test the complete MoE feedforward block."""

    def test_moe_forward_pass(self, moe_config, device):
        """Test basic forward pass through MoE block."""
        moe = build_moe(moe_config).to(device)

        B, S, D = 2, 8, moe_config.d_model
        x = torch.randn(B, S, D, device=device)

        y, metrics = moe(x)

        # Check output shape
        assert y.shape == (B, S, D)

        # Check metrics
        assert "aux_total" in metrics
        assert "aux_load_balance" in metrics
        assert "fraction_dropped_tokens" in metrics
        assert "expert_utilization" in metrics

        # Expert utilization should be per expert
        assert metrics["expert_utilization"].shape == (moe_config.router.n_experts,)

    def test_dispatch_modes(self, device):
        """Test both dense and indices dispatch modes."""
        for mode in ["dense", "indices"]:
            cfg = MoEConfig(
                d_model=64,
                d_hidden=128,
                router=RouterConfig(
                    n_experts=4,
                    k=2,
                    dispatch_mode=mode,
                ),
                expert=ExpertConfig(d_model=64, d_hidden=128),
            )
            moe = build_moe(cfg).to(device)

            x = torch.randn(2, 8, 64, device=device)
            y, metrics = moe(x)

            assert y.shape == x.shape
            assert not torch.isnan(y).any()
            assert not torch.isinf(y).any()

    def test_fallback_policy(self, device):
        """Test fallback policies for dropped tokens."""
        # Test with zero fallback
        cfg_zero = MoEConfig(
            d_model=64,
            d_hidden=128,
            router=RouterConfig(n_experts=2, k=1, capacity_factor=0.5),  # Low capacity
            expert=ExpertConfig(d_model=64, d_hidden=128),
            fallback_policy="zero",
        )
        moe_zero = build_moe(cfg_zero).to(device)

        # Test with dense fallback
        cfg_dense = MoEConfig(
            d_model=64,
            d_hidden=128,
            router=RouterConfig(n_experts=2, k=1, capacity_factor=0.5),
            expert=ExpertConfig(d_model=64, d_hidden=128),
            fallback_policy="dense",
            fallback_weight=1.0,
        )
        moe_dense = build_moe(cfg_dense).to(device)

        x = torch.randn(4, 16, 64, device=device)  # Many tokens to ensure dropping

        y_zero, metrics_zero = moe_zero(x)
        y_dense, metrics_dense = moe_dense(x)

        # Both should produce valid outputs
        assert y_zero.shape == x.shape
        assert y_dense.shape == x.shape

        # Dense fallback should have different output (non-zero for dropped tokens)
        assert not torch.allclose(y_zero, y_dense)

        # Check that tokens were actually dropped
        assert metrics_zero["fraction_dropped_tokens"] > 0

    def test_gradient_flow(self, moe_config, device):
        """Test gradient flow through MoE block."""
        moe = build_moe(moe_config).to(device)

        x = torch.randn(2, 8, moe_config.d_model, device=device, requires_grad=True)
        y, metrics = moe(x)

        # Create loss including auxiliary losses
        loss = y.sum() + metrics["aux_total"]
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert moe.router.router.weight.grad is not None
        assert moe.experts.W1.grad is not None
        assert moe.experts.W2.grad is not None

        # Gradients should be non-zero
        assert x.grad.abs().sum() > 0
        assert moe.router.router.weight.grad.abs().sum() > 0

    def test_metrics_callback(self, moe_config, device):
        """Test metrics callback functionality."""
        moe = build_moe(moe_config).to(device)

        # Set up callback
        captured_metrics = []

        def callback(metrics):
            captured_metrics.append(metrics)

        moe.set_metrics_callback(callback)

        x = torch.randn(2, 8, moe_config.d_model, device=device)
        y, _ = moe(x)

        # Callback should have been called
        assert len(captured_metrics) == 1
        assert "expert_utilization" in captured_metrics[0]
        assert "fraction_dropped_tokens" in captured_metrics[0]

    def test_dtype_casting(self, device):
        """Test dtype casting for mixed precision."""
        for dtype_str in ["fp16", "bf16"]:
            if dtype_str == "fp16" and not torch.cuda.is_available():
                continue  # fp16 requires CUDA

            cfg = MoEConfig(
                d_model=64,
                d_hidden=128,
                router=RouterConfig(n_experts=2, k=1),
                expert=ExpertConfig(d_model=64, d_hidden=128),
                dtype=dtype_str,
            )
            moe = build_moe(cfg).to(device)

            # Check model dtype
            if dtype_str == "fp16":
                expected_dtype = torch.float16
            else:
                expected_dtype = torch.bfloat16

            # Model parameters should be in the specified dtype
            for param in moe.parameters():
                assert param.dtype == expected_dtype

            # Forward pass should work
            x = torch.randn(2, 8, 64, device=device, dtype=expected_dtype)
            y, metrics = moe(x)
            assert y.dtype == expected_dtype


class TestMoECapacityBehavior:
    """Test capacity-related behavior in MoE."""

    def test_capacity_calculation(self, device):
        """Test that capacity is calculated correctly."""
        cfg = MoEConfig(
            d_model=64,
            router=RouterConfig(
                n_experts=4,
                k=2,
                capacity_factor=1.5,
            ),
            expert=ExpertConfig(d_model=64, d_hidden=128),
        )
        moe = build_moe(cfg).to(device)

        B, S = 4, 8
        N = B * S
        x = torch.randn(B, S, 64, device=device)

        _, metrics = moe(x)

        # Calculate expected capacity
        expected_capacity = int(1.5 * (N * 2) / 4)  # capacity_factor * (N * k) / E
        assert metrics["capacity"].item() == expected_capacity

    def test_load_balancing(self, device):
        """Test that load balancing loss encourages uniform distribution."""
        cfg = MoEConfig(
            d_model=64,
            router=RouterConfig(
                n_experts=4,
                k=1,
                load_balance_weight=1.0,  # High weight for testing
                init_std=0.0,  # Start uniform
            ),
            expert=ExpertConfig(d_model=64, d_hidden=128),
        )
        moe = build_moe(cfg).to(device)
        optimizer = torch.optim.Adam(moe.parameters(), lr=0.1)

        # Train for a few steps
        for _ in range(10):
            x = torch.randn(8, 8, 64, device=device)
            y, metrics = moe(x)
            loss = y.sum() + metrics["aux_total"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check final distribution
        x_test = torch.randn(32, 8, 64, device=device)
        _, metrics_test = moe(x_test)

        # Expert utilization should be relatively uniform
        utilization = metrics_test["expert_utilization"]
        std = utilization.std()
        mean = utilization.mean()

        # Coefficient of variation should be reasonably small
        cv = std / mean if mean > 0 else 0
        assert cv < 1.0  # Reasonable threshold for uniformity


class TestMoEEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_expert(self, device):
        """Test MoE with single expert (degenerates to standard FFN)."""
        cfg = MoEConfig(
            d_model=64,
            router=RouterConfig(n_experts=1, k=1),
            expert=ExpertConfig(d_model=64, d_hidden=128),
        )
        moe = build_moe(cfg).to(device)

        x = torch.randn(2, 8, 64, device=device)
        y, metrics = moe(x)

        assert y.shape == x.shape
        # All tokens should go to the single expert
        assert metrics["fraction_dropped_tokens"].item() == 0

    def test_extreme_capacity(self, device):
        """Test with very low and very high capacity factors."""
        # Very low capacity - most tokens dropped
        cfg_low = MoEConfig(
            d_model=64,
            router=RouterConfig(
                n_experts=4,
                k=2,
                capacity_factor=0.1,  # Very low
            ),
            expert=ExpertConfig(d_model=64, d_hidden=128),
            fallback_policy="zero",
        )
        moe_low = build_moe(cfg_low).to(device)

        x = torch.randn(4, 8, 64, device=device)
        y_low, metrics_low = moe_low(x)

        # Many tokens should be dropped
        assert metrics_low["fraction_dropped_tokens"] > 0.5

        # Very high capacity - no tokens dropped
        cfg_high = MoEConfig(
            d_model=64,
            router=RouterConfig(
                n_experts=4,
                k=2,
                capacity_factor=10.0,  # Very high
            ),
            expert=ExpertConfig(d_model=64, d_hidden=128),
        )
        moe_high = build_moe(cfg_high).to(device)

        y_high, metrics_high = moe_high(x)

        # No tokens should be dropped
        assert metrics_high["fraction_dropped_tokens"].item() == 0

    def test_zero_batch_size(self, device):
        """Test with zero batch size."""
        # Skip on MPS due to known PyTorch MPS backend issue with empty tensors
        if device == "mps":
            pytest.skip("MPS backend has issues with empty tensors")

        cfg = MoEConfig(
            d_model=64,
            router=RouterConfig(n_experts=4, k=2),
            expert=ExpertConfig(d_model=64, d_hidden=128),
        )
        moe = build_moe(cfg).to(device)

        # Empty input
        x = torch.randn(0, 8, 64, device=device)
        y, metrics = moe(x)

        assert y.shape == (0, 8, 64)
        assert metrics["fraction_dropped_tokens"].item() == 0


@pytest.mark.parametrize(
    "router_type,dispatch_mode",
    [
        ("topk", "dense"),
        ("topk", "indices"),
        ("sinkhorn", "dense"),
        ("sinkhorn", "indices"),
    ],
)
class TestMoECombinations:
    """Test different combinations of routers and dispatch modes."""

    def test_router_dispatch_combinations(self, router_type, dispatch_mode, device):
        """Test that all combinations work correctly."""
        cfg = MoEConfig(
            d_model=64,
            d_hidden=128,
            router=RouterConfig(
                router_type=router_type,
                n_experts=4,
                k=2,
                dispatch_mode=dispatch_mode,
            ),
            expert=ExpertConfig(d_model=64, d_hidden=128),
        )
        moe = build_moe(cfg).to(device)

        x = torch.randn(2, 8, 64, device=device)
        y, metrics = moe(x)

        # All combinations should produce valid outputs
        assert y.shape == x.shape
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
        assert metrics["aux_total"].item() >= 0
