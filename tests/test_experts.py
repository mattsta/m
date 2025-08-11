"""
Tests for expert networks in MoE.
"""

import pytest
import torch

from m.moe import (
    DenseFFN,
    ExpertConfig,
    ExpertFFN,
)


class TestExpertFFN:
    """Test expert feed-forward networks."""

    def test_expert_initialization(self, expert_config):
        """Test expert network initialization."""
        n_experts = 4
        expert = ExpertFFN(cfg=expert_config, n_experts=n_experts)

        # Check weight shapes
        E, D, H = n_experts, expert_config.d_model, expert_config.d_hidden

        if expert_config.activation in ("swiglu", "geglu", "reglu"):
            # Gated activation needs 2x hidden size
            assert expert.W1.shape == (E, D, 2 * H)
        else:
            assert expert.W1.shape == (E, D, H)

        assert expert.W2.shape == (E, H, D)

        # Check bias
        if expert_config.bias:
            assert expert.b1 is not None
            assert expert.b2 is not None
        else:
            assert expert.b1 is None
            assert expert.b2 is None

    @pytest.mark.parametrize(
        "activation", ["gelu", "relu", "silu", "swiglu", "geglu", "reglu"]
    )
    def test_activation_functions(self, activation, device):
        """Test different activation functions."""
        cfg = ExpertConfig(
            d_model=64,
            d_hidden=128,
            activation=activation,
            dropout=0.0,
        )
        expert = ExpertFFN(cfg=cfg, n_experts=2).to(device)

        # Input: [E, C, D]
        x = torch.randn(2, 8, 64, device=device)
        y = expert(x)

        # Output should have same shape
        assert y.shape == x.shape

        # Check that activation is applied (output differs from linear)
        with torch.no_grad():
            # Compute linear transformation only
            if cfg.activation in ("swiglu", "geglu", "reglu"):
                h_linear = torch.bmm(x, expert.W1[:, :, :128])
            else:
                h_linear = torch.bmm(x, expert.W1)
            y_linear = torch.bmm(h_linear, expert.W2)

        assert not torch.allclose(y, y_linear, atol=1e-3)

    def test_grouped_gemm_vs_einsum(self, expert_config, device):
        """Test equivalence of grouped GEMM and einsum implementations."""
        expert_config.grouped_gemm = True
        expert1 = ExpertFFN(cfg=expert_config, n_experts=4).to(device)

        expert_config.grouped_gemm = False
        expert2 = ExpertFFN(cfg=expert_config, n_experts=4).to(device)

        # Copy weights
        expert2.W1.data = expert1.W1.data.clone()
        expert2.W2.data = expert1.W2.data.clone()
        if expert_config.bias:
            expert2.b1.data = expert1.b1.data.clone()
            expert2.b2.data = expert1.b2.data.clone()

        x = torch.randn(4, 8, expert_config.d_model, device=device)

        y1 = expert1(x)
        y2 = expert2(x)

        assert torch.allclose(y1, y2, atol=1e-5)

    def test_dropout(self, expert_config, device):
        """Test dropout in expert networks."""
        expert_config.dropout = 0.5
        expert = ExpertFFN(cfg=expert_config, n_experts=2).to(device)

        x = torch.randn(2, 8, expert_config.d_model, device=device)

        # Training mode - outputs should differ
        expert.train()
        y1 = expert(x)
        y2 = expert(x)
        assert not torch.allclose(y1, y2)

        # Eval mode - outputs should be identical
        expert.eval()
        y3 = expert(x)
        y4 = expert(x)
        assert torch.allclose(y3, y4)

    def test_gradient_checkpointing(self, expert_config, device):
        """Test activation checkpointing for memory efficiency."""
        expert_config.checkpoint_experts = True
        expert = ExpertFFN(cfg=expert_config, n_experts=4).to(device)

        x = torch.randn(4, 8, expert_config.d_model, device=device, requires_grad=True)

        # Forward and backward should work with checkpointing
        expert.train()
        y = expert(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert expert.W1.grad is not None
        assert expert.W2.grad is not None


class TestDenseFFN:
    """Test dense fallback network."""

    def test_dense_ffn(self, expert_config, device):
        """Test dense FFN as fallback."""
        dense = DenseFFN(cfg=expert_config).to(device)

        # Check layer shapes
        assert dense.W1.in_features == expert_config.d_model
        if expert_config.activation in ("swiglu", "geglu", "reglu"):
            assert dense.W1.out_features == 2 * expert_config.d_hidden
        else:
            assert dense.W1.out_features == expert_config.d_hidden

        assert dense.W2.in_features == expert_config.d_hidden
        assert dense.W2.out_features == expert_config.d_model

        # Test forward pass
        x = torch.randn(4, 8, expert_config.d_model, device=device)
        y = dense(x)
        assert y.shape == x.shape

    def test_dense_vs_expert_equivalence(self, expert_config, device):
        """Test that dense FFN matches single expert behavior."""
        dense = DenseFFN(cfg=expert_config).to(device)
        expert = ExpertFFN(cfg=expert_config, n_experts=1).to(device)

        # Copy weights from expert to dense
        dense.W1.weight.data = expert.W1[0].data.t()
        dense.W2.weight.data = expert.W2[0].data.t()
        if expert_config.bias:
            dense.W1.bias.data = expert.b1[0].data
            dense.W2.bias.data = expert.b2[0].data

        x = torch.randn(4, 8, expert_config.d_model, device=device)

        # Dense output
        y_dense = dense(x)

        # Expert output (need to add batch dimension for expert)
        x_expert = x.unsqueeze(0)  # [1, B*S, D]
        x_expert = x_expert.view(1, -1, expert_config.d_model)  # [1, B*S, D]
        y_expert = expert(x_expert)
        y_expert = y_expert.squeeze(0).view(4, 8, expert_config.d_model)

        assert torch.allclose(y_dense, y_expert, atol=1e-5)


class TestExpertGradients:
    """Test gradient flow through experts."""

    def test_expert_gradients(self, expert_config, device):
        """Test that gradients flow correctly through experts."""
        expert = ExpertFFN(cfg=expert_config, n_experts=4).to(device)

        x = torch.randn(4, 8, expert_config.d_model, device=device, requires_grad=True)
        y = expert(x)

        loss = y.sum()
        loss.backward()

        # Check all gradients exist
        assert x.grad is not None
        assert expert.W1.grad is not None
        assert expert.W2.grad is not None

        # Gradients should be non-zero
        assert x.grad.abs().sum() > 0
        assert expert.W1.grad.abs().sum() > 0
        assert expert.W2.grad.abs().sum() > 0

    def test_dense_gradients(self, expert_config, device):
        """Test gradient flow through dense fallback."""
        dense = DenseFFN(cfg=expert_config).to(device)

        x = torch.randn(4, 8, expert_config.d_model, device=device, requires_grad=True)
        y = dense(x)

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert dense.W1.weight.grad is not None
        assert dense.W2.weight.grad is not None
