"""
Pytest configuration and shared fixtures for MoE platform tests.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from m.moe import (
    AttentionConfig,
    BlockConfig,
    ExpertConfig,
    ModelConfig,
    MoEConfig,
    MoESequenceRegressor,
    RouterConfig,
    build_moe,
)

# ========================= Configuration Fixtures =========================


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture
def router_config():
    """Basic router configuration for testing."""
    return RouterConfig(
        router_type="topk",
        n_experts=4,
        k=2,
        capacity_factor=1.25,
        load_balance_weight=0.01,
        z_loss_weight=0.001,
    )


@pytest.fixture
def expert_config():
    """Basic expert configuration for testing."""
    return ExpertConfig(
        expert_type="ffn",
        d_model=128,
        d_hidden=256,
        activation="swiglu",
        dropout=0.0,
        bias=True,
    )


@pytest.fixture
def moe_config(router_config, expert_config):
    """Complete MoE configuration for testing."""
    return MoEConfig(
        d_model=128,
        d_hidden=256,
        router=router_config,
        expert=expert_config,
        fallback_policy="dense",
        fallback_weight=0.5,
    )


@pytest.fixture
def attention_config():
    """Attention configuration for testing."""
    return AttentionConfig(
        n_heads=4,
        attn_dropout=0.0,
        resid_dropout=0.0,
        bias=True,
        causal=True,
    )


@pytest.fixture
def block_config(attention_config, moe_config):
    """Transformer block configuration for testing."""
    return BlockConfig(
        attn=attention_config,
        moe=moe_config,
        prenorm=True,
    )


@pytest.fixture
def model_config(block_config):
    """Complete model configuration for testing."""
    return ModelConfig(
        block=block_config,
        n_layers=2,
        input_dim=128,
        target_dim=1,
        pool="mean",
        torch_compile=False,
    )


# ========================= Model Fixtures =========================


@pytest.fixture
def moe_block(moe_config):
    """Create a MoE feedforward block."""
    return build_moe(moe_config)


@pytest.fixture
def model(model_config):
    """Create a complete MoE model."""
    return MoESequenceRegressor(model_config)


@pytest.fixture
def small_model(device):
    """Create a small model for quick testing."""
    cfg = ModelConfig(
        block=BlockConfig(
            attn=AttentionConfig(n_heads=2),
            moe=MoEConfig(
                d_model=64,
                d_hidden=128,
                router=RouterConfig(n_experts=2, k=1),
                expert=ExpertConfig(d_model=64, d_hidden=128),
            ),
        ),
        n_layers=1,
        input_dim=64,
        target_dim=1,
    )
    return MoESequenceRegressor(cfg).to(device)


# ========================= Data Fixtures =========================


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 32


@pytest.fixture
def sample_input(batch_size, seq_len, model_config):
    """Generate sample input tensor."""
    return torch.randn(batch_size, seq_len, model_config.input_dim)


@pytest.fixture
def sample_target(batch_size, model_config):
    """Generate sample target tensor."""
    return torch.randn(batch_size, model_config.target_dim)


@pytest.fixture
def sample_tokens(batch_size, seq_len):
    """Generate sample token IDs for inference testing."""
    return torch.randint(0, 1000, (batch_size, seq_len))


# ========================= File System Fixtures =========================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def checkpoint_dir(temp_dir):
    """Create a directory for checkpoint testing."""
    ckpt_dir = temp_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    return ckpt_dir


# ========================= Training Fixtures =========================


@pytest.fixture
def optimizer(model):
    """Create an optimizer for the model."""
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


@pytest.fixture
def saved_checkpoint(model, optimizer, checkpoint_dir):
    """Save a checkpoint and return its path."""
    checkpoint_path = checkpoint_dir / "test_checkpoint.pt"

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "model": model.cfg.__dict__,
        },
        "trainer": {
            "global_step": 100,
            "epoch": 5,
            "best_val": 0.5,
        },
    }

    torch.save(state, checkpoint_path)
    return checkpoint_path


# ========================= Utility Functions =========================


@pytest.fixture
def assert_tensors_close():
    """Utility function to assert tensors are close."""

    def _assert_close(a, b, rtol=1e-5, atol=1e-6):
        assert torch.allclose(a, b, rtol=rtol, atol=atol), (
            f"Tensors not close: max diff = {(a - b).abs().max()}"
        )

    return _assert_close


@pytest.fixture
def count_parameters():
    """Utility to count model parameters."""

    def _count(model):
        return sum(p.numel() for p in model.parameters())

    return _count


# ========================= Test Markers =========================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "distributed: marks distributed training tests")
    config.addinivalue_line("markers", "inference: marks inference-specific tests")


# ========================= Test Settings =========================


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
