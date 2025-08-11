#!/usr/bin/env python3
"""
Quick test of 3D trajectory training components.
"""

from pathlib import Path

import torch

from m.moe import (
    AttentionConfig,
    BlockConfig,
    ExpertConfig,
    ModelConfig,
    MoEConfig,
    RouterConfig,
)

# Test imports
from .datasets import TrajectoryConfig, create_trajectory_dataset
from .training import Trajectory3DTrainer, TrajectoryTrainingConfig

print("✅ All imports successful")


def test_minimal_training():
    """Test minimal training setup."""
    print("🧪 Testing minimal 3D trajectory training setup...")

    # Create minimal configurations
    dataset_config = TrajectoryConfig(
        sequence_length=16, prediction_length=4, sampling_rate=10.0, noise_std=0.01
    )

    training_config = TrajectoryTrainingConfig(
        batch_size=4,
        learning_rate=0.01,
        max_steps=10,  # Very few steps for testing
        eval_interval=5,
        eval_steps=2,
        save_visualizations=False,
        output_dir="test_outputs",
        experiment_name="minimal_test",
    )

    # Create minimal model config
    attn_config = AttentionConfig(
        n_heads=2, d_head=8, d_model=16, dropout=0.0, causal=True
    )

    router_config = RouterConfig(type="topk", temperature=1.0, straight_through=True)

    expert_config = ExpertConfig(d_ff=32, dropout=0.0, activation="gelu")

    moe_config = MoEConfig(
        d_model=16, n_experts=2, k=1, router=router_config, expert=expert_config
    )

    block_config = BlockConfig(attn=attn_config, moe=moe_config, use_rms_norm=True)

    model_config = ModelConfig(
        block=block_config, n_layers=1, input_dim=3, target_dim=3, pool="none"
    )

    print(
        f"📊 Model config: {model_config.n_layers} layers, {moe_config.n_experts} experts"
    )
    print(
        f"🎯 Dataset config: {dataset_config.sequence_length} -> {dataset_config.prediction_length}"
    )

    # Create trainer
    try:
        trainer = Trajectory3DTrainer(
            model_config=model_config,
            training_config=training_config,
            dataset_config=dataset_config,
            device="cpu",
        )
        print("✅ Trainer created successfully")

        # Test single training step
        print("🔥 Testing single training step...")
        trainer.model.train()

        # Get a batch manually
        batch = trainer._get_batch(trainer.train_dataset, 2)
        print(
            f"   Batch shapes: input={batch['input_sequence'].shape}, target={batch['target_sequence'].shape}"
        )

        # Test forward pass
        input_seq = batch["input_sequence"].to(trainer.device)
        target_seq = batch["target_sequence"].to(trainer.device)

        predictions, aux_metrics = trainer.model(input_seq)
        print(f"   Prediction shape: {predictions.shape}")
        print(f"   Aux metrics keys: {list(aux_metrics.keys())}")

        # Test loss computation
        pred_len = target_seq.shape[1]
        pred_outputs = predictions[:, -pred_len:, :]
        mse_loss = torch.nn.functional.mse_loss(pred_outputs, target_seq)
        print(f"   MSE Loss: {mse_loss.item():.6f}")

        print("✅ Forward pass working correctly")

        # Test a few training steps (don't run full training)
        print("🚀 Testing a few training steps...")

        for step in range(3):
            metrics = trainer._training_step()
            print(
                f"   Step {step}: loss={metrics['train_loss']:.6f}, pos_error={metrics['position_error']:.4f}"
            )

        print("✅ Training steps working correctly")

        # Clean up
        import shutil

        if Path("test_outputs").exists():
            shutil.rmtree("test_outputs")

        return True

    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dataset_generation():
    """Test dataset generation."""
    print("🧪 Testing 3D trajectory dataset...")

    config = TrajectoryConfig(
        sequence_length=32, prediction_length=8, sampling_rate=20.0
    )

    dataset = create_trajectory_dataset(config, seed=42)

    # Test multiple samples
    samples = []
    for i, sample in enumerate(dataset):
        if i >= 3:
            break
        samples.append(sample)

    print(f"✅ Generated {len(samples)} samples")
    for i, sample in enumerate(samples):
        print(
            f"   Sample {i}: {sample['trajectory_type']}, "
            f"input={sample['input_sequence'].shape}, "
            f"target={sample['target_sequence'].shape}"
        )

    return True


if __name__ == "__main__":
    print("🎯 3D Trajectory Learning - Component Tests")
    print("=" * 50)

    # Test dataset
    if test_dataset_generation():
        print("✅ Dataset generation test passed")
    else:
        print("❌ Dataset generation test failed")
        exit(1)

    # Test training
    if test_minimal_training():
        print("✅ Training pipeline test passed")
    else:
        print("❌ Training pipeline test failed")
        exit(1)

    print("=" * 50)
    print("🎉 All component tests passed!")
    print("📚 Ready for full 3D trajectory learning experiments")
