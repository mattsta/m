"""
3D Trajectory Training Pipeline

Training system for 3D trajectory learning with MoE transformers.
Handles multi-dimensional continuous sequence learning with specialized metrics.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml


class TrajectoryBatch(TypedDict):
    """Batch of trajectory samples for training."""

    input_sequence: torch.Tensor
    target_sequence: torch.Tensor
    trajectory_types: list[str]


from m.moe import (
    AttentionConfig,
    BlockConfig,
    ExpertConfig,
    ModelConfig,
    MoEConfig,
    MoESequenceRegressor,
    RouterConfig,
)
from m.rich_trainer_viz import (
    RichTrainerVisualizer,
    TrainingSnapshot,
    VisualizationConfig,
)
from m.training_viz import create_moe_visualizer

from .datasets import TrajectoryConfig, create_trajectory_dataset
from .visualization import Trajectory3DVisualizer


@dataclass(slots=True)
class TrajectoryTrainingConfig:
    """Configuration for 3D trajectory training."""

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 0.001
    max_steps: int = 20000
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0

    # Learning rate schedule
    lr_schedule: str = "cosine"
    min_learning_rate: float = 0.0001

    # Evaluation and saving
    eval_interval: int = 500
    eval_steps: int = 100
    save_interval: int = 2000

    # MoE specific
    aux_loss_weight: float = 0.1
    load_balance_weight: float = 0.1

    # Output configuration
    output_dir: str = "outputs/trajectory_3d"
    experiment_name: str = "trajectory_experiment"

    # Visualization
    save_visualizations: bool = True
    vis_interval: int = 1000


class Trajectory3DTrainer:
    """
    3D trajectory training pipeline with specialized metrics and visualization.

    Handles multi-dimensional continuous sequence learning with real-time monitoring.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrajectoryTrainingConfig,
        dataset_config: TrajectoryConfig,
        device: str = "auto",
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.dataset_config = dataset_config

        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🚀 Using device: {self.device}")

        # Create output directory
        self.output_dir = (
            Path(training_config.output_dir) / training_config.experiment_name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = MoESequenceRegressor(model_config).to(self.device)

        # Initialize datasets
        self.train_dataset = create_trajectory_dataset(dataset_config, seed=42)
        self.val_dataset = create_trajectory_dataset(dataset_config, seed=123)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Initialize visualization
        self.visualizer = Trajectory3DVisualizer(self.output_dir / "visualizations")

        # Always show real-time matplotlib charts during training
        # The save_visualizations setting only controls whether static images are saved
        # Pass the number of experts from model config for dynamic expert visualization
        n_experts = model_config.block.moe.router.n_experts
        self.real_time_vis = create_moe_visualizer(
            update_matplotlib=1, update_rich_tables=25, n_experts=n_experts
        )

        # Initialize rich CLI updater for trajectory training
        viz_config = VisualizationConfig(
            show_progress_bar=True,
            show_tables=True,
            table_update_interval=25,  # Show tables every 25 steps (like original)
            show_throughput=True,
            show_system_metrics=True,
            show_expert_utilization=True,
        )
        self.rich_visualizer = RichTrainerVisualizer(viz_config)

        # Metrics tracking
        self.step = 0
        self.best_val_loss = float("inf")
        self.metrics_history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "position_error": [],
            "velocity_error": [],
            "expert_entropy": [],
            "load_balance": [],
            "samples_per_sec": [],
        }
        self._last_aux_metrics = None
        self._last_grad_norm = None

        print(f"📊 Model: {self._count_parameters():,} parameters")
        print(f"🎯 Training for {training_config.max_steps:,} steps")

    def train(self) -> None:
        """Run the complete training loop."""
        print("🔥 Starting 3D trajectory training...")

        # Start rich CLI visualizer
        self.rich_visualizer.start(
            total_steps=self.training_config.max_steps,
            description="3D Trajectory Training",
        )

        start_time = time.time()

        try:
            while self.step < self.training_config.max_steps:
                # Training step
                train_metrics = self._training_step()

                # Update metrics (throughput calculation now handled by rich_updater)
                self._update_metrics(train_metrics)

                # Validation
                if self.step % self.training_config.eval_interval == 0:
                    val_metrics = self._validation_step()
                    self._update_metrics(val_metrics)

                    # Check for best model
                    if val_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]
                        self._save_checkpoint("best_model.pt")
                        # print(f"✨ New best model at step {self.step}: val_loss={val_metrics['val_loss']:.6f}")  # Handled by rich updater

                # Save checkpoint
                if (
                    self.step % self.training_config.save_interval == 0
                    and self.step > 0
                ):
                    self._save_checkpoint(f"checkpoint_step_{self.step}.pt")

                # Visualizations
                if (
                    self.training_config.save_visualizations
                    and self.step % self.training_config.vis_interval == 0
                    and self.step > 0
                ):
                    self._create_prediction_visualizations()

                self.step += 1

        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")

        # Final save and cleanup
        total_time = time.time() - start_time
        print(f"⏱️ Training completed in {total_time / 3600:.2f} hours")

        self._save_checkpoint("final_model.pt")
        self._save_training_summary()

        # Close real-time matplotlib visualizer
        self.real_time_vis.close()

        # Close rich visualizer
        self.rich_visualizer.stop()
        self.rich_visualizer.print_final_summary()

        print(f"💾 All outputs saved to: {self.output_dir}")

    def _training_step(self) -> dict[str, float]:
        """Execute single training step."""
        self.model.train()

        # Get training batch
        batch = self._get_batch(self.train_dataset, self.training_config.batch_size)

        # Forward pass
        input_seq = batch["input_sequence"].to(self.device)  # [B, seq_len, 3]
        target_seq = batch["target_sequence"].to(self.device)  # [B, pred_len, 3]

        # Predict next positions
        predictions, aux_metrics = self.model(input_seq)  # [B, seq_len, 3]

        # Use last prediction_length outputs for loss
        pred_len = target_seq.shape[1]
        pred_outputs = predictions[:, -pred_len:, :]  # [B, pred_len, 3]

        # Compute losses
        mse_loss = nn.functional.mse_loss(pred_outputs, target_seq)
        aux_loss = (
            aux_metrics.get("aux_total", torch.tensor(0.0))
            if aux_metrics
            else torch.tensor(0.0)
        )

        total_loss = mse_loss + self.training_config.aux_loss_weight * aux_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Calculate gradient norm before clipping
        grad_norm = None
        if any(p.grad is not None for p in self.model.parameters()):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float("inf"),  # Don't clip, just calculate norm
            ).item()

        # Gradient clipping
        if self.training_config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.training_config.gradient_clip_norm
            )

        self.optimizer.step()
        self.scheduler.step()

        # Compute 3D-specific metrics
        with torch.no_grad():
            position_error = torch.mean(torch.norm(pred_outputs - target_seq, dim=-1))

            # Velocity error (difference between consecutive positions)
            if pred_len > 1:
                pred_velocities = pred_outputs[:, 1:] - pred_outputs[:, :-1]
                target_velocities = target_seq[:, 1:] - target_seq[:, :-1]
                velocity_error = torch.mean(
                    torch.norm(pred_velocities - target_velocities, dim=-1)
                )
            else:
                velocity_error = torch.tensor(0.0)

        # Store aux_metrics for use in visualization
        self._last_aux_metrics = aux_metrics
        # Store gradient norm for system metrics
        self._last_grad_norm = grad_norm

        return {
            "train_loss": mse_loss.item(),
            "aux_loss": aux_loss.item(),
            "total_loss": total_loss.item(),
            "position_error": position_error.item(),
            "velocity_error": velocity_error.item(),
            "expert_entropy": aux_metrics.get("routing_entropy", 0.0)
            if aux_metrics
            else 0.0,
            "load_balance": aux_metrics.get("load_balance_factor", 0.0)
            if aux_metrics
            else 0.0,
        }

    def _validation_step(self) -> dict[str, float]:
        """Execute validation evaluation."""
        self.model.eval()

        val_losses = []
        position_errors = []
        velocity_errors = []

        with torch.no_grad():
            for _ in range(self.training_config.eval_steps):
                # Get validation batch
                batch = self._get_batch(
                    self.val_dataset, self.training_config.batch_size
                )

                input_seq = batch["input_sequence"].to(self.device)
                target_seq = batch["target_sequence"].to(self.device)

                # Forward pass
                predictions, aux_metrics = self.model(input_seq)

                # Extract predictions
                pred_len = target_seq.shape[1]
                pred_outputs = predictions[:, -pred_len:, :]

                # Losses
                mse_loss = nn.functional.mse_loss(pred_outputs, target_seq)
                val_losses.append(mse_loss.item())

                # 3D metrics
                position_error = torch.mean(
                    torch.norm(pred_outputs - target_seq, dim=-1)
                )
                position_errors.append(position_error.item())

                if pred_len > 1:
                    pred_velocities = pred_outputs[:, 1:] - pred_outputs[:, :-1]
                    target_velocities = target_seq[:, 1:] - target_seq[:, :-1]
                    velocity_error = torch.mean(
                        torch.norm(pred_velocities - target_velocities, dim=-1)
                    )
                    velocity_errors.append(velocity_error.item())

        return {
            "val_loss": float(np.mean(val_losses)),
            "val_position_error": float(np.mean(position_errors)),
            "val_velocity_error": float(np.mean(velocity_errors))
            if velocity_errors
            else 0.0,
        }

    def _get_batch(self, dataset, batch_size: int) -> TrajectoryBatch:
        """Get a batch from the dataset."""
        batch_samples = []
        for i, sample in enumerate(dataset):
            if i >= batch_size:
                break
            batch_samples.append(sample)

        # Stack tensors and create properly typed batch
        batch: TrajectoryBatch = {
            "input_sequence": torch.stack([s["input_sequence"] for s in batch_samples]),
            "target_sequence": torch.stack(
                [s["target_sequence"] for s in batch_samples]
            ),
            "trajectory_types": [s["trajectory_type"] for s in batch_samples],
        }

        return batch

    def _update_metrics(self, metrics: dict[str, float]) -> None:
        """Update metrics history and real-time visualization."""
        # Store metrics
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        # Get current learning rate
        lr = (
            self.scheduler.get_last_lr()[0]
            if hasattr(self.scheduler, "get_last_lr")
            else self.training_config.learning_rate
        )

        # Update real-time visualization with enriched metrics
        # Create enriched metrics for matplotlib visualization
        enriched_metrics = dict(metrics)  # Start with base metrics

        # Add learning rate
        enriched_metrics["learning_rate"] = lr

        # Add throughput metrics if we can calculate them
        current_time = time.time()
        if not hasattr(self, "_last_update_time"):
            self._last_update_time = current_time

        time_diff = current_time - self._last_update_time
        if time_diff > 0:
            enriched_metrics["steps_per_sec"] = 1.0 / time_diff
            enriched_metrics["samples_per_sec"] = (
                1.0 / time_diff
            ) * self.training_config.batch_size
            if self.dataset_config.sequence_length:
                enriched_metrics["tokens_per_sec"] = (
                    enriched_metrics["samples_per_sec"]
                    * self.dataset_config.sequence_length
                )

        self._last_update_time = current_time

        # Add expert utilization metrics from aux_metrics
        if hasattr(self, "_last_aux_metrics") and self._last_aux_metrics:
            if "expert_utilization" in self._last_aux_metrics:
                expert_util_tensor = self._last_aux_metrics["expert_utilization"]
                if expert_util_tensor is not None:
                    # Convert tensor to individual expert metrics for matplotlib
                    if hasattr(expert_util_tensor, "cpu"):
                        expert_util_list = expert_util_tensor.cpu().numpy()
                    else:
                        expert_util_list = expert_util_tensor

                    # Add individual expert utilization as separate metrics
                    for expert_id, util in enumerate(expert_util_list):
                        enriched_metrics[f"expert_{expert_id}_utilization"] = float(
                            util
                        )

                    # Note: expert_entropy and load_balance are always 0.0, so not including them
        self.real_time_vis.update_metrics(enriched_metrics)

        # Calculate GPU memory if available
        gpu_memory_mb = None
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and self.device == "mps"
        ):
            gpu_memory_mb = torch.mps.current_allocated_memory() / 1024 / 1024

        # Create expert utilization dict from aux_metrics
        expert_utilization: dict[int, float] = {}
        if hasattr(self, "_last_aux_metrics") and self._last_aux_metrics:
            if "expert_utilization" in self._last_aux_metrics:
                expert_util_tensor = self._last_aux_metrics["expert_utilization"]
                if expert_util_tensor is not None:
                    # Convert tensor to dict with expert IDs
                    if hasattr(expert_util_tensor, "cpu"):
                        expert_util_list = expert_util_tensor.cpu().numpy()
                    else:
                        expert_util_list = expert_util_tensor

                    for expert_id, util in enumerate(expert_util_list):
                        expert_utilization[expert_id] = float(util)

        # Create simple training snapshot - visualizer handles all calculations internally
        snapshot = TrainingSnapshot(
            step=self.step,
            train_loss=metrics.get("train_loss", 0.0),
            val_loss=metrics.get("val_loss"),
            learning_rate=lr,
            aux_loss=metrics.get("aux_loss"),
            expert_utilization=expert_utilization,
            batch_size=self.training_config.batch_size,
            sequence_length=self.dataset_config.sequence_length,
            gpu_memory_mb=gpu_memory_mb,
            gradient_norm=self._last_grad_norm,
            custom_metrics={
                "position_error": metrics.get("position_error", 0.0),
                "velocity_error": metrics.get("velocity_error", 0.0),
                "val_position_error": metrics.get("val_position_error"),
                "val_velocity_error": metrics.get("val_velocity_error"),
                "expert_entropy": metrics.get("expert_entropy"),
            },
        )

        # Update rich visualizer
        self.rich_visualizer.update(snapshot)

    def _create_prediction_visualizations(self) -> None:
        """Create visualization of model predictions."""
        self.model.eval()

        with torch.no_grad():
            # Get a validation sample
            sample = next(iter(self.val_dataset))
            input_seq = sample["input_sequence"].unsqueeze(0).to(self.device)
            target_seq = sample["target_sequence"]

            # Generate prediction
            predictions, _ = self.model(input_seq)
            pred_len = target_seq.shape[0]
            pred_output = predictions[0, -pred_len:, :].cpu()

            # Create comparison visualization
            save_path = (
                self.output_dir / "visualizations" / f"prediction_step_{self.step}.png"
            )
            self.visualizer.plot_prediction_comparison(
                input_seq[0].cpu(),
                target_seq,
                pred_output,
                sample["trajectory_type"],
                str(save_path),
            )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.training_config.lr_schedule == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.max_steps,
                eta_min=self.training_config.min_learning_rate,
            )
        elif self.training_config.lr_schedule == "constant":
            return optim.lr_scheduler.ConstantLR(self.optimizer)
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer)

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "model_config": self.model_config,
            "training_config": self.training_config,
            "dataset_config": self.dataset_config,
            "metrics_history": self.metrics_history,
        }

        torch.save(checkpoint, self.output_dir / filename)

    def _save_training_summary(self) -> None:
        """Save training summary and final visualizations."""
        # Save metrics plot
        self.visualizer.plot_training_progress(
            self.metrics_history, str(self.output_dir / "training_progress.png")
        )

        # Save training summary
        summary = {
            "total_steps": self.step,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.metrics_history["train_loss"][-1]
            if self.metrics_history["train_loss"]
            else 0,
            "model_parameters": self._count_parameters(),
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "dataset_config": asdict(self.dataset_config),
        }

        with open(self.output_dir / "training_summary.yaml", "w") as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)

    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def load_config_from_yaml(
    config_path: str,
) -> tuple[ModelConfig, TrajectoryTrainingConfig, TrajectoryConfig]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Convert nested dictionaries to appropriate config objects
    # Build model config
    model_cfg = config["model"]
    block_cfg = model_cfg["block"]
    attn_cfg = AttentionConfig(**block_cfg["attn"])
    router_cfg = RouterConfig(**block_cfg["moe"]["router"])
    expert_cfg = ExpertConfig(**block_cfg["moe"]["expert"])
    moe_cfg = MoEConfig(
        d_model=block_cfg["moe"]["d_model"],
        router=router_cfg,
        expert=expert_cfg,
        **{
            k: v
            for k, v in block_cfg["moe"].items()
            if k not in ["router", "expert", "d_model"]
        },
    )
    block_config = BlockConfig(
        attn=attn_cfg, moe=moe_cfg, use_rms_norm=block_cfg["use_rms_norm"]
    )
    model_config = ModelConfig(
        block=block_config,
        n_layers=model_cfg["n_layers"],
        input_dim=model_cfg["input_dim"],
        target_dim=model_cfg["target_dim"],
        pool=model_cfg["pool"],
    )

    # Build training config
    training_config = TrajectoryTrainingConfig(**config["training"])

    # Build dataset config
    dataset_config = TrajectoryConfig(**config["dataset"])

    return model_config, training_config, dataset_config


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="3D Trajectory MoE Training")
    parser.add_argument("config", help="Path to training configuration file")
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument("--name", help="Experiment name (optional)")

    args = parser.parse_args()

    print("🎯 3D Trajectory Learning with MoE Transformers")
    print(f"📁 Config: {args.config}")
    if args.name:
        print(f"🏷️  Experiment: {args.name}")

    # Load configuration
    model_config, training_config, dataset_config = load_config_from_yaml(args.config)

    # Override experiment name if provided
    if args.name:
        training_config.experiment_name = args.name

    # Create and run trainer
    trainer = Trajectory3DTrainer(
        model_config=model_config,
        training_config=training_config,
        dataset_config=dataset_config,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
