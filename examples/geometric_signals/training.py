"""
Training pipeline for geometric signal learning with real-time monitoring.
Provides comprehensive training with visualization and expert analysis.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader

# Import our MoE system
from m.moe import (
    AttentionConfig,
    BlockConfig,
    ExpertConfig,
    ModelConfig,
    MoEConfig,
    MoESequenceRegressor,
    RouterConfig,
)

# Import example modules
from .datasets import create_signal_dataset
from .visualization import (
    LiveTrainingVisualizer,
    TrainingMetrics,
    VisualizationConfig,
    save_metrics_csv,
)


@dataclass(slots=True)
class SignalTrainingConfig:
    """Configuration for signal learning experiments."""

    # Model configuration
    model: dict[str, Any]

    # Training configuration
    training: dict[str, Any]

    # Dataset configuration
    dataset: dict[str, Any]

    # Optimizer configuration
    optimizer: dict[str, Any]

    # Experiment metadata
    experiment_name: str = "signal_learning"
    output_dir: Path = Path("outputs/geometric_signals")
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda, mps


class SignalTrainer:
    """Specialized trainer for signal learning tasks."""

    def __init__(self, config: SignalTrainingConfig):
        self.config = config
        self.device = self._setup_device()

        # Set random seeds
        torch.manual_seed(config.seed)

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = self._create_model()

        # Set up metrics callback for MoE routing statistics
        self.routing_metrics: dict[str, Any] = {}
        self._setup_moe_metrics_callback()

        # Initialize datasets
        self.train_dataset, self.val_dataset = self._create_datasets()

        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Initialize visualization (matplotlib plots)
        vis_config = VisualizationConfig(
            output_dir=self.config.output_dir,
            update_interval=100,
            save_interval=1000,
            show_plots=False,
        )
        self.visualizer = LiveTrainingVisualizer(vis_config)

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.metrics_history: list[TrainingMetrics] = []

        # EMA tracking for all metrics
        self.samples_per_second_ema: float | None = None
        self.batches_per_second_ema: float | None = None
        self.steps_per_second_ema: float | None = None
        self.tokens_per_second_ema: float | None = None
        self.loss_per_second_ema: float | None = None
        self.ema_alpha = 0.1  # EMA smoothing factor

        # For loss rate calculation
        self.previous_loss: float | None = None
        self.previous_loss_time: float | None = None

        print("Initialized SignalTrainer:")
        print(f"  Device: {self.device}")
        print(
            f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        print(f"  Output directory: {self.config.output_dir}")

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)

        print(f"Using device: {device}")
        return device

    def _setup_moe_metrics_callback(self):
        """Set up callback to capture MoE routing metrics."""

        def metrics_callback(metrics: dict):
            # Process the metrics to extract expert utilization
            processed_metrics = {}
            if "expert_utilization" in metrics:
                # Convert tensor to dict format expected by visualization
                expert_util_tensor = metrics["expert_utilization"]
                if hasattr(expert_util_tensor, "cpu"):
                    expert_util = expert_util_tensor.cpu().numpy()
                    processed_metrics["expert_utilization"] = {
                        i: float(expert_util[i]) for i in range(len(expert_util))
                    }
                else:
                    processed_metrics["expert_utilization"] = metrics[
                        "expert_utilization"
                    ]

            # Store all processed metrics
            processed_metrics.update(
                {k: v for k, v in metrics.items() if k != "expert_utilization"}
            )
            self.routing_metrics = processed_metrics

        # Find all MoE layers and set the callback
        for name, module in self.model.named_modules():
            if hasattr(module, "set_metrics_callback"):
                module.set_metrics_callback(metrics_callback)

    def _create_model(self) -> MoESequenceRegressor:
        """Create and initialize the MoE model."""
        # Config classes already imported at top

        # Parse the nested config structure
        model_cfg_dict = self.config.model

        # Create the nested configs
        attn_config = AttentionConfig(**model_cfg_dict["block"]["attn"])

        # Filter router config to only include supported parameters
        router_cfg_dict = model_cfg_dict["block"]["moe"]["router"]
        # Get the RouterConfig field names to filter unsupported parameters
        # fields already imported at top

        router_field_names = {field.name for field in fields(RouterConfig)}
        filtered_router_cfg = {
            k: v for k, v in router_cfg_dict.items() if k in router_field_names
        }
        router_config = RouterConfig(**filtered_router_cfg)

        expert_config = ExpertConfig(**model_cfg_dict["block"]["moe"]["expert"])
        moe_config = MoEConfig(
            d_model=model_cfg_dict["block"]["moe"]["d_model"],
            router=router_config,
            expert=expert_config,
        )
        block_config = BlockConfig(
            attn=attn_config,
            moe=moe_config,
            use_rms_norm=model_cfg_dict["block"]["use_rms_norm"],
        )

        # Create the main model config
        model_config = ModelConfig(
            block=block_config,
            n_layers=model_cfg_dict["n_layers"],
            input_dim=model_cfg_dict["input_dim"],
            target_dim=model_cfg_dict["target_dim"],
            pool=model_cfg_dict.get("pool", "mean"),
        )

        model = MoESequenceRegressor(model_config)
        model.to(self.device)
        return model

    def _create_datasets(
        self,
    ) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Create training and validation datasets."""
        dataset_config = self.config.dataset

        train_dataset = create_signal_dataset(
            dataset_type=dataset_config["type"],
            sequence_length=dataset_config["sequence_length"],
            prediction_length=dataset_config["prediction_length"],
            num_samples=dataset_config["num_samples"],
            seed=self.config.seed,
        )

        # Validation dataset with different seed
        val_dataset = create_signal_dataset(
            dataset_type=dataset_config["type"],
            sequence_length=dataset_config["sequence_length"],
            prediction_length=dataset_config["prediction_length"],
            num_samples=dataset_config["num_samples"] // 10,  # Smaller validation set
            seed=self.config.seed + 10000,
        )

        return train_dataset, val_dataset

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        opt_config = self.config.optimizer
        opt_type = opt_config["type"].lower()

        if opt_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training["learning_rate"],
                betas=opt_config.get("betas", [0.9, 0.999]),
                eps=float(opt_config.get("eps", 1e-8)),
                weight_decay=self.config.training.get("weight_decay", 0.01),
            )
        elif opt_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training["learning_rate"],
                betas=opt_config.get("betas", [0.9, 0.999]),
                eps=float(opt_config.get("eps", 1e-8)),
                weight_decay=self.config.training.get("weight_decay", 0.01),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

    def _create_scheduler(self) -> _LRScheduler | None:
        """Create learning rate scheduler."""
        scheduler_type = self.config.training.get("scheduler")
        if not scheduler_type:
            return None

        if scheduler_type == "cosine_with_warmup":
            # Simple cosine scheduler - can be enhanced with warmup
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training["max_steps"],
                eta_min=self.config.training["learning_rate"] * 0.01,
            )
            return cast(_LRScheduler, scheduler)

        return None

    def _collect_system_metrics(self) -> dict[str, float]:
        """Collect system performance metrics."""
        metrics = {}

        if self.device.type == "cuda":
            try:
                # GPU memory usage
                gpu_memory_used = torch.cuda.memory_allocated(self.device) / (
                    1024**2
                )  # MB
                gpu_memory_total = torch.cuda.max_memory_allocated(self.device) / (
                    1024**2
                )  # MB
                metrics["gpu_memory_used_mb"] = gpu_memory_used
                metrics["gpu_memory_total_mb"] = gpu_memory_total

                # GPU utilization would require nvidia-ml-py, skip for now
                # metrics["gpu_utilization_percent"] = get_gpu_utilization()

            except Exception:
                pass  # Skip if CUDA not available

        elif self.device.type == "mps":
            try:
                # Apple Silicon GPU memory (if available)
                gpu_memory_used = torch.mps.current_allocated_memory() / (1024**2)  # MB
                metrics["gpu_memory_used_mb"] = gpu_memory_used
            except Exception:
                pass  # Skip if not available

        return metrics

    def _compute_expert_utilization(self) -> dict[int, float]:
        """Compute current expert utilization statistics."""
        n_experts = self.config.model["block"]["moe"]["router"]["n_experts"]

        # Use real routing metrics if available
        if self.routing_metrics and "expert_utilization" in self.routing_metrics:
            return self.routing_metrics["expert_utilization"]

        # If no routing metrics available yet, return uniform distribution
        return {i: 1.0 / n_experts for i in range(n_experts)}

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
        """Perform single training step."""
        self.model.train()

        input_seq, target_seq = batch
        input_seq = input_seq.to(self.device)
        target_seq = target_seq.to(self.device)

        # For sequence prediction, concatenate input and target
        # Model will see: [input_seq | target_seq] and predict entire sequence
        # We'll compute loss only on the target portion
        full_seq = torch.cat(
            [input_seq, target_seq], dim=1
        )  # [B, input_len + pred_len, 1]

        # Forward pass - no targets passed, we'll compute loss manually
        logits, _ = self.model(full_seq, targets=None)

        # Handle different pooling modes
        if len(logits.shape) == 2:
            # Pooled output [B, target_dim] - use full sequence target
            pred_logits = logits
            # For pooled mode, use the mean of the target sequence
            target_mean = target_seq.mean(dim=1)  # [B, target_dim]
            mse_loss = F.mse_loss(pred_logits, target_mean)
        else:
            # Sequence output [B, S, target_dim] - extract predictions for target portion
            input_len = input_seq.shape[1]
            pred_logits = logits[:, input_len:, :]  # [B, pred_len, target_dim]
            # Compute loss only on prediction portion
            mse_loss = F.mse_loss(pred_logits, target_seq)

        # Get auxiliary losses by running a dummy forward pass (inefficient but works)
        # TODO: Make this more efficient by extracting aux losses properly
        if len(logits.shape) == 2:
            # For pooled mode, create compatible target for aux loss
            target_mean_full = full_seq.mean(dim=1)  # [B, target_dim]
            _, full_loss = self.model(full_seq, targets=target_mean_full)
            if full_loss is not None and isinstance(full_loss, torch.Tensor):
                # Estimate aux loss as difference
                full_mse = F.mse_loss(logits, target_mean_full)
                aux_loss = full_loss - full_mse
                aux_loss = torch.clamp(aux_loss, min=0)  # Ensure non-negative
            else:
                aux_loss = 0
        else:
            # For sequence mode, use original approach
            _, full_loss = self.model(full_seq, targets=full_seq)
            if full_loss is not None and isinstance(full_loss, torch.Tensor):
                # Estimate aux loss as difference
                full_mse = F.mse_loss(logits, full_seq)
                aux_loss = full_loss - full_mse
                aux_loss = torch.clamp(aux_loss, min=0)  # Ensure non-negative
            else:
                aux_loss = 0

        total_loss = mse_loss + aux_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if self.config.training.get("gradient_clip_norm"):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training["gradient_clip_norm"]
            )

        # Optimizer step
        if (self.step + 1) % self.config.training.get(
            "gradient_accumulation_steps", 1
        ) == 0:
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

        return {
            "total_loss": total_loss.item(),
            "mse_loss": mse_loss.item()
            if isinstance(mse_loss, torch.Tensor)
            else mse_loss,
            "aux_loss": aux_loss.item()
            if isinstance(aux_loss, torch.Tensor)
            else aux_loss,
        }

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        """Perform validation."""
        self.model.eval()

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training["batch_size"],
            num_workers=0,  # Single worker for validation
        )

        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0
        example_predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for i, batch in enumerate(val_loader):
            if i >= 50:  # Limit validation batches
                break

            input_seq, target_seq = batch
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            # Same approach as training
            full_seq = torch.cat([input_seq, target_seq], dim=1)
            logits, _ = self.model(full_seq, targets=None)

            # Handle different pooling modes
            if len(logits.shape) == 2:
                # Pooled output [B, target_dim] - use full sequence target
                pred_logits = logits
                # For pooled mode, use the mean of the target sequence
                target_mean = target_seq.mean(dim=1)  # [B, target_dim]
                mse_loss = F.mse_loss(pred_logits, target_mean)
            else:
                # Sequence output [B, S, target_dim] - extract predictions for target portion
                input_len = input_seq.shape[1]
                pred_logits = logits[:, input_len:, :]
                mse_loss = F.mse_loss(pred_logits, target_seq)

            # Estimate aux loss
            if len(logits.shape) == 2:
                # For pooled mode, create compatible target for aux loss
                target_mean_full = full_seq.mean(dim=1)  # [B, target_dim]
                _, full_loss = self.model(full_seq, targets=target_mean_full)
                if full_loss is not None and isinstance(full_loss, torch.Tensor):
                    full_mse = F.mse_loss(logits, target_mean_full)
                    aux_loss = full_loss - full_mse
                    aux_loss = torch.clamp(aux_loss, min=0)
                else:
                    aux_loss = 0
            else:
                # For sequence mode, use original approach
                _, full_loss = self.model(full_seq, targets=full_seq)
                if full_loss is not None and isinstance(full_loss, torch.Tensor):
                    full_mse = F.mse_loss(logits, full_seq)
                    aux_loss = full_loss - full_mse
                    aux_loss = torch.clamp(aux_loss, min=0)
                else:
                    aux_loss = 0

            total_loss += (mse_loss + aux_loss).item()
            total_mse += float(mse_loss.item())
            num_batches += 1

            # Collect prediction examples
            if len(example_predictions) < 4:
                # Take first sample from batch
                if len(logits.shape) == 2:
                    # For pooled mode, create a dummy prediction sequence for visualization
                    pred_len = target_seq.shape[1]
                    # Expand the pooled prediction to match sequence length
                    pred_sample = pred_logits[0].unsqueeze(0).expand(pred_len, -1).cpu()
                else:
                    # Use the prediction portion for sequence mode
                    pred_sample = pred_logits[0].cpu()

                example_predictions.append(
                    (input_seq[0].cpu(), target_seq[0].cpu(), pred_sample)
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_mse = total_mse / num_batches if num_batches > 0 else 0

        return {
            "val_loss": avg_loss,
            "val_mse": avg_mse,
            "example_predictions": example_predictions,
        }

    def train(self) -> dict[str, Any]:
        """Main training loop."""
        batch_size = self.config.training["batch_size"]
        max_steps = self.config.training["max_steps"]
        total_samples = max_steps * batch_size

        print(f"\n{'=' * 80}")
        print("TRAINING CONFIGURATION:")
        print(f"{'=' * 80}")
        print(f"• Total training steps: {max_steps:,}")
        print(f"• Batch size: {batch_size} samples per step")
        print(f"• Total samples to process: {total_samples:,} samples")
        print(f"• Progress bar shows: step_number / {max_steps}")
        print(f"• Each step processes 1 batch of {batch_size} samples")
        print(f"• Throughput = steps/second × {batch_size} = samples/second")
        print(f"{'=' * 80}")
        print("Starting training...")

        # Initialize progress tracking
        self.visualizer.start_progress(self.config.training["max_steps"], batch_size)

        # Create data loader (no multiprocessing to avoid pickle issues with inner classes)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training["batch_size"],
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        start_time = time.time()
        last_throughput_time = start_time
        throughput_step_count = 0

        # Training loop
        data_iter = iter(train_loader)

        for step in range(self.config.training["max_steps"]):
            self.step = step

            try:
                batch = next(data_iter)
            except StopIteration:
                # Reset data iterator
                data_iter = iter(train_loader)
                batch = next(data_iter)
                self.epoch += 1

            # Training step
            step_metrics = self.train_step(batch)

            # Compute comprehensive throughput metrics (every 10 steps)
            throughput_step_count += 1
            current_time = time.time()

            if throughput_step_count >= 10:  # Calculate metrics every 10 steps
                elapsed_time = current_time - last_throughput_time
                if elapsed_time > 0:
                    batch_size = self.config.training["batch_size"]
                    sequence_length = self.config.dataset["sequence_length"]

                    # Calculate all throughput metrics
                    current_steps_per_sec = throughput_step_count / elapsed_time
                    current_batches_per_sec = current_steps_per_sec  # 1 step = 1 batch
                    current_samples_per_sec = current_steps_per_sec * batch_size
                    current_tokens_per_sec = current_samples_per_sec * sequence_length

                    # Update EMAs
                    def update_ema(current_ema, new_value):
                        if current_ema is None:
                            return new_value
                        return (
                            self.ema_alpha * new_value
                            + (1 - self.ema_alpha) * current_ema
                        )

                    self.steps_per_second_ema = update_ema(
                        self.steps_per_second_ema, current_steps_per_sec
                    )
                    self.batches_per_second_ema = update_ema(
                        self.batches_per_second_ema, current_batches_per_sec
                    )
                    self.samples_per_second_ema = update_ema(
                        self.samples_per_second_ema, current_samples_per_sec
                    )
                    self.tokens_per_second_ema = update_ema(
                        self.tokens_per_second_ema, current_tokens_per_sec
                    )

                # Reset counters
                last_throughput_time = current_time
                throughput_step_count = 0

            # Calculate loss change rate
            current_loss = step_metrics["total_loss"]
            loss_rate = None
            if self.previous_loss is not None and self.previous_loss_time is not None:
                time_diff = current_time - self.previous_loss_time
                if time_diff > 0:
                    loss_diff = (
                        self.previous_loss - current_loss
                    )  # positive = loss decreasing
                    current_loss_rate = loss_diff / time_diff
                    self.loss_per_second_ema = (
                        update_ema(self.loss_per_second_ema, current_loss_rate)
                        if "update_ema" in locals()
                        else current_loss_rate
                    )

            # Update previous loss tracking (every 10 steps to avoid noise)
            if step % 10 == 0:
                self.previous_loss = current_loss
                self.previous_loss_time = current_time

            # Collect system metrics
            system_metrics = self._collect_system_metrics()

            # Create comprehensive training metrics
            metrics = TrainingMetrics(
                step=step,
                epoch=self.epoch,
                train_loss=step_metrics["total_loss"],
                learning_rate=self.optimizer.param_groups[0]["lr"],
                aux_loss=step_metrics.get("aux_loss", 0),
                expert_utilization=self._compute_expert_utilization(),
                # Throughput metrics
                samples_per_second=self.samples_per_second_ema,
                batches_per_second=self.batches_per_second_ema,
                steps_per_second=self.steps_per_second_ema,
                tokens_per_second=self.tokens_per_second_ema,
                # System metrics
                gpu_memory_used_mb=system_metrics.get("gpu_memory_used_mb"),
                gpu_memory_total_mb=system_metrics.get("gpu_memory_total_mb"),
                gpu_utilization_percent=system_metrics.get("gpu_utilization_percent"),
                # Training efficiency metrics
                loss_per_second=self.loss_per_second_ema,
                gradient_norm=step_metrics.get("gradient_norm"),
            )

            # Validation
            val_metrics = None
            example_predictions = None
            if step % self.config.training["eval_interval"] == 0 and step > 0:
                val_results = self.validate()
                val_loss = val_results["val_loss"]
                example_predictions = val_results["example_predictions"]

                metrics.val_loss = val_loss
                val_metrics = val_results

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best_model.pt")

            # Update visualization
            self.visualizer.update(metrics)

            self.metrics_history.append(metrics)

            # Logging - more frequent console logging
            if step % 25 == 0 or val_metrics:
                self._log_progress(step, metrics, val_metrics)

            # Save checkpoint
            if step % self.config.training["save_interval"] == 0 and step > 0:
                self._save_checkpoint(f"checkpoint_step_{step}.pt")

        # Training complete
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")

        # Final validation
        final_val = self.validate()

        # Save final model
        self._save_checkpoint("final_model.pt")

        # Generate final report
        self.visualizer.save_final_report(self.config.experiment_name, total_time)

        # Save metrics CSV
        csv_path = self.config.output_dir / f"metrics_{self.config.experiment_name}.csv"
        save_metrics_csv(self.metrics_history, csv_path)

        # Cleanup
        self.visualizer.close()

        return {
            "final_train_loss": self.metrics_history[-1].train_loss,
            "final_val_loss": final_val["val_loss"],
            "best_val_loss": self.best_val_loss,
            "total_time": total_time,
            "total_steps": self.config.training["max_steps"],
        }

    def _log_progress(
        self, step: int, metrics: TrainingMetrics, val_metrics: dict | None = None
    ):
        """Log comprehensive training progress."""
        msg = f"Step {step:6d} | Loss: {metrics.train_loss:.4f}"

        if metrics.learning_rate:
            msg += f" | LR: {metrics.learning_rate:.6f}"

        if metrics.aux_loss:
            msg += f" | Aux: {metrics.aux_loss:.4f}"

        if val_metrics:
            msg += f" | Val: {val_metrics['val_loss']:.4f}"

        # Comprehensive throughput metrics
        if metrics.steps_per_second:
            msg += f" | {metrics.steps_per_second:.1f} steps/s | {metrics.samples_per_second:.0f} samples/s | {metrics.tokens_per_second:.0f} tokens/s"

        # System metrics
        if metrics.gpu_memory_used_mb:
            msg += f" | GPU: {metrics.gpu_memory_used_mb:.0f}MB"

        # Loss improvement rate
        if metrics.loss_per_second:
            msg += f" | ΔLoss: {metrics.loss_per_second:.6f}/s"

        print(msg)

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config)
            if hasattr(self.config, "__dataclass_fields__")
            else vars(self.config),
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        save_path = self.config.output_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")


def load_config(config_path: Path) -> SignalTrainingConfig:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Convert paths
    if "output_dir" in config_dict:
        config_dict["output_dir"] = Path(config_dict["output_dir"])

    return SignalTrainingConfig(**config_dict)


def train_from_config(
    config_path: Path, experiment_name: str | None = None
) -> dict[str, Any]:
    """Train model from configuration file."""
    config = load_config(config_path)

    if experiment_name:
        config.experiment_name = experiment_name
        # Update output directory with experiment name
        config.output_dir = config.output_dir / experiment_name

    trainer = SignalTrainer(config)
    results = trainer.train()

    return results


def main():
    """Main CLI entrypoint for geometric signals training."""
    # sys already imported at top
    from .model_discovery import find_geometric_signals_models

    if len(sys.argv) < 2:
        print("🌊 Geometric Signals Training")
        print("Usage: signals-train <config_path> [experiment_name]")
        print("")

        # Show available configs
        config_dir = Path(__file__).parent / "configs"
        if config_dir.exists():
            configs = list(config_dir.glob("*.yaml"))
            if configs:
                print("📁 Available configurations:")
                for config in configs:
                    print(f"  {config}")
                print("")
                print("Examples:")
                print(f"  signals-train {configs[0]} my_experiment")
                print(f"  signals-train {configs[0]}")
        else:
            print("❌ No configuration files found")

        # Show existing models
        existing_models = find_geometric_signals_models()
        if existing_models:
            print(f"📊 Existing trained models: {len(existing_models)}")

        sys.exit(1)

    config_path = Path(sys.argv[1])
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else None

    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    print("🚀 Starting geometric signals training")
    print(f"📁 Config: {config_path}")
    if experiment_name:
        print(f"🏷️  Experiment: {experiment_name}")

    try:
        results = train_from_config(config_path, experiment_name)
        print("✅ Training completed successfully!")
        print(f"📊 Results: {results}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
