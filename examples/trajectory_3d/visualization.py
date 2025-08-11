"""
3D Trajectory Visualization System

Provides comprehensive visualization for 3D trajectory learning:
- Interactive 3D trajectory plots
- Real-time training visualization
- Prediction vs ground truth comparison
- Expert utilization in 3D space
- Animation support for trajectory evolution
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # type: ignore

from .datasets import generate_sample_trajectories


class Trajectory3DVisualizer:
    """
    Comprehensive visualization system for 3D trajectory learning.

    Supports static plots, animations, and real-time training monitoring.
    """

    def __init__(self, output_dir: Path | str = "outputs/trajectory_3d"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up matplotlib for better 3D plots
        plt.style.use("default")
        self.colors = {
            "helical": "#FF6B6B",  # Red
            "orbital": "#4ECDC4",  # Teal
            "lissajous": "#45B7D1",  # Blue
            "lorenz": "#96CEB4",  # Green
            "robotic": "#FFEAA7",  # Yellow
            "prediction": "#DDA0DD",  # Plum
            "ground_truth": "#2F3542",  # Dark gray
        }

    def plot_trajectory_samples(self, save_path: str | None = None) -> None:
        """Plot sample trajectories of each type for visualization."""
        print("🎨 Generating sample trajectory visualizations...")

        # Generate sample trajectories
        trajectories = generate_sample_trajectories()

        # Create subplots for different trajectory types
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(
            "3D Trajectory Types for MoE Learning", fontsize=16, fontweight="bold"
        )

        for i, (traj_type, trajectory) in enumerate(trajectories.items()):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")

            # Plot trajectory
            color = self.colors[traj_type]
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=color,
                linewidth=2,
                alpha=0.8,
            )

            # Mark start and end points
            ax.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                trajectory[0, 2],
                color="green",
                s=100,
                marker="o",
                label="Start",
            )
            ax.scatter(
                trajectory[-1, 0],
                trajectory[-1, 1],
                trajectory[-1, 2],
                color="red",
                s=100,
                marker="s",
                label="End",
            )

            # Customize plot
            ax.set_title(f"{traj_type.capitalize()} Trajectory", fontweight="bold")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_zlabel("Z Position")
            ax.legend()

            # Set equal aspect ratio for better visualization
            self._set_equal_aspect_3d(ax, trajectory)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"   📊 Saved trajectory samples to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_prediction_comparison(
        self,
        input_sequence: torch.Tensor,  # [seq_len, 3]
        target_sequence: torch.Tensor,  # [pred_len, 3]
        predicted_sequence: torch.Tensor,  # [pred_len, 3]
        trajectory_type: str,
        save_path: str | None = None,
    ) -> None:
        """Plot comparison between predicted and ground truth trajectories."""

        # Convert to numpy
        input_seq = input_sequence.detach().cpu().numpy()
        target_seq = target_sequence.detach().cpu().numpy()
        pred_seq = predicted_sequence.detach().cpu().numpy()

        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(
            f"3D Trajectory Prediction: {trajectory_type.capitalize()}",
            fontsize=14,
            fontweight="bold",
        )

        # Plot 1: Full trajectory with prediction
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")

        # Input sequence (context)
        ax1.plot(
            input_seq[:, 0],
            input_seq[:, 1],
            input_seq[:, 2],
            color="blue",
            linewidth=2,
            label="Input Context",
            alpha=0.8,
        )

        # Connect continuation from last input point
        if len(input_seq) > 0:
            last_input_point = input_seq[-1:, :]  # [1, 3]

            # Ground truth continuation (connected)
            ground_truth_continuous = np.vstack([last_input_point, target_seq])
            ax1.plot(
                ground_truth_continuous[:, 0],
                ground_truth_continuous[:, 1],
                ground_truth_continuous[:, 2],
                color="green",
                linewidth=2,
                label="Ground Truth",
                alpha=0.8,
            )

            # Predicted continuation (connected)
            prediction_continuous = np.vstack([last_input_point, pred_seq])
            ax1.plot(
                prediction_continuous[:, 0],
                prediction_continuous[:, 1],
                prediction_continuous[:, 2],
                color="red",
                linewidth=2,
                label="Prediction",
                alpha=0.8,
                linestyle="--",
            )
        else:
            # Fallback if no input sequence
            ax1.plot(
                target_seq[:, 0],
                target_seq[:, 1],
                target_seq[:, 2],
                color="green",
                linewidth=2,
                label="Ground Truth",
                alpha=0.8,
            )
            ax1.plot(
                pred_seq[:, 0],
                pred_seq[:, 1],
                pred_seq[:, 2],
                color="red",
                linewidth=2,
                label="Prediction",
                alpha=0.8,
                linestyle="--",
            )

        # Mark transition point
        if len(input_seq) > 0:
            ax1.scatter(
                input_seq[-1, 0],
                input_seq[-1, 1],
                input_seq[-1, 2],
                color="orange",
                s=100,
                marker="o",
                label="Transition",
            )

        ax1.set_title("3D Trajectory Prediction")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.legend()

        # Combine all points for equal aspect ratio
        all_points = np.vstack([input_seq, target_seq, pred_seq])
        self._set_equal_aspect_3d(ax1, all_points)

        # Plot 2: X-Y projection
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(
            input_seq[:, 0],
            input_seq[:, 1],
            "b-",
            linewidth=2,
            label="Input Context",
            alpha=0.8,
        )

        # Connect continuation from last input point for X-Y projection
        if len(input_seq) > 0:
            # Ground truth continuation (connected)
            ax2.plot(
                ground_truth_continuous[:, 0],
                ground_truth_continuous[:, 1],
                "g-",
                linewidth=2,
                label="Ground Truth",
                alpha=0.8,
            )

            # Predicted continuation (connected)
            ax2.plot(
                prediction_continuous[:, 0],
                prediction_continuous[:, 1],
                "r--",
                linewidth=2,
                label="Prediction",
                alpha=0.8,
            )

            # Mark transition point
            ax2.scatter(
                input_seq[-1, 0], input_seq[-1, 1], color="orange", s=100, marker="o"
            )
        else:
            # Fallback if no input sequence
            ax2.plot(
                target_seq[:, 0],
                target_seq[:, 1],
                "g-",
                linewidth=2,
                label="Ground Truth",
                alpha=0.8,
            )
            ax2.plot(
                pred_seq[:, 0],
                pred_seq[:, 1],
                "r--",
                linewidth=2,
                label="Prediction",
                alpha=0.8,
            )

        ax2.set_title("X-Y Projection")
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal")

        # Plot 3: Error analysis
        ax3 = fig.add_subplot(1, 3, 3)

        if len(target_seq) == len(pred_seq):
            # Compute position errors
            position_errors = np.linalg.norm(target_seq - pred_seq, axis=1)
            time_steps = np.arange(len(position_errors))

            ax3.plot(time_steps, position_errors, "r-", linewidth=2, marker="o")
            ax3.set_title("Position Error Over Time")
            ax3.set_xlabel("Prediction Step")
            ax3.set_ylabel("Euclidean Distance Error")
            ax3.grid(True, alpha=0.3)

            # Add error statistics
            mean_error = np.mean(position_errors)
            max_error = np.max(position_errors)
            ax3.axhline(
                y=mean_error,
                color="orange",
                linestyle="--",
                label=f"Mean: {mean_error:.3f}",
            )
            ax3.text(
                0.05,
                0.95,
                f"Max Error: {max_error:.3f}",
                transform=ax3.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            ax3.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def plot_training_progress(
        self, metrics: dict[str, list[float]], save_path: str | None = None
    ) -> None:
        """Plot real-time training progress with trajectory-specific metrics."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("3D Trajectory Learning Progress", fontsize=16, fontweight="bold")

        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        if "train_loss" in metrics:
            train_steps = range(len(metrics["train_loss"]))
            ax1.plot(
                train_steps, metrics["train_loss"], label="Training Loss", color="blue"
            )
        if "val_loss" in metrics:
            val_steps = range(len(metrics["val_loss"]))
            ax1.plot(
                val_steps, metrics["val_loss"], label="Validation Loss", color="red"
            )
        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("MSE Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # Plot 2: Position error metrics
        ax2 = axes[0, 1]
        if "position_error" in metrics:
            pos_error_steps = range(len(metrics["position_error"]))
            ax2.plot(
                pos_error_steps,
                metrics["position_error"],
                label="Position Error",
                color="green",
            )
        if "velocity_error" in metrics:
            vel_error_steps = range(len(metrics["velocity_error"]))
            ax2.plot(
                vel_error_steps,
                metrics["velocity_error"],
                label="Velocity Error",
                color="orange",
            )
        ax2.set_title("3D Error Metrics")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Error Magnitude")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Expert utilization
        ax3 = axes[1, 0]
        if "expert_entropy" in metrics:
            entropy_steps = range(len(metrics["expert_entropy"]))
            ax3.plot(
                entropy_steps,
                metrics["expert_entropy"],
                label="Expert Entropy",
                color="purple",
            )
        if "load_balance" in metrics:
            balance_steps = range(len(metrics["load_balance"]))
            ax3.plot(
                balance_steps,
                metrics["load_balance"],
                label="Load Balance",
                color="brown",
            )
        ax3.set_title("Expert Utilization")
        ax3.set_xlabel("Training Step")
        ax3.set_ylabel("Utilization Metric")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Throughput and performance
        ax4 = axes[1, 1]
        if "samples_per_sec" in metrics:
            throughput_steps = range(len(metrics["samples_per_sec"]))
            ax4.plot(
                throughput_steps,
                metrics["samples_per_sec"],
                label="Samples/sec",
                color="cyan",
            )
        if "tokens_per_sec" in metrics:
            ax4_twin = ax4.twinx()
            tokens_steps = range(len(metrics["tokens_per_sec"]))
            ax4_twin.plot(
                tokens_steps,
                metrics["tokens_per_sec"],
                label="Tokens/sec",
                color="magenta",
            )
            ax4_twin.set_ylabel("Tokens per Second")
            ax4_twin.legend(loc="upper right")
        ax4.set_title("Training Throughput")
        ax4.set_xlabel("Training Step")
        ax4.set_ylabel("Samples per Second")
        ax4.legend(loc="upper left")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def create_trajectory_animation(
        self,
        trajectory: np.ndarray,
        title: str = "3D Trajectory Animation",
        save_path: str | None = None,
        fps: int = 30,
        trail_length: int = 50,
    ) -> None:
        """Create animated visualization of 3D trajectory evolution."""

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Set up the plot
        ax.set_title(title)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")

        # Set axis limits
        self._set_equal_aspect_3d(ax, trajectory)

        # Initialize empty line objects
        (line,) = ax.plot([], [], [], "b-", linewidth=2, alpha=0.7)
        (point,) = ax.plot([], [], [], "ro", markersize=8)
        (trail,) = ax.plot([], [], [], "b-", linewidth=1, alpha=0.3)

        def animate(frame):
            if frame >= len(trajectory):
                return line, point, trail

            # Current position
            current_pos = trajectory[frame]
            point.set_data([current_pos[0]], [current_pos[1]])
            point.set_3d_properties([current_pos[2]])

            # Trail (last N points)
            start_idx = max(0, frame - trail_length)
            trail_data = trajectory[start_idx : frame + 1]
            if len(trail_data) > 1:
                trail.set_data(trail_data[:, 0], trail_data[:, 1])
                trail.set_3d_properties(trail_data[:, 2])

            # Full trajectory up to current point (faded)
            if frame > 0:
                full_data = trajectory[: frame + 1]
                line.set_data(full_data[:, 0], full_data[:, 1])
                line.set_3d_properties(full_data[:, 2])

            return line, point, trail

        # Create animation
        total_frames = len(trajectory)
        interval = 1000 // fps  # milliseconds per frame

        anim = FuncAnimation(
            fig,
            animate,
            frames=total_frames,
            interval=interval,
            blit=False,
            repeat=True,
        )

        if save_path:
            # Save as GIF
            anim.save(save_path, writer="pillow", fps=fps)
            print(f"   🎬 Saved trajectory animation to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_expert_specialization_3d(
        self,
        expert_activations: dict[
            int, list[np.ndarray]
        ],  # expert_id -> list of 3D positions
        trajectory_types: dict[int, list[str]],  # expert_id -> list of trajectory types
        save_path: str | None = None,
    ) -> None:
        """Visualize which experts specialize on which 3D regions/trajectory types."""

        num_experts = len(expert_activations)
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(
            "Expert Specialization in 3D Space", fontsize=16, fontweight="bold"
        )

        # Create subplots for each expert
        subplot_rows = int(np.ceil(np.sqrt(num_experts)))
        subplot_cols = int(np.ceil(num_experts / subplot_rows))

        for expert_id, positions_list in expert_activations.items():
            if not positions_list:
                continue

            ax = fig.add_subplot(
                subplot_rows, subplot_cols, expert_id + 1, projection="3d"
            )

            # Combine all positions for this expert
            all_positions = np.vstack(positions_list)
            types_list = trajectory_types.get(expert_id, [])

            # Color points by trajectory type
            for traj_type in set(types_list):
                type_positions = []
                for i, pos_batch in enumerate(positions_list):
                    if i < len(types_list) and types_list[i] == traj_type:
                        type_positions.append(pos_batch)

                if type_positions:
                    combined_pos = np.vstack(type_positions)
                    color = self.colors.get(traj_type, "gray")
                    ax.scatter(
                        combined_pos[:, 0],
                        combined_pos[:, 1],
                        combined_pos[:, 2],
                        c=color,
                        alpha=0.6,
                        s=20,
                        label=traj_type,
                    )

            ax.set_title(f"Expert {expert_id}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.legend()

            # Set equal aspect
            self._set_equal_aspect_3d(ax, all_positions)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def create_sequence_prediction_plot(
        self,
        examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        title: str,
        save_path: str | None = None,
        dataset_name: str = "trajectory",
    ) -> None:
        """Create enhanced prediction visualization showing model behavior for trajectory sequences."""

        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        n_examples = min(len(examples), 4)  # Show up to 4 examples

        for i, (input_seq, target_seq, pred_seq) in enumerate(examples[:n_examples]):
            # Convert to numpy
            input_np = input_seq.detach().cpu().numpy()
            target_np = target_seq.detach().cpu().numpy()
            pred_np = pred_seq.detach().cpu().numpy()

            # Create 3D subplot for this example
            ax = fig.add_subplot(2, n_examples, i + 1, projection="3d")

            # Plot input trajectory (context)
            ax.plot(
                input_np[:, 0],
                input_np[:, 1],
                input_np[:, 2],
                "b-",
                linewidth=2,
                label="Input Context",
                alpha=0.8,
            )

            # Connect continuations from last input point
            if len(input_np) > 0:
                last_input_point = input_np[-1:, :]  # [1, 3]

                # Ground truth continuation (connected)
                ground_truth_continuous = np.vstack([last_input_point, target_np])
                ax.plot(
                    ground_truth_continuous[:, 0],
                    ground_truth_continuous[:, 1],
                    ground_truth_continuous[:, 2],
                    "g-",
                    linewidth=2,
                    label="Ground Truth",
                    alpha=0.8,
                )

                # Predicted continuation (connected)
                prediction_continuous = np.vstack([last_input_point, pred_np])
                ax.plot(
                    prediction_continuous[:, 0],
                    prediction_continuous[:, 1],
                    prediction_continuous[:, 2],
                    "r--",
                    linewidth=2,
                    label="Prediction",
                    alpha=0.8,
                )
            else:
                # Fallback if no input sequence
                ax.plot(
                    target_np[:, 0],
                    target_np[:, 1],
                    target_np[:, 2],
                    "g-",
                    linewidth=2,
                    label="Ground Truth",
                    alpha=0.8,
                )
                ax.plot(
                    pred_np[:, 0],
                    pred_np[:, 1],
                    pred_np[:, 2],
                    "r--",
                    linewidth=2,
                    label="Prediction",
                    alpha=0.8,
                )

            # Mark start and end points
            if len(input_np) > 0:
                ax.scatter(
                    input_np[0, 0],
                    input_np[0, 1],
                    input_np[0, 2],
                    color="blue",
                    s=100,
                    marker="o",
                    alpha=0.8,
                )
                ax.scatter(
                    input_np[-1, 0],
                    input_np[-1, 1],
                    input_np[-1, 2],
                    color="orange",
                    s=100,
                    marker="s",
                    alpha=0.8,
                )

            if len(target_np) > 0:
                ax.scatter(
                    target_np[-1, 0],
                    target_np[-1, 1],
                    target_np[-1, 2],
                    color="green",
                    s=100,
                    marker="^",
                    alpha=0.8,
                )

            if len(pred_np) > 0:
                ax.scatter(
                    pred_np[-1, 0],
                    pred_np[-1, 1],
                    pred_np[-1, 2],
                    color="red",
                    s=100,
                    marker="v",
                    alpha=0.8,
                )

            ax.set_title(f"Example {i + 1} - 3D Trajectory")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_zlabel("Z Position")
            ax.legend()

            # Set equal aspect
            all_points = np.vstack([input_np, target_np, pred_np])
            self._set_equal_aspect_3d(ax, all_points)

            # Create 2D projection subplot (X-Y plane)
            ax2d = fig.add_subplot(2, n_examples, i + 1 + n_examples)

            # Plot 2D projections
            ax2d.plot(
                input_np[:, 0],
                input_np[:, 1],
                "b-",
                linewidth=2,
                label="Input Context",
                alpha=0.8,
            )

            # Connect 2D continuations from last input point
            if len(input_np) > 0:
                # Use the same continuous arrays from 3D plot
                ax2d.plot(
                    ground_truth_continuous[:, 0],
                    ground_truth_continuous[:, 1],
                    "g-",
                    linewidth=2,
                    label="Ground Truth",
                    alpha=0.8,
                )
                ax2d.plot(
                    prediction_continuous[:, 0],
                    prediction_continuous[:, 1],
                    "r--",
                    linewidth=2,
                    label="Prediction",
                    alpha=0.8,
                )
            else:
                # Fallback if no input sequence
                ax2d.plot(
                    target_np[:, 0],
                    target_np[:, 1],
                    "g-",
                    linewidth=2,
                    label="Ground Truth",
                    alpha=0.8,
                )
                ax2d.plot(
                    pred_np[:, 0],
                    pred_np[:, 1],
                    "r--",
                    linewidth=2,
                    label="Prediction",
                    alpha=0.8,
                )

            # Mark points
            if len(input_np) > 0:
                ax2d.scatter(
                    input_np[0, 0],
                    input_np[0, 1],
                    color="blue",
                    s=100,
                    marker="o",
                    alpha=0.8,
                )
                ax2d.scatter(
                    input_np[-1, 0],
                    input_np[-1, 1],
                    color="orange",
                    s=100,
                    marker="s",
                    alpha=0.8,
                )

            if len(target_np) > 0:
                ax2d.scatter(
                    target_np[-1, 0],
                    target_np[-1, 1],
                    color="green",
                    s=100,
                    marker="^",
                    alpha=0.8,
                )

            if len(pred_np) > 0:
                ax2d.scatter(
                    pred_np[-1, 0],
                    pred_np[-1, 1],
                    color="red",
                    s=100,
                    marker="v",
                    alpha=0.8,
                )

            ax2d.set_title(f"Example {i + 1} - X-Y Projection")
            ax2d.set_xlabel("X Position")
            ax2d.set_ylabel("Y Position")
            ax2d.legend()
            ax2d.grid(True, alpha=0.3)
            ax2d.set_aspect("equal")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def _set_equal_aspect_3d(self, ax: Axes3D, data: np.ndarray) -> None:
        """Set equal aspect ratio for 3D plot based on data range."""
        if len(data) == 0:
            return

        # Get data ranges
        x_range = data[:, 0].max() - data[:, 0].min()
        y_range = data[:, 1].max() - data[:, 1].min()
        z_range = data[:, 2].max() - data[:, 2].min()

        # Find maximum range
        max_range = max(x_range, y_range, z_range)

        # Get centers
        x_center = (data[:, 0].max() + data[:, 0].min()) / 2
        y_center = (data[:, 1].max() + data[:, 1].min()) / 2
        z_center = (data[:, 2].max() + data[:, 2].min()) / 2

        # Set limits
        ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
        ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)


# Real-time training visualization has been moved to m.training_viz
# Use: from m.training_viz import RealTimeTrainingVisualizer, create_moe_visualizer


def demo_main():
    """Main CLI entry point for 3D trajectory visualization demo."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="3D Trajectory Visualization Demo")
    parser.add_argument(
        "--output-dir",
        default="outputs/trajectory_3d/demo",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--animate", action="store_true", help="Create trajectory animations (slower)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🎨 3D Trajectory Visualization Demo")
    print(f"📁 Output directory: {output_dir}")

    visualizer = Trajectory3DVisualizer(output_dir)

    # Generate sample trajectory plots
    print("📊 Creating trajectory type samples...")
    visualizer.plot_trajectory_samples(str(output_dir / "trajectory_samples.png"))

    if args.animate:
        print("🎬 Creating trajectory animations...")
        trajectories = generate_sample_trajectories()

        for traj_type, trajectory in trajectories.items():
            print(f"   Creating {traj_type} animation...")
            animation_path = output_dir / f"{traj_type}_animation.gif"

            # Use subset for faster animation
            subset = trajectory[::2][:100] if len(trajectory) > 100 else trajectory

            visualizer.create_trajectory_animation(
                subset,
                f"{traj_type.capitalize()} Trajectory",
                str(animation_path),
                fps=20,
            )

    print(f"✅ Demo completed! Check outputs in: {output_dir}")


if __name__ == "__main__":
    # Test visualization system
    print("🎨 Testing 3D Trajectory Visualization")

    visualizer = Trajectory3DVisualizer("test_outputs")

    # Test 1: Plot trajectory samples
    visualizer.plot_trajectory_samples("test_outputs/trajectory_samples.png")

    # Test 2: Test animation (just generate, don't save)
    trajectories = generate_sample_trajectories()
    print("   🎬 Testing trajectory animation...")
    # Note: Animation test commented out to avoid blocking
    # visualizer.create_trajectory_animation(
    #     trajectories['lorenz'][:200],  # First 200 points
    #     "Lorenz Attractor Animation",
    #     None  # Don't save, just test
    # )

    print("✅ 3D visualization system working correctly!")
