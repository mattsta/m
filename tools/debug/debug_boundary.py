#!/usr/bin/env python3
"""Debug script to find the boundary positioning bug"""

import matplotlib.pyplot as plt
import numpy as np


def test_boundary_positioning():
    """Test the exact boundary positioning logic"""

    # Simulate exact evaluation data
    input_len = 32  # From config
    target_len = 8
    total_len = input_len + target_len

    # Create dummy data
    np.random.seed(42)
    input_np = np.random.randn(input_len)
    target_np = np.random.randn(target_len)
    pred_np = target_np + np.random.randn(target_len) * 0.1

    print(f"input_len = {input_len}")
    print(f"target_len = {target_len}")
    print(f"total_len = {total_len}")

    # Replicate the EXACT visualization code
    full_time = np.arange(total_len)
    input_time = full_time[:input_len]
    full_target = np.concatenate([input_np, target_np])
    full_pred = np.concatenate([input_np, pred_np])

    print(f"full_time: {full_time}")
    print(f"input_time: {input_time}")
    print(f"Last input timestep: {input_time[-1]}")
    print(f"First prediction timestep should be: {input_len}")

    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot exactly like the code
    ax.plot(
        full_time,
        full_target,
        "g-",
        linewidth=1.5,
        alpha=0.6,
        label="Ground Truth Full Signal",
    )
    ax.plot(
        full_time,
        full_pred,
        "r--",
        linewidth=1.5,
        alpha=0.8,
        label="Model Prediction Full Signal",
    )
    ax.plot(
        input_time,
        input_np,
        "b-",
        linewidth=3,
        label="Input Context (Model Sees)",
        alpha=0.9,
    )

    # THE BOUNDARY LINE - let's test different positions
    print(f"Setting boundary at x = {input_len}")
    ax.axvline(
        input_len,
        color="orange",
        linestyle=":",
        linewidth=2,
        label="Prediction Boundary",
        alpha=0.7,
    )

    # Add grid and markers to see exactly where things are
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 41)

    # Mark key positions
    ax.axvline(31, color="red", linestyle="-", alpha=0.3, label="x=31")
    ax.axvline(31.5, color="purple", linestyle="-", alpha=0.3, label="x=31.5")
    ax.axvline(32, color="black", linestyle="-", alpha=0.3, label="x=32")

    ax.legend()
    ax.set_title(f"Boundary Debug: input_len={input_len}, should be at x={input_len}")
    ax.set_xlabel("Time Steps")

    plt.tight_layout()
    plt.savefig("debug_boundary_detailed.png", dpi=150, bbox_inches="tight")
    print("Saved debug_boundary_detailed.png")

    # Also print the exact axvline call that should be made
    print("\nThe axvline call should be:")
    print(
        f"ax.axvline({input_len}, color='orange', linestyle=':', linewidth=2, label='Prediction Boundary', alpha=0.7)"
    )


if __name__ == "__main__":
    test_boundary_positioning()
