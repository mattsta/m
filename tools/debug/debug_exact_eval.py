#!/usr/bin/env python3
"""Debug script that exactly replicates the evaluation visualization call"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(".")

from examples.geometric_signals.visualization import create_sequence_prediction_plot


def debug_exact_eval():
    """Create the exact same call as the evaluation"""

    # Create fake data that matches what evaluation creates
    input_len = 32
    target_len = 8

    # Create fake examples list like evaluation does
    np.random.seed(42)
    examples = []

    for i in range(4):
        # Create tensors exactly like the evaluation
        input_seq = torch.randn(input_len, 1)  # [32, 1]
        target_seq = torch.randn(target_len, 1)  # [8, 1]
        pred_seq = target_seq + torch.randn(target_len, 1) * 0.1  # Close to target

        examples.append((input_seq, target_seq, pred_seq))

    print(f"Created {len(examples)} examples")
    print(f"Input shape: {examples[0][0].shape}")
    print(f"Target shape: {examples[0][1].shape}")
    print(f"Prediction shape: {examples[0][2].shape}")

    # Call the EXACT function that evaluation calls
    create_sequence_prediction_plot(
        examples=examples,
        title="Debug Test - Geometric Dataset",
        save_path=Path("debug_exact_eval.png"),
        dataset_name="geometric",
    )

    print("Saved debug_exact_eval.png - this should show the EXACT same issues")


if __name__ == "__main__":
    debug_exact_eval()
