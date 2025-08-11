#!/usr/bin/env python3
"""Debug checkpoint best tracking."""

import tempfile
from pathlib import Path

import torch

# Set globals
import m.moe
from m.moe import CheckpointManager, ModelConfig, MoESequenceRegressor

m.moe.keep_last_global = 5
m.moe.keep_best_global = 2

with tempfile.TemporaryDirectory() as temp_dir:
    manager = CheckpointManager(
        out_dir=str(temp_dir),
        run_name="test",
        keep_last=5,
        keep_best=2,
    )

    # Create dummy model and optimizer
    model = MoESequenceRegressor(ModelConfig())
    optimizer = torch.optim.AdamW(model.parameters())

    # Save checkpoints with metrics
    for i, metric in enumerate([0.5, 0.3, 0.7, 0.2, 0.6]):
        print(f"Saving step{i} with metric {metric}")
        manager.save(
            tag=f"step{i}",
            model=model,
            optimizer=optimizer,
            scheduler_state={},
            scaler=None,
            trainer_state={"global_step": i},
            config_snapshot={},
            is_best=metric,
        )

    # Check what's in the best dir
    best_dir = Path(temp_dir) / "test" / "checkpoints" / "best"
    if best_dir.exists():
        best_files = list(best_dir.glob("*.pt"))
        print(f"\nFound {len(best_files)} best files:")
        for f in best_files:
            print(f"  {f.name}")
    else:
        print("Best dir doesn't exist!")
