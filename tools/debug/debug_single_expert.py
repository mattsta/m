#!/usr/bin/env python3
"""Debug script for single expert issue."""

import torch

from m.moe import ExpertConfig, MoEConfig, RouterConfig, build_moe

# Test configuration with single expert
cfg = MoEConfig(
    d_model=64,
    router=RouterConfig(router_type="sinkhorn", n_experts=1, k=1),
    expert=ExpertConfig(d_model=64, d_hidden=128),
)

moe = build_moe(cfg).to("cpu")

# Test input
x = torch.randn(2, 8, 64)

# Calculate what capacity should be
B, S, D = x.shape
N = B * S  # 16 tokens
E = 1  # 1 expert
k = 1  # top-1
capacity_factor = 1.25

expected_capacity = int(capacity_factor * (N * k) / E)
print(f"N={N}, E={E}, k={k}")
print(f"Expected capacity: {expected_capacity}")
print(f"This means we want top {expected_capacity} tokens from dimension 0")
print(f"But P has shape [N, E] = [{N}, {E}]")
print(
    f"So torch.topk(P, k={expected_capacity}, dim=0) tries to select {expected_capacity} from dimension of size {N}"
)
print("This should work, but let's trace through...")

try:
    y, metrics = moe(x)
    print("Success!")
except RuntimeError as e:
    print(f"Error: {e}")

# Let's look at the actual Sinkhorn router behavior
from m.moe import SinkhornRouter

router = SinkhornRouter(d_model=64, cfg=cfg.router)
x_flat = x.view(-1, 64)  # [16, 64]
print(f"\nx_flat shape: {x_flat.shape}")

# Get logits and compute P
logits = router.route_logits(x_flat)  # [16, 1]
print(f"logits shape: {logits.shape}")

K = torch.exp(logits / max(cfg.router.sinkhorn_tau, 1e-6))
print(f"K shape: {K.shape}")

# The issue: we're trying to topk on P which has shape [16, 1]
# and we want top-20 from dimension 0, which has 16 elements!
print(f"\nThe problem: capacity={expected_capacity} > N={N}")
