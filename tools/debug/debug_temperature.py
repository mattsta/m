#!/usr/bin/env python3
"""Debug temperature scaling test."""

import torch

from m.moe import BaseRouter, RouterConfig

# Create router with temperature 2.0
router_config = RouterConfig(n_experts=4, temperature=2.0, init_std=0.1)
router = BaseRouter(d_model=128, cfg=router_config)

x = torch.randn(10, 128)

# Save original weights
original_weights = router.router.weight.data.clone()

logits = router.route_logits(x)
print(f"Temperature 2.0 logits mean abs: {logits.abs().mean()}")

# Create router with temperature 1.0 but same weights
router_config_1 = RouterConfig(n_experts=4, temperature=1.0, init_std=0.1)
router2 = BaseRouter(d_model=128, cfg=router_config_1)
router2.router.weight.data = original_weights
if router.router.bias is not None:
    router2.router.bias.data = router.router.bias.data.clone()

logits2 = router2.route_logits(x)
print(f"Temperature 1.0 logits mean abs: {logits2.abs().mean()}")

# Let's trace through manually
raw_logits = router.router(x).float()
print(f"\nRaw logits mean abs: {raw_logits.abs().mean()}")
print(f"After temp=2.0: {(raw_logits / 2.0).abs().mean()}")
print(f"After temp=1.0: {(raw_logits / 1.0).abs().mean()}")

# The issue: if init_std=0, raw logits are 0, so temperature has no effect!
print("\n--- With zero init ---")
router_config_zero = RouterConfig(n_experts=4, temperature=2.0, init_std=0.0)
router_zero = BaseRouter(d_model=128, cfg=router_config_zero)
logits_zero = router_zero.route_logits(x)
print(f"Zero init logits: {logits_zero}")
print(f"All zeros? {torch.allclose(logits_zero, torch.zeros_like(logits_zero))}")
