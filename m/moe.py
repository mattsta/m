from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import signal
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import einsum, rearrange
from jsonargparse import CLI
from torch.utils.checkpoint import checkpoint as ckpt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ========================= Registries =========================
# Simple registries to allow drop-in extension of router/expert implementations.

ROUTER_REGISTRY: dict[str, type[nn.Module]] = {}
EXPERT_REGISTRY: dict[str, type[nn.Module]] = {}


def register_router(name: str):
    def deco(cls):
        ROUTER_REGISTRY[name] = cls
        return cls

    return deco


def register_expert(name: str):
    def deco(cls):
        EXPERT_REGISTRY[name] = cls
        return cls

    return deco


# ========================= Configs =========================
# All configs are dataclasses to integrate cleanly with jsonargparse.


@dataclass
class RouterConfig:
    # Router selection and basic routing policy
    router_type: Literal["topk", "sinkhorn"] = "sinkhorn"
    n_experts: int = 16
    k: int = 1
    capacity_factor: float = 1.25

    # Router logits shaping
    temperature: float = 1.0
    noise_type: Literal["none", "gaussian", "gumbel"] = "none"
    noise_std: float = 0.0
    router_dropout_prob: float = 0.0

    # Regularizers
    z_loss_weight: float = 0.0
    load_balance_weight: float = 1e-2
    entropy_weight: float = 0.0
    margin_weight: float = 0.0
    margin_target: float = 0.0

    # Mechanics
    renorm_after_drops: bool = True
    dispatch_mode: Literal["dense", "indices"] = (
        "indices"  # indices mode is memory friendly
    )
    use_router_ln: bool = False
    use_rms_norm: bool = True  # Use RMSNorm instead of LayerNorm for router
    router_bias: bool = False
    init_std: float = 0.0  # 0 => zeros init so routers start uniform

    # Sinkhorn-specific params
    sinkhorn_n_iter: int = 5
    sinkhorn_tau: float = 0.7
    sinkhorn_topk: int | None = None  # restrict candidate experts per token for speed


@dataclass
class ExpertConfig:
    expert_type: Literal["ffn"] = "ffn"
    d_model: int = 768
    d_hidden: int = 3072
    activation: Literal["gelu", "relu", "silu", "swiglu", "geglu", "reglu"] = "swiglu"
    dropout: float = 0.0
    bias: bool = True
    init: Literal[
        "xavier_uniform", "xavier_normal", "scaled_xavier", "scaled_kaiming"
    ] = "scaled_xavier"
    grouped_gemm: bool = True  # use bmm instead of einsum
    checkpoint_experts: bool = False  # activation checkpoint in expert forward


@dataclass
class MoEConfig:
    d_model: int = 768
    d_hidden: int = 3072
    router: RouterConfig = field(default_factory=RouterConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    fallback_policy: Literal["zero", "dense"] = "dense"
    fallback_weight: float = 1.0
    dtype: Literal["auto", "fp32", "fp16", "bf16"] = "auto"


# ========================= Routing Base =========================
# RoutingInfo is a lightweight container holding all outputs from routing.
# BaseRouter encapsulates common utilities (LayerNorm, noise, temperature, aux losses).


@dataclass
class RoutingInfo:
    # Core routing results (always required)
    combine_weights: torch.Tensor  # [N, E] per token weights after drop renorm
    kept_mask: (
        torch.Tensor
    )  # [N, E] bool, True if token routed to expert and kept under capacity
    capacity: int  # capacity per expert
    aux_lb: torch.Tensor  # scalar load-balance loss
    aux_z: torch.Tensor  # scalar z-loss

    # Optional fields with defaults
    top_idx: torch.Tensor | None = None  # [C, E] token indices per expert capacity slot
    valid_ce: torch.Tensor | None = None  # [C, E] bool, valid capacity slot
    aux_entropy: torch.Tensor | None = None  # scalar entropy regularizer
    aux_margin: torch.Tensor | None = None  # scalar margin regularizer
    gates: torch.Tensor | None = None  # [N, E] softmax probabilities (for metrics)
    logits_fp32: torch.Tensor | None = None  # [N, E] router logits (fp32)


class BaseRouter(nn.Module):
    def __init__(self, d_model: int, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        self.ln: nn.Module | None
        if cfg.use_router_ln:
            norm_class = RMSNorm if cfg.use_rms_norm else nn.LayerNorm
            self.ln = norm_class(d_model)
        else:
            self.ln = None
        self.router = nn.Linear(d_model, cfg.n_experts, bias=cfg.router_bias)
        # Initialization: zeros -> uniform softmax; otherwise small random
        if cfg.init_std == 0.0:
            nn.init.zeros_(self.router.weight)
            if cfg.router_bias:
                nn.init.zeros_(self.router.bias)
        else:
            nn.init.normal_(self.router.weight, mean=0.0, std=cfg.init_std)
            if cfg.router_bias:
                nn.init.normal_(self.router.bias, mean=0.0, std=cfg.init_std)

    def route_logits(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D], returns fp32 logits with optional noise and temperature
        if self.ln is not None:
            x = self.ln(x)
        logits = self.router(x).float()
        if self.training and self.cfg.noise_type != "none" and self.cfg.noise_std > 0:
            if self.cfg.noise_type == "gaussian":
                logits = logits + torch.randn_like(logits) * self.cfg.noise_std
            elif self.cfg.noise_type == "gumbel":
                # Standard Gumbel(0,1) noise for Noisy Top-K
                u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
                g = -torch.log(-torch.log(u))
                logits = logits + g * self.cfg.noise_std
        temp = max(self.cfg.temperature, 1e-6)
        logits = logits / temp
        return logits

    def _aux_from_gates(
        self,
        gates: torch.Tensor,
        kept_mask: torch.Tensor,
        logits: torch.Tensor,
        k: int,
        E: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load-balancing (Switch/GShard style): encourage equal expert usage
        p = gates.mean(dim=0)  # [E] average gate probability mass per expert
        assignment_fraction = kept_mask.to(gates.dtype).mean(dim=0) / float(max(k, 1))
        aux_lb = (E * (p * assignment_fraction).sum()) * self.cfg.load_balance_weight

        # z-loss stabilizes the router by penalizing extreme logsumexp
        z = logits.logsumexp(dim=-1)
        aux_z = (z.square().mean()) * self.cfg.z_loss_weight

        # Optional entropy regularizer: higher entropy routing early in training
        ent = -(gates.clamp_min(1e-9).log() * gates).sum(dim=-1).mean()
        aux_entropy = (
            -self.cfg.entropy_weight
        ) * ent  # negative sign => reward entropy

        # Optional margin regularizer: enforce top1 - top2 >= margin_target
        aux_margin = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if self.cfg.margin_weight > 0.0:
            v, _ = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1)
            if v.shape[-1] == 2:
                margin = v[:, 0] - v[:, 1]
                aux_margin = (
                    F.relu(self.cfg.margin_target - margin).mean()
                    * self.cfg.margin_weight
                )

        return aux_lb, aux_z, aux_entropy, aux_margin


# ========================= Routers =========================
# Two routers:
# - TopKRouter: classic MoE router with order-agnostic capacity assignment.
# - SinkhornRouter: uses Sinkhorn-Knopp to softly balance assignments before applying capacity.


@register_router("topk")
class TopKRouter(BaseRouter):
    def forward(self, x: torch.Tensor) -> RoutingInfo:
        # x: [B, S, D]
        B, S, D = x.shape
        N = B * S
        E, k, cfg = self.cfg.n_experts, self.cfg.k, self.cfg

        x_flat = rearrange(x, "b s d -> (b s) d")
        logits = self.route_logits(x_flat)  # [N, E]
        gates = F.softmax(logits, dim=-1)  # [N, E]

        # Token-wise top-k (candidates)
        # Handle case where k > n_experts (e.g., single expert with k=2)
        actual_k = min(k, E)
        topk_val_tok, topk_idx_tok = torch.topk(
            gates, k=actual_k, dim=-1
        )  # [N, actual_k]
        if self.training and cfg.router_dropout_prob > 0 and actual_k == 2:
            # With small probability, route only to second-best expert (exploration)
            bern = torch.rand(N, 1, device=x.device) < cfg.router_dropout_prob
            mask = torch.zeros_like(gates).scatter(1, topk_idx_tok, 1.0)
            second = torch.zeros_like(gates).scatter(1, topk_idx_tok[:, 1:2], 1.0)
            mask = torch.where(bern, second, mask)
        else:
            mask = torch.zeros_like(gates).scatter(1, topk_idx_tok, 1.0)

        # Capacity per expert
        capacity = max(1, int(cfg.capacity_factor * (N * k) / E))
        # Cap capacity at N since we can't select more tokens than we have
        capacity = min(capacity, N)

        # Expert-wise top capacity tokens across batch (order-agnostic over tokens)
        scores = gates.masked_fill(mask <= 0, float("-inf"))  # [N, E]
        top_val_ce, top_idx_ce = torch.topk(scores, k=capacity, dim=0)  # [C, E]
        valid_ce = torch.isfinite(
            top_val_ce
        )  # False where not enough tokens were selected

        # Build kept_mask [N, E] from valid capacity slots
        kept_mask = torch.zeros(N, E, device=x.device, dtype=torch.bool)
        if valid_ce.any():
            e_grid = (
                torch.arange(E, device=x.device).unsqueeze(0).expand_as(top_idx_ce)
            )  # [C, E]
            kept_mask[top_idx_ce[valid_ce], e_grid[valid_ce]] = True

        # Combine weights: renormalize on the remaining experts after capacity drops
        combine = gates * kept_mask.to(gates.dtype)
        if cfg.renorm_after_drops:
            denom = combine.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            combine = combine / denom

        aux_lb, aux_z, aux_entropy, aux_margin = self._aux_from_gates(
            gates, kept_mask, logits, k, E
        )

        return RoutingInfo(
            combine_weights=combine,
            kept_mask=kept_mask,
            capacity=capacity,
            top_idx=top_idx_ce,
            valid_ce=valid_ce,
            aux_lb=aux_lb,
            aux_z=aux_z,
            aux_entropy=aux_entropy,
            aux_margin=aux_margin,
            gates=gates,
            logits_fp32=logits,
        )


@register_router("sinkhorn")
class SinkhornRouter(BaseRouter):
    def forward(self, x: torch.Tensor) -> RoutingInfo:
        B, S, D = x.shape
        N = B * S
        E, k, cfg = self.cfg.n_experts, self.cfg.k, self.cfg

        x_flat = rearrange(x, "b s d -> (b s) d")
        logits = self.route_logits(x_flat)  # [N, E]
        K = torch.exp(logits / max(cfg.sinkhorn_tau, 1e-6))  # non-negative kernel

        # Optionally restrict per-token candidate experts to speed up Sinkhorn
        if cfg.sinkhorn_topk is not None and cfg.sinkhorn_topk < E:
            _, idx = torch.topk(K, k=cfg.sinkhorn_topk, dim=-1)
            mask = torch.zeros_like(K)
            mask.scatter_(1, idx, 1.0)
            K = K * mask

        # Sinkhorn-Knopp normalization: target row sums ~1, column sums ~ N/E
        r = torch.ones(N, device=x.device, dtype=K.dtype)
        c = torch.ones(E, device=x.device, dtype=K.dtype) * (N / E)

        u = torch.ones_like(r)
        v = torch.ones_like(c)
        eps = 1e-9
        for _ in range(cfg.sinkhorn_n_iter):
            Kv = torch.clamp(K @ v, min=eps)
            u = r / Kv
            KTu = torch.clamp(K.t() @ u, min=eps)
            v = c / KTu
        P = (u.unsqueeze(1) * K) * v.unsqueeze(0)  # [N, E]
        P = P / P.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # row-stochastic safeguard

        gates = F.softmax(logits, dim=-1)  # for metrics/loss consistency

        capacity = max(1, int(cfg.capacity_factor * (N * k) / E))
        # Cap capacity at N since we can't select more tokens than we have
        capacity = min(capacity, N)
        top_val_ce, top_idx_ce = torch.topk(P, k=capacity, dim=0)  # [C, E]
        valid_ce = torch.isfinite(top_val_ce)

        kept_mask = torch.zeros(N, E, device=x.device, dtype=torch.bool)
        if valid_ce.any():
            e_grid = torch.arange(E, device=x.device).unsqueeze(0).expand_as(top_idx_ce)
            kept_mask[top_idx_ce[valid_ce], e_grid[valid_ce]] = True

        combine = P * kept_mask.to(P.dtype)
        if cfg.renorm_after_drops:
            denom = combine.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            combine = combine / denom

        aux_lb, aux_z, aux_entropy, aux_margin = self._aux_from_gates(
            gates, kept_mask, logits, k, E
        )

        return RoutingInfo(
            combine_weights=combine,
            kept_mask=kept_mask,
            capacity=capacity,
            top_idx=top_idx_ce,
            valid_ce=valid_ce,
            aux_lb=aux_lb,
            aux_z=aux_z,
            aux_entropy=aux_entropy,
            aux_margin=aux_margin,
            gates=gates,
            logits_fp32=logits,
        )


# ========================= Experts =========================
# ExpertFFN implements both gated (SwiGLU/GEGLU/ReGLU) and ungated FFNs in a vectorized, per-expert fashion.


@register_expert("ffn")
class ExpertFFN(nn.Module):
    def __init__(self, cfg: ExpertConfig, n_experts: int, n_layers: int = 1):
        super().__init__()
        self.cfg = cfg
        E, D, H = n_experts, cfg.d_model, cfg.d_hidden
        self.gated = cfg.activation in ("swiglu", "geglu", "reglu")
        proj_in = 2 * H if self.gated else H

        self.W1 = nn.Parameter(torch.empty(E, D, proj_in))
        self.b1 = nn.Parameter(torch.zeros(E, proj_in)) if cfg.bias else None
        self.W2 = nn.Parameter(torch.empty(E, H, D))
        self.b2 = nn.Parameter(torch.zeros(E, D)) if cfg.bias else None
        self.drop = nn.Dropout(cfg.dropout)

        # Initialize weights based on configuration
        if cfg.init in ("scaled_xavier", "scaled_kaiming"):
            scaled_init_(self.W1, n_layers=n_layers, init_type=cfg.init)
            scaled_init_(self.W2, n_layers=n_layers, init_type=cfg.init)
        elif cfg.init == "xavier_uniform":
            nn.init.xavier_uniform_(self.W1)
            nn.init.xavier_uniform_(self.W2)
        else:
            nn.init.xavier_normal_(self.W1)
            nn.init.xavier_normal_(self.W2)

        Hh = H
        if self.gated:
            if cfg.activation == "swiglu":
                self.act = lambda x: F.silu(x[..., :Hh]) * x[..., Hh:]
            elif cfg.activation == "geglu":
                self.act = lambda x: F.gelu(x[..., :Hh]) * x[..., Hh:]
            elif cfg.activation == "reglu":
                self.act = lambda x: F.relu(x[..., :Hh]) * x[..., Hh:]
        else:
            if cfg.activation == "gelu":
                self.act = nn.GELU()
            elif cfg.activation == "relu":
                self.act = nn.ReLU()
            elif cfg.activation == "silu":
                self.act = nn.SiLU()

    def _forward_impl(self, x_ecd: torch.Tensor) -> torch.Tensor:
        # x_ecd: [E, C, D]
        E, C, D = x_ecd.shape
        H = self.cfg.d_hidden

        # First projection: either grouped GEMM (bmm) or einsum
        if self.cfg.grouped_gemm:
            # bmm expects [B, M, K] @ [B, K, N] -> [B, M, N], here B=E
            h = torch.bmm(x_ecd, self.W1)  # [E, C, H2]
        else:
            h = einsum(x_ecd, self.W1, "e c d, e d h2 -> e c h2")

        if self.b1 is not None:
            h = h + self.b1.unsqueeze(1)

        h = self.act(h)
        h = self.drop(h)
        if self.gated:
            h = h[..., :H]  # keep the "value" half after gating

        # Second projection
        if self.cfg.grouped_gemm:
            y = torch.bmm(h, self.W2)  # [E, C, D]
        else:
            y = einsum(h, self.W2, "e c h, e h d -> e c d")

        if self.b2 is not None:
            y = y + self.b2.unsqueeze(1)
        return y

    def forward(self, x_ecd: torch.Tensor) -> torch.Tensor:
        # Optional activation checkpointing to reduce memory
        if self.cfg.checkpoint_experts and self.training:
            return ckpt(self._forward_impl, x_ecd)
        return self._forward_impl(x_ecd)


class DenseFFN(nn.Module):
    # Compact dense fallback head with same activation choices as experts
    def __init__(self, cfg: ExpertConfig):
        super().__init__()
        D, H = cfg.d_model, cfg.d_hidden
        gated = cfg.activation in ("swiglu", "geglu", "reglu")
        proj_in = 2 * H if gated else H

        self.W1 = nn.Linear(D, proj_in, bias=cfg.bias)
        self.W2 = nn.Linear(H, D, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

        if cfg.init == "xavier_uniform":
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.xavier_uniform_(self.W2.weight)
        else:
            nn.init.xavier_normal_(self.W1.weight)
            nn.init.xavier_normal_(self.W2.weight)

        self.gated = gated
        self.H = H
        if gated:
            if cfg.activation == "swiglu":
                self.act = lambda x: F.silu(x[..., :H]) * x[..., H:]
            elif cfg.activation == "geglu":
                self.act = lambda x: F.gelu(x[..., :H]) * x[..., H:]
            elif cfg.activation == "reglu":
                self.act = lambda x: F.relu(x[..., :H]) * x[..., H:]
        else:
            if cfg.activation == "gelu":
                self.act = nn.GELU()
            elif cfg.activation == "relu":
                self.act = nn.ReLU()
            elif cfg.activation == "silu":
                self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.W1(x)
        h = self.act(h)
        h = self.drop(h)
        if self.gated:
            h = h[..., : self.H]
        return self.W2(h)


# ========================= Modern Initialization =========================
# Scaled initialization for better convergence in deep networks


def scaled_init_(
    tensor: torch.Tensor, n_layers: int = 1, init_type: str = "scaled_xavier"
) -> torch.Tensor:
    """
    Apply scaled initialization for better convergence in deep networks.

    Args:
        tensor: Tensor to initialize
        n_layers: Number of layers in the network (for residual scaling)
        init_type: Type of initialization ("scaled_xavier", "scaled_kaiming")
    """
    if init_type == "scaled_xavier":
        # Standard Xavier with residual scaling: std = 0.02 / sqrt(2 * n_layers)
        std = 0.02 / math.sqrt(2 * n_layers)
        nn.init.normal_(tensor, mean=0.0, std=std)
    elif init_type == "scaled_kaiming":
        # Kaiming with residual scaling
        nn.init.kaiming_normal_(tensor, mode="fan_in", nonlinearity="relu")
        tensor.data *= 1.0 / math.sqrt(2 * n_layers)
    else:
        # Fallback to Xavier uniform
        nn.init.xavier_uniform_(tensor)

    return tensor


# ========================= Modern Normalization =========================
# RMSNorm: More stable and efficient than LayerNorm, used in modern architectures


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More stable and efficient than LayerNorm, used in LLaMA, PaLM, and other modern models.
    """

    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Provides better position understanding and extrapolation than absolute position embeddings.
    Used in GPT-NeoX, PaLM, LLaMA, and other modern models.
    """

    def __init__(self, d_head: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for common sequence lengths
        self._cached_seq_len = 0
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Build or update the cos/sin cache for the given sequence length."""
        if seq_len <= self._cached_seq_len and self._cached_cos is not None:
            return

        # Create position indices
        inv_freq = cast(torch.Tensor, self.inv_freq)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)

        # Compute frequencies: outer product of positions and inv_freq
        freqs = einsum(t, inv_freq, "t, d -> t d")  # [seq_len, d_head//2]

        # Repeat frequencies to match full d_head: interleave each frequency twice
        freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, d_head]

        # Compute cos and sin
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)

        # Cache for reuse
        self._cached_seq_len = seq_len
        self._cached_cos = cos
        self._cached_sin = sin

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the features."""
        # x: [..., d_head]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to queries and keys.

        Args:
            q: Query tensor [..., seq_len, d_head]
            k: Key tensor [..., seq_len, d_head]

        Returns:
            Rotated (q, k) tensors with same shapes
        """
        seq_len = q.shape[-2]
        self._build_cache(seq_len, q.device, q.dtype)

        assert self._cached_cos is not None and self._cached_sin is not None
        cos = self._cached_cos[:seq_len]  # [seq_len, d_head]
        sin = self._cached_sin[:seq_len]  # [seq_len, d_head]

        # Apply rotation: q * cos + rotate_half(q) * sin
        # Ensure cos/sin match input dtype
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q_rotated = q * cos + self._rotate_half(q) * sin
        k_rotated = k * cos + self._rotate_half(k) * sin

        return q_rotated, k_rotated


# ========================= Attention (einops) =========================
# Multi-Head Self-Attention implemented with einops/einsum.
# Pre-LN blocks will pass LayerNorm(x) into this module.


@dataclass
class AttentionConfig:
    n_heads: int = 12  # number of attention heads
    attn_dropout: float = 0.0  # dropout on attention probabilities
    resid_dropout: float = 0.0  # dropout on the output projection
    bias: bool = True  # bias on linear layers
    causal: bool = True  # True for autoregressive decoding
    scale: float | None = None  # override 1/sqrt(d_head); rarely needed
    # Modern stability features
    use_rope: bool = True  # Use Rotary Position Embedding
    rope_max_seq_len: int = 2048  # Maximum sequence length for RoPE cache
    rope_base: float = 10000.0  # RoPE base frequency
    use_rms_norm: bool = True  # Use RMSNorm instead of LayerNorm
    init: Literal[
        "xavier_uniform", "xavier_normal", "scaled_xavier", "scaled_kaiming"
    ] = "scaled_xavier"


class MultiheadSelfAttentionEinops(nn.Module):
    """
    Multi-Head Self-Attention using einops/einsum.
    Shapes:
      x: [B, S, D]
      qkv projection: D -> 3*D (packed), reshaped to [B, H, S, d_head]
      attn weights: [B, H, S, S]
      output: [B, S, D]
    """

    def __init__(self, d_model: int, cfg: AttentionConfig, n_layers: int = 1):
        super().__init__()
        self.cfg = cfg
        H = cfg.n_heads
        assert d_model % H == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = H
        self.d_head = d_model // H
        self.scale = (1.0 / math.sqrt(self.d_head)) if cfg.scale is None else cfg.scale

        # Single fused projection for q,k,v to reduce memory traffic
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=cfg.bias)
        self.proj = nn.Linear(d_model, d_model, bias=cfg.bias)

        # Initialize weights with scaling
        if cfg.init in ("scaled_xavier", "scaled_kaiming"):
            scaled_init_(self.qkv.weight, n_layers=n_layers, init_type=cfg.init)
            scaled_init_(self.proj.weight, n_layers=n_layers, init_type=cfg.init)
        elif cfg.init == "xavier_uniform":
            nn.init.xavier_uniform_(self.qkv.weight)
            nn.init.xavier_uniform_(self.proj.weight)
        else:
            nn.init.xavier_normal_(self.qkv.weight)
            nn.init.xavier_normal_(self.proj.weight)

        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)

        # RoPE for position encoding
        self.rope = None
        if cfg.use_rope:
            self.rope = RoPE(
                d_head=self.d_head, max_seq_len=cfg.rope_max_seq_len, base=cfg.rope_base
            )

    def _build_causal_mask(self, S: int, device: torch.device) -> torch.Tensor:
        # Returns [1, 1, S, S] bool mask True where future positions should be masked
        return (
            torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def forward(
        self,
        x: torch.Tensor,  # [B, S, D]
        key_padding_mask: torch.Tensor | None = None,  # [B, S] (True where padding)
        attn_mask: torch.Tensor
        | None = None,  # [S, S] or [B, 1, S, S] additive or bool
    ) -> torch.Tensor:
        B, S, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # Project once then split into q, k, v, reshaping into heads
        qkv = self.qkv(x)  # [B, S, 3D]
        q, k, v = rearrange(qkv, "b s (three h d) -> three b h s d", three=3, h=H, d=Dh)

        # Apply RoPE if enabled
        if self.rope is not None:
            q, k = self.rope(q, k)

        # Scaled dot-product attention scores: [B, H, S, S]
        scores = einsum(q, k, "b h s d, b h t d -> b h s t") * self.scale

        # Apply masks:
        # - key_padding_mask: mask out key positions that are padding
        if key_padding_mask is not None:
            # key_padding_mask: [B, S] True for pad -> expand to [B, 1, 1, S]
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,S]
            scores = scores.masked_fill(kpm, torch.finfo(scores.dtype).min)

        # - causal: prevent attending to future positions
        if self.cfg.causal:
            causal = self._build_causal_mask(S, scores.device)  # [1,1,S,S]
            scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)

        # - custom attn_mask (bool or additive)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # broadcast to [B, H, S, S] if needed
                m = attn_mask
                if m.ndim == 2:  # [S, S]
                    m = m.unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(m, torch.finfo(scores.dtype).min)
            else:
                # additive mask: same broadcast rules
                m = attn_mask
                if m.ndim == 2:
                    m = m.unsqueeze(0).unsqueeze(0)
                scores = scores + m

        # Softmax over key positions
        probs = scores.softmax(dim=-1)
        probs = self.attn_drop(probs)

        # Weighted sum of values -> [B, H, S, d_head]
        ctx = einsum(probs, v, "b h s t, b h t d -> b h s d")

        # Merge heads and project
        out = rearrange(ctx, "b h s d -> b s (h d)")
        out = self.proj(out)
        out = self.resid_drop(out)
        return out


# ========================= Transformer Block (Pre-LN) =========================


@dataclass
class BlockConfig:
    attn: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    prenorm: bool = True  # Pre-LN is standard for stability
    use_rms_norm: bool = True  # Use RMSNorm instead of LayerNorm


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block: LN -> Attention -> residual -> LN -> MoE -> residual.
    Returns:
      x_out, aux_total (from MoE), metrics for loss aggregation.
    """

    def __init__(self, cfg: BlockConfig, n_layers: int = 1):
        super().__init__()
        self.cfg = cfg
        D = cfg.moe.d_model

        # Choose normalization layer
        norm_class = RMSNorm if cfg.use_rms_norm else nn.LayerNorm

        # Self-attention
        self.ln1 = norm_class(D)
        self.attn = MultiheadSelfAttentionEinops(
            d_model=D, cfg=cfg.attn, n_layers=n_layers
        )
        # MoE feed-forward
        self.ln2 = norm_class(D)
        self.moe = build_moe(
            cfg.moe, n_layers=n_layers
        )  # reuse builder from MoE section

    def forward(
        self,
        x: torch.Tensor,  # [B, S, D]
        key_padding_mask: torch.Tensor | None = None,  # [B, S]
        attn_mask: torch.Tensor | None = None,  # [S, S] or broadcastable
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Attention sublayer (Pre-LN)
        z = self.ln1(x)
        a = self.attn(z, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + a  # residual

        # MoE sublayer (Pre-LN)
        z = self.ln2(x)
        moe_out, metrics = self.moe(z)  # [B, S, D], metrics with 'aux_total'
        x = x + moe_out  # residual

        return x, metrics["aux_total"], metrics


# ========================= MoE Block =========================
# The MoEFeedForward block wires router + experts + combine logic.
# Indices mode includes the fixed valid_ce transpose bugfix.


class MoEFeedForward(nn.Module):
    def __init__(self, moe_cfg: MoEConfig, n_layers: int = 1):
        super().__init__()
        self.cfg = moe_cfg
        router_cls = ROUTER_REGISTRY[moe_cfg.router.router_type]
        self.router = router_cls(d_model=moe_cfg.d_model, cfg=moe_cfg.router)

        expert_cls = EXPERT_REGISTRY[moe_cfg.expert.expert_type]
        self.experts = expert_cls(
            cfg=moe_cfg.expert, n_experts=moe_cfg.router.n_experts, n_layers=n_layers
        )

        self.fallback = None
        if moe_cfg.fallback_policy == "dense":
            dense_cfg = ExpertConfig(
                expert_type="ffn",
                d_model=moe_cfg.expert.d_model,
                d_hidden=moe_cfg.expert.d_hidden,
                activation=moe_cfg.expert.activation,
                dropout=moe_cfg.expert.dropout,
                bias=moe_cfg.expert.bias,
                init=moe_cfg.expert.init,
            )
            self.fallback = DenseFFN(dense_cfg)

        self.metrics_callback: Callable[[dict[str, Any]], None] | None = None

    def set_metrics_callback(self, cb: Callable[[dict[str, Any]], None]):
        self.metrics_callback = cb

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        # x: [B, S, D]
        B, S, D = x.shape
        N = B * S
        E = self.cfg.router.n_experts

        routing = self.router(x)
        combine_ne = routing.combine_weights  # [N, E] post-drop renormalized
        kept_mask = routing.kept_mask  # [N, E] bool
        C = routing.capacity

        x_flat = rearrange(x, "b s d -> (b s) d")

        if self.cfg.router.dispatch_mode == "dense":
            # Build dispatch [E, C, N], one-hot, zero invalid
            dispatch = torch.zeros(E, C, N, device=x.device, dtype=x.dtype)
            e_ids = (
                torch.arange(E, device=x.device).unsqueeze(0).expand_as(routing.top_idx)
            )
            c_ids = (
                torch.arange(C, device=x.device).unsqueeze(1).expand_as(routing.top_idx)
            )
            valid = routing.valid_ce
            if valid.any():
                dispatch[e_ids[valid], c_ids[valid], routing.top_idx[valid]] = 1.0

            # Dispatch -> Experts -> Combine
            x_ecd = einsum(x_flat, dispatch, "n d, e c n -> e c d")  # [E, C, D]
            y_ecd = self.experts(x_ecd)  # [E, C, D]
            combine_ecn = dispatch * combine_ne.T.unsqueeze(1)  # [E, C, N]
            y_flat = einsum(y_ecd, combine_ecn, "e c d, e c n -> n d")  # [N, D]
        else:
            # Indices mode: gather (token_idx) -> experts -> scatter_add back
            token_idx = routing.top_idx.clamp(
                min=0
            )  # [C, E], invalid positions were -inf; clamp to 0 safely
            # Gather tokens: flatten [C, E] to [E*C], then reshape to [E, C, D]
            x_gather = x_flat.index_select(0, token_idx.view(-1))  # [E*C, D]
            x_ecd = x_gather.view(C, E, D).transpose(0, 1).contiguous()  # [E, C, D]
            # Zero invalid slots (BUGFIX: transpose valid_ce -> [E, C, 1])
            x_ecd = x_ecd * routing.valid_ce.T.unsqueeze(-1).to(x_ecd.dtype)
            # Experts
            y_ecd = self.experts(x_ecd)  # [E, C, D]
            # Per-slot weights: weight_ce[e, c] = combine_ne[token_idx[c, e], e]
            weights_ce = torch.gather(combine_ne.T, 1, token_idx.T)  # [E, C]
            weights_ce = weights_ce * routing.valid_ce.T.to(
                combine_ne.dtype
            )  # zero invalid
            # Weighted outputs -> flatten -> scatter back to tokens
            y_weighted = (
                (y_ecd * weights_ce.unsqueeze(-1))
                .transpose(0, 1)
                .contiguous()
                .view(-1, D)
            )  # [E*C, D]
            y_flat = torch.zeros(N, D, device=x.device, dtype=x.dtype)
            # Ensure y_weighted has same dtype as y_flat for index_add_
            y_weighted = y_weighted.to(y_flat.dtype)
            y_flat.index_add_(
                0, token_idx.view(-1), y_weighted
            )  # invalid slots add zeros

        # Fallback for tokens dropped by all experts
        dropped = kept_mask.sum(dim=-1) == 0  # [N]
        if dropped.any():
            if self.cfg.fallback_policy == "dense" and self.fallback is not None:
                y_flat[dropped] = y_flat[
                    dropped
                ] + self.cfg.fallback_weight * self.fallback(x_flat[dropped])
            # else: zero (typical in residual architectures)

        y = rearrange(y_flat, "(b s) d -> b s d", b=B, s=S)

        aux_total = (
            routing.aux_lb + routing.aux_z + routing.aux_entropy + routing.aux_margin
        )
        metrics = {
            "aux_total": aux_total,
            "aux_load_balance": routing.aux_lb,
            "aux_z": routing.aux_z,
            "aux_entropy": routing.aux_entropy,
            "aux_margin": routing.aux_margin,
            "fraction_dropped_tokens": dropped.float().mean(),
            "expert_utilization": kept_mask.float().mean(dim=0),  # [E]
            "gate_entropy": (
                -(routing.gates.clamp_min(1e-9).log() * routing.gates).sum(-1).mean()
            ),
            "capacity": torch.tensor(float(C), device=x.device),
        }
        if self.metrics_callback is not None:
            try:
                self.metrics_callback(
                    {
                        k: (v.detach() if isinstance(v, torch.Tensor) else v)
                        for k, v in metrics.items()
                    }
                )
            except Exception:
                pass
        return y, metrics


# ========================= Builders and Demo =========================


def build_moe(moe_cfg: MoEConfig, n_layers: int = 1) -> MoEFeedForward:
    """Factory that constructs the MoE block and casts dtype if requested."""
    model = MoEFeedForward(moe_cfg, n_layers=n_layers)
    if moe_cfg.dtype != "auto":
        if moe_cfg.dtype == "fp32":
            model = model.float()
        elif moe_cfg.dtype == "fp16":
            model = model.half()
        elif moe_cfg.dtype == "bf16":
            model = model.bfloat16()
    return model


@torch.no_grad()
def print_summary(model: nn.Module):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params / 1e6:.2f}M")


# ========================= Minimal Model for Training Demo (with Attention) =========================


@dataclass
class ModelConfig:
    # Transformer-like stack of [Attention + MoE] blocks
    block: BlockConfig = field(default_factory=BlockConfig)
    n_layers: int = 1
    input_dim: int = 768
    target_dim: int = 1
    pool: Literal["none", "mean"] = "mean"  # sequence pooling for the head
    torch_compile: bool = False


class MoESequenceRegressor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        D_in = cfg.input_dim
        D = cfg.block.moe.d_model

        # Input projection if needed
        self.in_proj = nn.Identity() if D_in == D else nn.Linear(D_in, D, bias=False)

        # Stack of Transformer blocks: LN->Attention->res -> LN->MoE->res
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.block, n_layers=cfg.n_layers)
                for _ in range(cfg.n_layers)
            ]
        )

        # Output head
        self.head = nn.Linear(D, cfg.target_dim)

        # Weight for auxiliary MoE loss (sum over blocks)
        self.aux_weight = 1e-2

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ):
        # x: [B, S, D_in]; key_padding_mask: [B, S] bool; attn_mask optional
        x = self.in_proj(x)

        aux_losses = []
        all_metrics = []
        for blk in self.blocks:
            x, aux, metrics = blk(
                x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
            aux_losses.append(aux)
            all_metrics.append(metrics)

        if self.cfg.pool == "mean":
            x_pooled = x.mean(dim=1)
            logits = self.head(x_pooled)  # [B, T]
        else:
            logits = self.head(x)  # [B, S, T]

        # Aggregate metrics from all blocks
        aggregated_metrics = {}
        if all_metrics:
            # Average metrics across blocks
            for key in all_metrics[0].keys():
                if key in ["expert_utilization"]:
                    # For expert utilization, average across blocks
                    values = [m[key] for m in all_metrics if key in m]
                    if values:
                        aggregated_metrics[key] = torch.stack(values).mean(dim=0)
                else:
                    # For scalar metrics, average
                    values = [m[key] for m in all_metrics if key in m]
                    if values:
                        if isinstance(values[0], torch.Tensor):
                            aggregated_metrics[key] = torch.stack(values).mean()
                        else:
                            aggregated_metrics[key] = sum(values) / len(values)

            # Add aux_total for compatibility
            if aux_losses:
                aggregated_metrics["aux_total"] = torch.stack(aux_losses).mean()

        return logits, aggregated_metrics


# ========================= Synthetic Dataset (Demo) =========================
# A toy dataset generating supervised regression pairs:
# y = (mean over sequence) of a fixed linear projection of x, with noise.
# Replace this with your own Dataset/DataLoader in real projects.


@dataclass
class DataConfig:
    train_size: int = 2048
    val_size: int = 512
    batch_size: int = 32
    seq_len: int = 128
    input_dim: int = 768
    target_dim: int = 1
    num_workers: int = 4
    pin_memory: bool = True


class ToySeqRegression(Dataset):
    def __init__(
        self, n: int, seq_len: int, input_dim: int, target_dim: int, seed: int = 0
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # True mapping: project input_dim -> target_dim
        self.W_true = torch.randn(input_dim, target_dim, generator=g) / math.sqrt(
            input_dim
        )
        self.n = n
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.g = g

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        x = torch.randn(self.seq_len, self.input_dim, generator=self.g)
        y = (x @ self.W_true).mean(dim=0)  # [target_dim], mean over sequence
        y = y + 0.05 * torch.randn(self.target_dim, generator=self.g)  # noise
        return x, y


# ========================= Optim/Scheduler Configs =========================


@dataclass
class OptimConfig:
    # Separate LRs for router vs experts vs rest
    lr_main: float = 1e-3
    lr_router: float | None = None  # if None, use lr_main
    lr_expert: float | None = None
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    # Gradient scaling
    max_grad_norm: float = 1.0
    # AMP
    amp_dtype: Literal["fp16", "bf16"] | None = "bf16"
    # Accumulation for large batches
    grad_accum_steps: int = 1


@dataclass
class SchedConfig:
    total_steps: int = 10000
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1  # cosine decay down to min_ratio * base_lr


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sched: SchedConfig = field(default_factory=SchedConfig)
    # Trainer behavior
    epochs: int = 10
    steps_per_epoch: int | None = (
        None  # if not None, cap steps per epoch (for quick tests)
    )
    log_interval: int = 50
    val_interval: int = 500
    ckpt_interval: int = 1000
    out_dir: str = "runs/moe"
    run_name: str = "exp1"
    keep_last: int = 3
    keep_best: int = 3
    seed: int = 42
    deterministic: bool = False
    resume: str | None = None  # path to checkpoint to resume
    early_stop_patience: int | None = None
    torch_compile: bool = False


# ========================= Utilities =========================


def set_seed(seed: int, deterministic: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cosine_with_warmup(step: int, cfg: SchedConfig) -> float:
    # Returns LR scale factor in [min_lr_ratio, 1]
    if step < cfg.warmup_steps:
        return max((step + 1) / max(1, cfg.warmup_steps), 1e-8)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine


def param_groups(model: nn.Module, cfg: OptimConfig):
    # Create parameter groups so router/experts can have different LRs if desired.
    router_params = []
    expert_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".router." in name:
            router_params.append(p)
        elif ".experts." in name:
            expert_params.append(p)
        else:
            other_params.append(p)
    lr_router = cfg.lr_router if cfg.lr_router is not None else cfg.lr_main
    lr_expert = cfg.lr_expert if cfg.lr_expert is not None else cfg.lr_main
    return [
        {"params": router_params, "lr": lr_router, "weight_decay": cfg.weight_decay},
        {"params": expert_params, "lr": lr_expert, "weight_decay": cfg.weight_decay},
        {"params": other_params, "lr": cfg.lr_main, "weight_decay": cfg.weight_decay},
    ]


def save_config_yaml(path: str, cfg: Any):
    try:
        with open(path, "w") as f:
            yaml.safe_dump(json.loads(json.dumps(asdict(cfg))), f, sort_keys=False)
    except Exception:
        # Fallback to JSON if YAML not installed
        with open(path, "w") as f:
            json.dump(asdict(cfg), f, indent=2)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now():
    return time.strftime("%Y%m%d_%H%M%S")


def get_device(pref: str | None = None) -> str:
    """
    Auto-detect the best available device with proper fallback hierarchy.

    Args:
        pref: Preferred device string. If provided, returns as-is (no validation).

    Returns:
        Device string following hierarchy: parameter → cuda → mps → cpu
    """
    if pref:
        return pref

    # Check CUDA first (highest performance for ML)
    if torch.cuda.is_available():
        return "cuda"

    # Check MPS (Apple Silicon GPU acceleration)
    if torch.backends.mps.is_available():
        return "mps"

    # Fallback to CPU
    return "cpu"


# ========================= Checkpoint Manager =========================


class CheckpointManager:
    """
    Handles saving/loading checkpoints and retention.
    Snapshot contains:
      - model_state, optimizer_state, scheduler_state, scaler_state, trainer_state (step/epoch),
        RNG states (torch, cuda), config snapshot.
    Keeps latest and top-K best by validation metric.
    """

    def __init__(self, out_dir: str, run_name: str, keep_last: int, keep_best: int):
        self.dir = os.path.join(out_dir, run_name)
        ensure_dir(self.dir)
        self.ckpt_dir = os.path.join(self.dir, "checkpoints")
        ensure_dir(self.ckpt_dir)
        self.best_dir = os.path.join(self.ckpt_dir, "best")
        ensure_dir(self.best_dir)
        self.last_ckpts: list[str] = []
        self.best_ckpts: list[tuple[float, str]] = []  # (metric, path)

    def _prune(self, paths: list[str], keep: int):
        if len(paths) <= keep:
            return paths
        for p in paths[:-keep]:
            if os.path.exists(p):
                os.remove(p)
        return paths[-keep:]

    def _prune_best(self):
        self.best_ckpts.sort(key=lambda x: x[0])  # lower metric is better
        if len(self.best_ckpts) > 0 and len(self.best_ckpts) > keep_best_global:
            # Remove the worst checkpoints (highest metrics)
            for _, p in self.best_ckpts[keep_best_global:]:
                if os.path.exists(p):
                    os.remove(p)
            # Keep the best checkpoints (lowest metrics)
            self.best_ckpts = self.best_ckpts[:keep_best_global]

    def save(
        self,
        tag: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler_state: dict,
        scaler: torch.cuda.amp.GradScaler | None,
        trainer_state: dict,
        config_snapshot: dict,
        is_best: float | None = None,
    ) -> str:
        path = os.path.join(self.ckpt_dir, f"{tag}.pt")
        # Package RNG states for reproducibility
        rng_state = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler_state,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "trainer": trainer_state,
            "rng": rng_state,
            "config": config_snapshot,
            "time": time.time(),
        }
        torch.save(payload, path)
        # Update "latest" symlink/copy
        latest_path = os.path.join(self.ckpt_dir, "latest.pt")
        try:
            if os.path.islink(latest_path):
                os.remove(latest_path)
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(path), latest_path)
        except Exception:
            shutil.copyfile(path, latest_path)
        # Retention: last K
        self.last_ckpts.append(path)
        self.last_ckpts = self._prune(self.last_ckpts, keep_last_global)
        # Best retention
        if is_best is not None:
            best_path = os.path.join(
                self.best_dir, f"best_{tag}_metric{is_best:.6f}.pt"
            )
            shutil.copyfile(path, best_path)
            self.best_ckpts.append((is_best, best_path))
            self._prune_best()
        return path

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.cuda.amp.GradScaler | None = None,
    ) -> dict:
        payload = torch.load(path, map_location="cpu")
        model.load_state_dict(payload["model"], strict=True)
        if (
            optimizer is not None
            and "optimizer" in payload
            and payload["optimizer"] is not None
        ):
            optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and "scaler" in payload and payload["scaler"] is not None:
            scaler.load_state_dict(payload["scaler"])
        # Restore RNG
        if "rng" in payload and payload["rng"] is not None:
            torch.set_rng_state(payload["rng"]["cpu"])
            if torch.cuda.is_available() and payload["rng"]["cuda"] is not None:
                torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
        return payload.get("trainer", {})


# Global keep settings needed in manager prune (set in trainer init)
keep_last_global = 3
keep_best_global = 3


# ========================= Trainer =========================


class Trainer:
    """
    Production-style single-process trainer (fits CPU/CUDA, AMP, resume, snapshots).
    Swap DataLoaders/Model with your own project; the training scaffolding remains the same.
    """

    def __init__(self, cfg: TrainConfig, device: str | None = None):
        self.cfg = cfg
        self.device = get_device()

        global keep_last_global, keep_best_global
        keep_last_global = cfg.keep_last
        keep_best_global = cfg.keep_best

        set_seed(cfg.seed, cfg.deterministic)

        # Build model and optionally compile
        self.model: nn.Module = MoESequenceRegressor(cfg.model).to(self.device)
        if cfg.torch_compile or cfg.model.torch_compile:
            try:
                self.model = cast(nn.Module, torch.compile(self.model))  # PyTorch 2.x
            except Exception as e:
                print(f"torch.compile failed; continuing without compile: {e}")

        # Parameter groups with specialized LRs for router/experts
        self.optimizer = torch.optim.AdamW(
            param_groups(self.model, cfg.optim),
            lr=cfg.optim.lr_main,
            betas=cfg.optim.betas,
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )

        # AMP policy
        self.amp_dtype = None
        if cfg.optim.amp_dtype == "fp16":
            self.amp_dtype = torch.float16
        elif cfg.optim.amp_dtype == "bf16":
            self.amp_dtype = torch.bfloat16

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.amp_dtype == torch.float16 and torch.cuda.is_available())
        )

        # Scheduler state: we store only the current step and recompute LR factor with cosine+warmup
        self.global_step = 0
        self.epoch = 0
        self.best_val = float("inf")
        self.steps_since_improvement = 0

        # Prepare output dirs and checkpoint manager
        self.out_dir = os.path.join(cfg.out_dir, cfg.run_name)
        ensure_dir(self.out_dir)
        self.ckpt = CheckpointManager(
            cfg.out_dir, cfg.run_name, cfg.keep_last, cfg.keep_best
        )

        # Install signal handlers to snapshot on SIGINT/SIGTERM
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Build data
        self.train_loader, self.val_loader = self._build_data(cfg.data)

        # If resuming, restore everything
        if cfg.resume is not None:
            self._resume(cfg.resume)

        # Persist configs for traceability
        save_config_yaml(os.path.join(self.out_dir, "train_config.yaml"), cfg)

        # Open JSONL logger for metrics
        self.log_path = os.path.join(self.out_dir, "train_log.jsonl")
        self.log_file = open(self.log_path, "a", buffering=1)

    def _build_data(self, dcfg: DataConfig):
        # Build synthetic dataset; replace with your own for real training
        train_ds = ToySeqRegression(
            dcfg.train_size,
            dcfg.seq_len,
            dcfg.input_dim,
            dcfg.target_dim,
            seed=self.cfg.seed,
        )
        val_ds = ToySeqRegression(
            dcfg.val_size,
            dcfg.seq_len,
            dcfg.input_dim,
            dcfg.target_dim,
            seed=self.cfg.seed + 1,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=dcfg.batch_size,
            shuffle=True,
            num_workers=dcfg.num_workers,
            pin_memory=dcfg.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=dcfg.batch_size,
            shuffle=False,
            num_workers=dcfg.num_workers,
            pin_memory=dcfg.pin_memory,
            drop_last=False,
        )
        return train_loader, val_loader

    def _resume(self, path: str):
        print(f"Resuming from {path}")
        trainer_state = self.ckpt.load(path, self.model, self.optimizer, self.scaler)
        if "global_step" in trainer_state:
            self.global_step = int(trainer_state["global_step"])
            self.epoch = int(trainer_state.get("epoch", 0))
            self.best_val = float(trainer_state.get("best_val", float("inf")))
            self.steps_since_improvement = int(trainer_state.get("ssi", 0))
        else:
            print("Warning: No trainer state in checkpoint; continuing with defaults.")

    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}; saving emergency snapshot...")
        self._save_snapshot(tag=f"signal_{now()}")

    def _save_snapshot(self, tag: str, is_best: float | None = None):
        # Compose "scheduler_state" as minimal info; we recompute LR factor each step.
        sched_state = {"global_step": self.global_step}
        trainer_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val": self.best_val,
            "ssi": self.steps_since_improvement,
        }
        cfg_snapshot = asdict(self.cfg)
        path = self.ckpt.save(
            tag=tag,
            model=self.model,
            optimizer=self.optimizer,
            scheduler_state=sched_state,
            scaler=self.scaler,
            trainer_state=trainer_state,
            config_snapshot=cfg_snapshot,
            is_best=is_best,
        )
        print(f"Saved checkpoint: {path}")
        return path

    def _set_lr(self):
        # Cosine + warmup schedule applied as LR multipliers to each param group
        scale = cosine_with_warmup(self.global_step, self.cfg.sched)
        for pg in self.optimizer.param_groups:
            base_lr = pg.get(
                "initial_lr", pg["lr"]
            )  # keep base in "initial_lr" if present
            pg["lr"] = base_lr * scale

    def _log_jsonl(self, record: dict):
        self.log_file.write(json.dumps(record) + "\n")

    def validate(self) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                with (
                    torch.cuda.amp.autocast(dtype=self.amp_dtype)
                    if self.amp_dtype
                    else contextlib.nullcontext()
                ):
                    preds, metrics = self.model(xb, yb)
                    # Compute loss from predictions and targets
                    main_loss = F.mse_loss(preds, yb)
                    aux_loss = metrics.get("aux_total", 0.0)
                    loss = main_loss + self.model.aux_weight * aux_loss
                losses.append(loss.item())
        val_loss = float(sum(losses) / max(1, len(losses)))
        self.model.train()
        return val_loss

    def train(self):
        self.model.train()
        total_steps_target = self.cfg.sched.total_steps

        # TQDM progress per epoch
        for epoch in range(self.epoch, self.cfg.epochs):
            self.epoch = epoch
            pbar = tqdm(
                self.train_loader,
                total=self.cfg.steps_per_epoch or len(self.train_loader),
                desc=f"Epoch {epoch}",
                leave=False,
            )
            accum = 0
            accum_loss = 0.0

            for step, batch in enumerate(pbar):
                if self.cfg.steps_per_epoch and step >= self.cfg.steps_per_epoch:
                    break

                xb, yb = batch  # xb: [B,S,D_in], yb: [B, target_dim]
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                # Update LR according to schedule
                self._set_lr()

                # Forward + loss under AMP
                with (
                    torch.cuda.amp.autocast(dtype=self.amp_dtype)
                    if self.amp_dtype
                    else contextlib.nullcontext()
                ):
                    preds, metrics = self.model(xb, yb)
                    # Compute loss from predictions and targets
                    main_loss = F.mse_loss(preds, yb)
                    aux_loss = metrics.get("aux_total", 0.0)
                    loss = main_loss + self.model.aux_weight * aux_loss
                    # Scale loss for gradient accumulation
                    loss = loss / max(1, self.cfg.optim.grad_accum_steps)

                # Backward with scaler if fp16; else standard backward
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum += 1
                accum_loss += loss.item()

                # Step optimizer after accumulating enough microsteps
                if accum % self.cfg.optim.grad_accum_steps == 0:
                    # Gradient clipping
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optim.max_grad_norm
                    )

                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # Global step increments only after optimizer step
                    self.global_step += 1
                    avg_loss = accum_loss
                    accum = 0
                    accum_loss = 0.0

                    # Logging to tqdm
                    pbar.set_postfix(
                        {"step": self.global_step, "loss": f"{avg_loss:.4f}"}
                    )

                    # Periodic JSONL logging
                    if self.global_step % self.cfg.log_interval == 0:
                        self._log_jsonl(
                            {
                                "time": time.time(),
                                "epoch": self.epoch,
                                "step": self.global_step,
                                "loss": avg_loss,
                                "lr": [pg["lr"] for pg in self.optimizer.param_groups],
                            }
                        )

                    # Periodic validation
                    if self.global_step % self.cfg.val_interval == 0:
                        val_loss = self.validate()
                        improved = val_loss < self.best_val
                        if improved:
                            self.best_val = val_loss
                            self.steps_since_improvement = 0
                        else:
                            self.steps_since_improvement += 1

                        print(
                            f"\nVal @ step {self.global_step}: loss={val_loss:.4f} (best={self.best_val:.4f})"
                        )
                        self._log_jsonl(
                            {
                                "time": time.time(),
                                "epoch": self.epoch,
                                "step": self.global_step,
                                "val_loss": val_loss,
                                "best_val": self.best_val,
                            }
                        )

                        # Early stopping
                        if (
                            self.cfg.early_stop_patience is not None
                            and self.steps_since_improvement
                            >= self.cfg.early_stop_patience
                        ):
                            print("Early stopping.")
                            self._save_snapshot(
                                tag=f"stop_step{self.global_step}", is_best=val_loss
                            )
                            return

                        # Save "best" snapshot
                        if improved:
                            self._save_snapshot(
                                tag=f"best_step{self.global_step}", is_best=val_loss
                            )

                    # Periodic snapshot (latest)
                    if self.global_step % self.cfg.ckpt_interval == 0:
                        self._save_snapshot(tag=f"step{self.global_step}")

                    # Stop when reaching target total steps if provided
                    if total_steps_target and self.global_step >= total_steps_target:
                        print("Reached target total steps; stopping.")
                        self._save_snapshot(
                            tag=f"final_step{self.global_step}", is_best=self.best_val
                        )
                        return

            # End of epoch snapshot
            self._save_snapshot(tag=f"epoch{self.epoch}")

        # Final snapshot at end of training
        self._save_snapshot(tag=f"final_step{self.global_step}", is_best=self.best_val)
        print("Training complete.")

    def close(self):
        try:
            self.log_file.close()
        except Exception:
            pass


# ========================= Simple CLI Entrypoints =========================
# - build_moe: build and return MoE block (debugging).
# - demo_run:  one forward/backward step demo.
# - train:     full training loop with snapshots and logging.


def demo_run(
    moe_cfg: MoEConfig,
    batch_size: int = 4,
    seq_len: int = 128,
    seed: int = 0,
    device: str = get_device(),
    autocast_dtype: Literal["fp16", "bf16"] | None = None,
):
    torch.manual_seed(seed)
    model = build_moe(moe_cfg).to(device)
    print_summary(model)
    x = torch.randn(batch_size, seq_len, moe_cfg.d_model, device=device)
    y = torch.randn(batch_size, seq_len, moe_cfg.d_model, device=device)
    amp_dtype = (
        torch.float16
        if autocast_dtype == "fp16"
        else (torch.bfloat16 if autocast_dtype == "bf16" else None)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if amp_dtype is not None and device.startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler(enabled=(autocast_dtype == "fp16"))
        with torch.cuda.amp.autocast(dtype=amp_dtype):
            out, metrics = model(x)
            loss = (out - y).pow(2).mean() + metrics["aux_total"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        out, metrics = model(x)
        loss = (out - y).pow(2).mean() + metrics["aux_total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"Output: {tuple(out.shape)}, loss={loss.detach().item():.4f}")
    return {"loss": loss.detach().item()}


def train(cfg: TrainConfig, device: str = get_device()):
    trainer = Trainer(cfg, device=device)
    try:
        trainer.train()
    finally:
        trainer.close()
    return {"best_val": trainer.best_val, "global_step": trainer.global_step}


# ============== Main CLI ==============
# CLI wrapper functions for entry points
def cli_train():
    """CLI entry point for training."""
    CLI([train])


def cli_demo():
    """CLI entry point for demo."""
    CLI([demo_run])


def cli_build():
    """CLI entry point for building MoE."""
    CLI([build_moe])


# Expose build_moe, demo_run, and train via jsonargparse CLI.
if __name__ == "__main__":
    CLI([build_moe, demo_run, train])
