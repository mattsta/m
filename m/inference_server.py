"""
Production Inference Server for MoE Transformers
Supports KV caching and prefix caching for efficient context reuse.
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
import socket
import time
from collections import OrderedDict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import uvicorn
from einops import einsum, rearrange
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from jsonargparse import CLI
from torch.nn import functional as F

# Add safe globals for checkpoint loading
torch.serialization.add_safe_globals([Path, pathlib.PosixPath])

from m.moe import (
    AttentionConfig,
    BlockConfig,
    ExpertConfig,
    ModelConfig,
    MoEConfig,
    MoESequenceRegressor,
    MultiheadSelfAttentionEinops,
    RouterConfig,
    TransformerBlock,
    get_device,
)

# ========================= Cache Data Types =========================


@dataclass(slots=True)
class AttentionCacheResult:
    """Result from cached attention forward pass."""

    output: torch.Tensor
    key_cache: torch.Tensor | None
    value_cache: torch.Tensor | None


@dataclass(slots=True)
class BlockCacheResult:
    """Result from cached transformer block forward pass."""

    output: torch.Tensor
    aux_loss: torch.Tensor
    key_cache: torch.Tensor | None
    value_cache: torch.Tensor | None


@dataclass(slots=True, frozen=True)
class KVState:
    """Key-Value state pair for attention caching."""

    key: torch.Tensor
    value: torch.Tensor


@dataclass(slots=True)
class ModelCacheResult:
    """Result from cached model forward pass."""

    logits: torch.Tensor
    kv_states: list[KVState] | None


# ========================= KV Cache =========================


@dataclass
class KVCacheEntry:
    """Single entry in the KV cache for a sequence."""

    key_states: list[torch.Tensor]  # [n_layers] each [B, H, S, D_head]
    value_states: list[torch.Tensor]  # [n_layers] each [B, H, S, D_head]
    seq_len: int
    last_access: float
    prefix_hash: str | None = None
    tokens: torch.Tensor | None = None  # Store tokens for prefix matching


class KVCache:
    """
    Manages key-value caches for multiple sequences with LRU eviction.
    Supports prefix sharing across requests.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        max_total_tokens: int = 1_000_000,
        device: str = "cuda",
    ):
        self.max_entries = max_entries
        self.max_total_tokens = max_total_tokens
        self.device = device
        self.cache: OrderedDict[str, KVCacheEntry] = OrderedDict()
        self.total_tokens = 0
        self.prefix_cache: dict[str, str] = {}  # prefix_hash -> cache_id mapping

    def _evict_lru(self, needed_tokens: int):
        """Evict least recently used entries to make room."""
        while (
            self.total_tokens + needed_tokens > self.max_total_tokens
            or len(self.cache) >= self.max_entries
        ) and self.cache:
            cache_id, entry = self.cache.popitem(last=False)
            self.total_tokens -= entry.seq_len
            # Remove from prefix cache if present
            if entry.prefix_hash and entry.prefix_hash in self.prefix_cache:
                del self.prefix_cache[entry.prefix_hash]

    def get(self, cache_id: str, move_to_end: bool = True) -> KVCacheEntry | None:
        """Retrieve cache entry and optionally mark as recently used."""
        if cache_id not in self.cache:
            return None
        entry = self.cache[cache_id]
        if move_to_end:
            self.cache.move_to_end(cache_id)
            entry.last_access = time.time()
        return entry

    def get_by_prefix(self, prefix_hash: str) -> tuple[str, KVCacheEntry] | None:
        """Retrieve cache entry by prefix hash."""
        if prefix_hash not in self.prefix_cache:
            return None
        cache_id = self.prefix_cache[prefix_hash]
        entry = self.get(cache_id)
        if entry is None:
            del self.prefix_cache[prefix_hash]
            return None
        return cache_id, entry

    def put(
        self,
        cache_id: str,
        key_states: list[torch.Tensor],
        value_states: list[torch.Tensor],
        prefix_hash: str | None = None,
        tokens: torch.Tensor | None = None,
    ):
        """Store new cache entry, evicting if necessary."""
        seq_len = key_states[0].shape[2] if key_states else 0
        self._evict_lru(seq_len)

        entry = KVCacheEntry(
            key_states=key_states,
            value_states=value_states,
            seq_len=seq_len,
            last_access=time.time(),
            prefix_hash=prefix_hash,
            tokens=tokens,
        )

        if cache_id in self.cache:
            old_entry = self.cache[cache_id]
            self.total_tokens -= old_entry.seq_len
            if old_entry.prefix_hash and old_entry.prefix_hash in self.prefix_cache:
                del self.prefix_cache[old_entry.prefix_hash]

        self.cache[cache_id] = entry
        self.total_tokens += seq_len

        if prefix_hash:
            self.prefix_cache[prefix_hash] = cache_id

    def extend(
        self,
        cache_id: str,
        new_keys: list[torch.Tensor],
        new_values: list[torch.Tensor],
    ):
        """Extend existing cache entry with new KV states."""
        entry = self.get(cache_id, move_to_end=False)
        if entry is None:
            self.put(cache_id, new_keys, new_values)
            return

        # Concatenate along sequence dimension
        for i, (k, v) in enumerate(zip(new_keys, new_values)):
            entry.key_states[i] = torch.cat([entry.key_states[i], k], dim=2)
            entry.value_states[i] = torch.cat([entry.value_states[i], v], dim=2)

        new_len = new_keys[0].shape[2]
        self.total_tokens += new_len
        entry.seq_len += new_len
        entry.last_access = time.time()
        self.cache.move_to_end(cache_id)

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.prefix_cache.clear()
        self.total_tokens = 0


# ========================= Prefix Cache =========================


class PrefixCache:
    """
    Manages prefix caching for fast context reuse.
    Computes hashes of token prefixes to identify reusable contexts.
    """

    def __init__(self, hash_chunk_size: int = 128):
        self.hash_chunk_size = hash_chunk_size
        self.prefix_hashes: dict[str, list[str]] = {}  # request_id -> prefix hashes

    def compute_prefix_hash(self, tokens: torch.Tensor, length: int) -> str:
        """Compute hash for a prefix of tokens."""
        prefix = tokens[:, :length].cpu().numpy().tobytes()
        return hashlib.sha256(prefix).hexdigest()

    def find_longest_prefix_match(
        self, tokens: torch.Tensor, kv_cache: KVCache
    ) -> tuple[int, str | None, KVCacheEntry | None]:
        """
        Find the longest matching prefix in cache by comparing actual token sequences.
        Returns: (prefix_length, cache_id, cache_entry)
        """
        best_length = 0
        best_cache_id = None
        best_entry = None

        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)

        seq_len = tokens.shape[1]

        # Iterate through all cached entries that have stored tokens
        for cache_id, entry in kv_cache.cache.items():
            if entry.tokens is not None:
                cached_tokens = entry.tokens
                if cached_tokens.ndim == 1:
                    cached_tokens = cached_tokens.unsqueeze(0)

                # Find the length of the common prefix
                max_check_len = min(seq_len, cached_tokens.shape[1], entry.seq_len)

                # Compare tokens element by element to find common prefix length
                for length in range(max_check_len, 0, -1):
                    if torch.equal(tokens[:, :length], cached_tokens[:, :length]):
                        if length > best_length:
                            best_length = length
                            best_cache_id = cache_id
                            best_entry = entry
                        break

        return best_length, best_cache_id, best_entry


# ========================= Attention with KV Cache =========================


class CachedMultiheadSelfAttention(MultiheadSelfAttentionEinops):
    """
    Extension of MultiheadSelfAttention with KV cache support.
    """

    def forward_with_cache(
        self,
        x: torch.Tensor,  # [B, S_new, D] - only new tokens
        past_key: torch.Tensor | None = None,  # [B, H, S_past, D_head]
        past_value: torch.Tensor | None = None,  # [B, H, S_past, D_head]
        use_cache: bool = True,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> AttentionCacheResult:
        """
        Forward with KV caching support.
        Returns: (output, new_key_states, new_value_states)
        """
        B, S_new, D = x.shape
        H, Dh = self.n_heads, self.d_head

        # Compute Q, K, V for new tokens only
        qkv = self.qkv(x)  # [B, S_new, 3D]
        q_new, k_new, v_new = rearrange(
            qkv, "b s (three h d) -> three b h s d", three=3, h=H, d=Dh
        )

        # Concatenate with past KV if available
        if past_key is not None and past_value is not None:
            k = torch.cat([past_key, k_new], dim=2)  # [B, H, S_total, Dh]
            v = torch.cat([past_value, v_new], dim=2)  # [B, H, S_total, Dh]
            S_total = k.shape[2]
        else:
            k = k_new
            v = v_new
            S_total = S_new

        # Compute attention scores
        scores = einsum(q_new, k, "b h s d, b h t d -> b h s t") * self.scale

        # Apply masks (causal mask for full sequence)
        if self.cfg.causal:
            # Create causal mask for new queries attending to all keys
            causal_mask = torch.ones(
                S_new, S_total, device=scores.device, dtype=torch.bool
            )
            # Allow attending to all past and current positions
            causal_mask = torch.triu(causal_mask, diagonal=S_total - S_new + 1)
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min
            )

        if key_padding_mask is not None:
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,S_total]
            scores = scores.masked_fill(kpm, torch.finfo(scores.dtype).min)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + attn_mask

        # Softmax and dropout
        probs = scores.softmax(dim=-1)
        probs = self.attn_drop(probs)

        # Weighted sum of values
        ctx = einsum(probs, v, "b h s t, b h t d -> b h s d")

        # Merge heads and project
        out = rearrange(ctx, "b h s d -> b s (h d)")
        out = self.proj(out)
        out = self.resid_drop(out)

        return AttentionCacheResult(
            output=out,
            key_cache=k if use_cache else None,
            value_cache=v if use_cache else None,
        )


# ========================= Cached Transformer Block =========================


class CachedTransformerBlock(TransformerBlock):
    """
    Transformer block with KV cache support.
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Replace attention with cached version
        D = cfg.moe.d_model
        self.attn: CachedMultiheadSelfAttention = CachedMultiheadSelfAttention(
            d_model=D, cfg=cfg.attn
        )

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
        use_cache: bool = True,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> BlockCacheResult:
        """
        Forward with KV caching.
        Returns: (output, aux_loss, new_key, new_value)
        """
        # Attention sublayer with caching
        z = self.ln1(x)
        attn_result = self.attn.forward_with_cache(
            z, past_key, past_value, use_cache, key_padding_mask, attn_mask
        )
        x = x + attn_result.output

        # MoE sublayer (no caching needed)
        z = self.ln2(x)
        moe_out, metrics = self.moe(z)
        x = x + moe_out

        return BlockCacheResult(
            output=x,
            aux_loss=metrics["aux_total"],
            key_cache=attn_result.key_cache,
            value_cache=attn_result.value_cache,
        )


# ========================= Cached Model =========================


class CachedMoEModel(nn.Module):
    """
    MoE model with full KV caching and prefix reuse support.
    """

    def __init__(self, base_model: MoESequenceRegressor):
        super().__init__()
        self.cfg = base_model.cfg
        self.in_proj = base_model.in_proj
        self.head = base_model.head
        self.aux_weight = base_model.aux_weight

        # Replace blocks with cached versions
        self.blocks: nn.ModuleList = nn.ModuleList(
            [CachedTransformerBlock(self.cfg.block) for _ in range(self.cfg.n_layers)]
        )

        # Copy weights from base model
        for i, block in enumerate(base_model.blocks):
            self.blocks[i].load_state_dict(block.state_dict())

    def forward_with_cache(
        self,
        x: torch.Tensor,
        past_kv_states: list[KVState] | None = None,
        use_cache: bool = True,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> ModelCacheResult:
        """
        Forward pass with KV caching.
        Returns: (logits, new_kv_states)
        """
        x = self.in_proj(x)

        new_kv_states = []
        aux_losses = []

        for i, blk in enumerate(self.blocks):
            cached_blk = cast(CachedTransformerBlock, blk)
            if past_kv_states and i < len(past_kv_states):
                kv_state = past_kv_states[i]
                past_k, past_v = kv_state.key, kv_state.value
            else:
                past_k, past_v = None, None
            block_result = cached_blk.forward_with_cache(
                x, past_k, past_v, use_cache, key_padding_mask, attn_mask
            )
            x = block_result.output
            aux_losses.append(block_result.aux_loss)
            if (
                use_cache
                and block_result.key_cache is not None
                and block_result.value_cache is not None
            ):
                new_kv_states.append(
                    KVState(key=block_result.key_cache, value=block_result.value_cache)
                )

        # Output projection
        if self.cfg.pool == "mean":
            x_pooled = x.mean(dim=1)
            logits = self.head(x_pooled)
        else:
            logits = self.head(x)

        return ModelCacheResult(
            logits=logits, kv_states=new_kv_states if use_cache else None
        )


# ========================= Inference Engine =========================


@dataclass
class InferenceRequest:
    """Single inference request."""

    request_id: str
    tokens: torch.Tensor  # [1, seq_len]
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    use_cache: bool = True
    reuse_prefix: bool = True


@dataclass
class InferenceResponse:
    """Response from inference engine."""

    request_id: str
    generated_tokens: torch.Tensor
    generation_time: float
    tokens_per_second: float
    cache_hit_rate: float
    prefix_reused_tokens: int


class InferenceEngine:
    """
    High-performance inference engine with KV caching and prefix reuse.
    """

    def __init__(
        self,
        model: MoESequenceRegressor,
        device: str = "cuda",
        max_batch_size: int = 32,
        kv_cache_size: int = 1000,
        kv_cache_max_tokens: int = 1_000_000,
    ):
        self.device = device
        self.max_batch_size = max_batch_size

        # Convert to cached model
        self.model = CachedMoEModel(model).to(device)
        self.model.eval()

        # Initialize caches
        self.kv_cache = KVCache(
            max_entries=kv_cache_size,
            max_total_tokens=kv_cache_max_tokens,
            device=device,
        )
        self.prefix_cache = PrefixCache()

        # Stats
        self.total_requests = 0
        self.cache_hits = 0
        self.total_prefix_tokens_reused = 0

    @torch.no_grad()
    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """
        Generate tokens for a single request with caching.
        """
        start_time = time.time()
        tokens = request.tokens.to(self.device)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        B, input_len = tokens.shape

        # Check for prefix match if enabled
        prefix_len = 0
        past_kv: list[KVState] | None = None
        reused_cache_id = None

        if request.use_cache and request.reuse_prefix:
            prefix_len, reused_cache_id, cache_entry = (
                self.prefix_cache.find_longest_prefix_match(tokens, self.kv_cache)
            )
            if cache_entry is not None and prefix_len > 0:
                # Reuse KV states from prefix
                # Extract only the prefix portion of the KV states
                past_kv = []
                for k, v in zip(cache_entry.key_states, cache_entry.value_states):
                    # k and v are [B, H, S, D] - we want [:, :, :prefix_len, :]
                    k_prefix = k[:, :, :prefix_len, :] if k.shape[2] > prefix_len else k
                    v_prefix = v[:, :, :prefix_len, :] if v.shape[2] > prefix_len else v
                    past_kv.append(KVState(key=k_prefix, value=v_prefix))

                self.cache_hits += 1
                self.total_prefix_tokens_reused += prefix_len

        # Process remaining tokens with cache
        if prefix_len < input_len:
            remaining_tokens = tokens[:, prefix_len:]

            # Convert token IDs to embeddings (create dummy embeddings for testing)
            # In production, this would use a proper embedding layer
            D = self.model.cfg.input_dim
            remaining_embeddings = torch.randn(
                B, remaining_tokens.shape[1], D, device=self.device
            )

            # Get model output with caching
            model_result = self.model.forward_with_cache(
                remaining_embeddings,
                past_kv_states=past_kv,
                use_cache=request.use_cache,
            )
            logits = model_result.logits
            new_kv = model_result.kv_states

            # Update KV cache
            if request.use_cache and new_kv is not None:
                if past_kv is None:
                    # Store new cache entry with tokens for future prefix matching
                    keys = [kv.key for kv in new_kv]
                    values = [kv.value for kv in new_kv]
                    prefix_hash = self.prefix_cache.compute_prefix_hash(
                        tokens, input_len
                    )
                    self.kv_cache.put(
                        request.request_id,
                        keys,
                        values,
                        prefix_hash=prefix_hash,
                        tokens=tokens.clone(),  # Store tokens for prefix matching
                    )
                else:
                    # Merge past KV with new KV and store complete cache
                    merged_keys = []
                    merged_values = []
                    for i, new_kv_state in enumerate(new_kv):
                        if i < len(past_kv):
                            past_kv_state = past_kv[i]
                            # Concatenate past and new along sequence dimension
                            merged_k = torch.cat(
                                [
                                    past_kv_state.key,
                                    new_kv_state.key[:, :, prefix_len:],
                                ],
                                dim=2,
                            )
                            merged_v = torch.cat(
                                [
                                    past_kv_state.value,
                                    new_kv_state.value[:, :, prefix_len:],
                                ],
                                dim=2,
                            )
                        else:
                            merged_k = new_kv_state.key
                            merged_v = new_kv_state.value
                        merged_keys.append(merged_k)
                        merged_values.append(merged_v)

                    # Store the complete merged cache
                    prefix_hash = self.prefix_cache.compute_prefix_hash(
                        tokens, input_len
                    )
                    self.kv_cache.put(
                        request.request_id,
                        merged_keys,
                        merged_values,
                        prefix_hash=prefix_hash,
                        tokens=tokens.clone(),
                    )
                past_kv = new_kv
        else:
            # Entire input was cached
            cached_entry = self.kv_cache.get(request.request_id)
            if cached_entry:
                past_kv = [
                    KVState(key=k, value=v)
                    for k, v in zip(cached_entry.key_states, cached_entry.value_states)
                ]

        # Autoregressive generation
        generated = []
        current_tokens = tokens
        D = self.model.cfg.input_dim

        for _ in range(request.max_new_tokens):
            # Get logits for last token
            if past_kv is not None:
                # Use cache - only process the last token
                # Convert to embeddings
                last_embeddings = torch.randn(B, 1, D, device=self.device)
                model_result = self.model.forward_with_cache(
                    last_embeddings,
                    past_kv_states=past_kv,
                    use_cache=request.use_cache,
                )
                logits = model_result.logits
                new_kv = model_result.kv_states
                if request.use_cache:
                    past_kv = new_kv
            else:
                # No cache - process all tokens
                all_embeddings = torch.randn(
                    B, current_tokens.shape[1], D, device=self.device
                )
                model_result = self.model.forward_with_cache(
                    all_embeddings,
                    use_cache=request.use_cache,
                )
                logits = model_result.logits
                new_kv = model_result.kv_states
                if request.use_cache:
                    past_kv = new_kv

            # Sample next token
            next_token_logits = logits[:, -1, :] if logits.ndim == 3 else logits
            next_token_logits = next_token_logits / request.temperature

            # Apply top-k filtering
            if request.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(
                    next_token_logits, min(request.top_k, next_token_logits.size(-1))
                )
                next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)

            # Apply top-p (nucleus) filtering
            if request.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > request.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated.append(next_token)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)

            # Check for EOS token (assuming 2 is EOS)
            if next_token.item() == 2:
                break

        # Calculate stats
        generation_time = time.time() - start_time
        num_generated = len(generated)
        tokens_per_second = (
            num_generated / generation_time if generation_time > 0 else 0
        )

        self.total_requests += 1
        cache_hit_rate = (
            self.cache_hits / self.total_requests if self.total_requests > 0 else 0
        )

        return InferenceResponse(
            request_id=request.request_id,
            generated_tokens=torch.cat(generated, dim=1)
            if generated
            else torch.empty(1, 0),
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            cache_hit_rate=cache_hit_rate,
            prefix_reused_tokens=prefix_len,
        )

    async def generate_batch(
        self,
        requests: list[InferenceRequest],
    ) -> list[InferenceResponse]:
        """
        Process multiple requests, potentially in parallel.
        """
        # For now, process sequentially
        # TODO: Implement true batching with attention masking
        responses = []
        for request in requests:
            response = self.generate(request)
            responses.append(response)
        return responses

    def clear_cache(self):
        """Clear all caches."""
        self.kv_cache.clear()
        self.prefix_cache = PrefixCache()
        self.cache_hits = 0
        self.total_prefix_tokens_reused = 0

    def get_stats(self) -> dict:
        """Get inference engine statistics."""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / self.total_requests
            if self.total_requests > 0
            else 0,
            "total_prefix_tokens_reused": self.total_prefix_tokens_reused,
            "kv_cache_entries": len(self.kv_cache.cache),
            "kv_cache_total_tokens": self.kv_cache.total_tokens,
        }


# ========================= Server Interface =========================


class InferenceServer:
    """
    HTTP/gRPC server wrapper for the inference engine.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        port: int = 8080,
    ):
        # Handle device="auto" properly using centralized device detection
        if device == "auto":
            device = get_device()

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Reconstruct model from config
        # Handle both direct config and nested config formats
        if "model" in checkpoint["config"]:
            # Training format: {"config": {"model": {...}, "training": {...}}}
            model_cfg_dict = checkpoint["config"]["model"]
        else:
            # Direct format: {"config": {...}}
            model_cfg_dict = checkpoint["config"]

        # Reconstruct nested config structure
        if isinstance(model_cfg_dict.get("block"), dict):
            # Need to reconstruct from dict format
            block_cfg = model_cfg_dict["block"]

            # Create nested configs
            attn_config = AttentionConfig(**block_cfg["attn"])

            # Filter router config to only include supported parameters
            router_cfg_dict = block_cfg["moe"]["router"]
            router_field_names = {field.name for field in fields(RouterConfig)}
            filtered_router_cfg = {
                k: v for k, v in router_cfg_dict.items() if k in router_field_names
            }
            router_config = RouterConfig(**filtered_router_cfg)

            expert_config = ExpertConfig(**block_cfg["moe"]["expert"])
            moe_config = MoEConfig(
                d_model=block_cfg["moe"]["d_model"],
                router=router_config,
                expert=expert_config,
            )
            block_config = BlockConfig(
                attn=attn_config,
                moe=moe_config,
                use_rms_norm=block_cfg["use_rms_norm"],
            )

            # Create the main model config
            model_config = ModelConfig(
                block=block_config,
                n_layers=model_cfg_dict["n_layers"],
                input_dim=model_cfg_dict["input_dim"],
                target_dim=model_cfg_dict["target_dim"],
                pool=model_cfg_dict.get("pool", "mean"),
            )
        else:
            # Already structured config objects
            model_config = ModelConfig(**model_cfg_dict)

        model = MoESequenceRegressor(model_config)

        # Handle both "model" and "model_state_dict" keys
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            raise ValueError("Checkpoint missing model weights")

        # Initialize inference engine
        self.engine = InferenceEngine(model, device=device)
        self.port = port

    async def handle_request(self, request_data: dict) -> dict:
        """Handle single inference request."""
        request = InferenceRequest(
            request_id=request_data["request_id"],
            tokens=torch.tensor(request_data["tokens"]),
            max_new_tokens=request_data.get("max_new_tokens", 100),
            temperature=request_data.get("temperature", 1.0),
            top_k=request_data.get("top_k", 50),
            top_p=request_data.get("top_p", 0.9),
            use_cache=request_data.get("use_cache", True),
            reuse_prefix=request_data.get("reuse_prefix", True),
        )

        response = self.engine.generate(request)

        return {
            "request_id": response.request_id,
            "generated_tokens": response.generated_tokens.tolist(),
            "generation_time": response.generation_time,
            "tokens_per_second": response.tokens_per_second,
            "cache_hit_rate": response.cache_hit_rate,
            "prefix_reused_tokens": response.prefix_reused_tokens,
        }

    def _find_available_port(self, start_port: int, max_attempts: int = 100) -> int:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("localhost", port))
                    return port
            except OSError:
                continue

        raise RuntimeError(
            f"Could not find available port in range {start_port}-{start_port + max_attempts}"
        )

    async def run(self, host: str = "0.0.0.0"):
        """Run the inference server with FastAPI."""
        try:
            # Check if imports are available (they should be since they're at top)
            pass
        except ImportError:
            raise ImportError(
                "FastAPI and uvicorn required for server. Install with: pip install fastapi uvicorn"
            )

        # Find available port if current port is in use
        actual_port = self._find_available_port(self.port)
        if actual_port != self.port:
            print(f"⚠️  Port {self.port} in use, using port {actual_port} instead")
            self.port = actual_port

        # Create FastAPI app
        app = FastAPI(
            title="MoE Inference Server",
            description="Production inference server for MoE transformers",
            version="1.0.0",
        )

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.post("/infer")
        async def infer(request_data: dict):
            """Handle inference request."""
            try:
                return await self.handle_request(request_data)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "stats": self.engine.get_stats()}

        @app.get("/stats")
        async def get_stats():
            """Get inference engine statistics."""
            return self.engine.get_stats()

        print(f"Inference server running on {host}:{self.port}")
        print(f"Stats: {self.engine.get_stats()}")
        print("🌐 API endpoints:")
        print(f"   POST http://{host}:{self.port}/infer - Run inference")
        print(f"   GET  http://{host}:{self.port}/health - Health check")
        print(f"   GET  http://{host}:{self.port}/stats - Engine statistics")

        # Run the server
        config = uvicorn.Config(app, host=host, port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


# ========================= CLI Interface =========================


def serve(
    model_path: str,
    device: str = "auto",
    port: int = 8080,
):
    """
    Start the inference server.

    Args:
        model_path: Path to model checkpoint
        device: Device to run on
        port: Port to serve on
    """
    server = InferenceServer(model_path, device, port)
    asyncio.run(server.run())


# CLI wrapper functions for entry points
def cli_serve():
    """CLI entry point for serving."""
    CLI([serve])


if __name__ == "__main__":
    CLI([serve])
