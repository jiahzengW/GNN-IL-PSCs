#!/usr/bin/env python3
"""
attention_layer.py

Highly robust and configurable attention mechanisms for graph and sequence models:
  - Scaled dot-product attention with optional masking and dropout
  - Multi-head attention with residual connections, layer normalization, and dropout
  - Extensive input validation and type hints to reduce misuse
  - Single-head pooling attention for flexible aggregation

Usage examples:
    attn = MultiHeadAttention(embed_dim=64, num_heads=8)
    out = attn(query, key, value, mask=mask)

    pool = AttentionPool(input_dim=128)
    pooled = pool(node_features)
"""

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Parameter
import math


class ScaledDotProductAttention(Module):
    """Compute scaled dot-product attention with optional mask."""
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = Dropout(dropout)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Tensor = None
                ) -> Tensor:
        """
        Args:
            query: (batch, heads, seq_len, head_dim)
            key:   (batch, heads, seq_len, head_dim)
            value: (batch, heads, seq_len, head_dim)
            mask:  optional boolean mask broadcastable to (batch, heads, seq_len, seq_len)
        Returns:
            output: (batch, heads, seq_len, head_dim)
        """
        # Validate tensor dimensions
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError("query, key, and value must be 4-D tensors")
        batch, heads, seq_q, dim_q = query.shape
        _, _, seq_k, dim_k = key.shape
        if dim_q != dim_k:
            raise ValueError("query and key must have same head_dim")

        # Compute scaled dot-product
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_q)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, value)
        return output


class MultiHeadAttention(Module):
    """Multi-head attention with residual and normalization."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1
                 ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        # Output projection
        self.out_proj = Linear(embed_dim, embed_dim)

        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.dropout = Dropout(dropout)
        self.norm = LayerNorm(embed_dim)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Tensor = None
                ) -> Tensor:
        """
        Args:
            query, key, value: (batch, seq_len, embed_dim)
            mask: boolean tensor broadcastable to (batch, num_heads, seq_len, seq_len)
        Returns:
            Tensor: (batch, seq_len, embed_dim)
        """
        # Initial shape checks
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("query/key/value must be 3-D tensors")

        batch, seq_len, _ = query.shape
        # Project and reshape for multi-head
        def shape(x: Tensor) -> Tensor:
            return x.view(batch, seq_len, self.num_heads, self.head_dim) \
                    .transpose(1, 2)  # (batch, heads, seq_len, head_dim)

        Q = shape(self.q_proj(query))
        K = shape(self.k_proj(key))
        V = shape(self.v_proj(value))

        # Compute attention
        attn_out = self.attn(Q, K, V, mask=mask)
        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).contiguous()  # (batch, seq_len, heads, head_dim)
        attn_out = attn_out.view(batch, seq_len, self.embed_dim)

        # Final linear + dropout
        out = self.out_proj(attn_out)
        out = self.dropout(out)

        # Residual + LayerNorm
        return self.norm(query + out)


class AttentionPool(Module):
    """Simple attention-based pooling for node features."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = Linear(input_dim, 1)
        self.norm = LayerNorm(input_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: node features, shape (num_nodes, input_dim)
        Returns:
            pooled: shape (1, input_dim)
        """
        if x.dim() != 2:
            raise ValueError("Input to AttentionPool must be 2-D")
        scores = self.proj(x).squeeze(-1)  # (num_nodes,)
        weights = torch.softmax(scores, dim=0).unsqueeze(1)  # (num_nodes,1)
        aggregated = (x * weights).sum(dim=0, keepdim=True)  # (1, input_dim)
        return self.norm(aggregated)


# Alias for backward compatibility
MultiHeadGraphAttention = MultiHeadAttention
