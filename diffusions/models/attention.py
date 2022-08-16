from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        glu: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super(FeedForward, self).__init__()

        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        in_proj = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            in_proj,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super(GEGLU, self).__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class AttentionBlock(nn.Module):

    channels: int
    num_heads: int
    n_heads: int
    rescale_output_factor: float
    norm: nn.GroupNorm
    qkv: nn.Conv1d
    encoder_kv: nn.Conv1d
    proj: nn.Conv1d

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: Optional[int] = None,
        num_groups: int = 32,
        encoder_channels: Optional[int] = None,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels is None:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = nn.GroupNorm(
            num_channels=channels, num_groups=num_groups, eps=eps, affine=True
        )
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.n_heads = self.num_heads
        self.rescale_output_factor = rescale_output_factor

        if encoder_channels is not None:
            self.encoder_kv = nn.Conv1d(encoder_channels, channels * 2, 1)

        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, encoder_out: Optional[torch.Tensor] = None):
        B, C, *spatial = x.size()
        hidden_states = self.norm(x).view(B, C, -1)

        qkv = self.qkv(hidden_states)
        bs, width, length = qkv.size()
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)

        if encoder_out is not None:
            encoder_kv = self.encoder_kv(encoder_out)
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)

        scale = 1 / np.sqrt(np.sqrt(ch))
        weight = torch.einsum("bct, bcs -> bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        a = torch.einsum("bts, bcs -> bct", weight, v)
        h = a.reshape(bs, -1, length)

        h = self.proj(h)
        h = h.reshape(B, C, *spatial)

        out = x + h
        out = out / self.rescale_output_factor

        return out


class CrossAttention(nn.Module):

    heads: int
    scale: float
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    out_proj: nn.Sequential

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super(CrossAttention, self).__init__()

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        self.heads = heads
        self.scale = dim_head**-0.5

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)

        self.out_proj = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.size()

        query = self.q_proj(x)
        if context is None:
            key = self.k_proj(query)
            value = self.v_proj(query)
        else:
            key = self.k_proj(context)
            value = self.v_proj(context)

        query = rearrange(query, "bi(dh) -> bhid", h=self.heads)
        key = rearrange(key, "bj(dh) -> bhjd", h=self.heads)
        value = rearrange(value, "bj(dh) -> bhjd", h=self.heads)

        sim = torch.einsum("bhid, bhjd -> bhij", query, key) * self.scale

        if mask is not None:
            mask = mask.reshape(B, -1)
            big_neg = -torch.finfo(sim.dtype).max
            mask = mask[:, None, :].repeat(self.heads, 1, 1)
            sim.masked_fill_(~mask, big_neg)

        attention_weights = F.softmax(sim, dim=-1)

        out = torch.einsum("bhij, bhjd -> bhid", attention_weights, value)
        out = rearrange(out, "bhid -> bi(hd)")
        return out


class SpatialTransformer(nn.Module):

    n_heads: int
    d_head: int
    in_channels: int
    norm: nn.GroupNorm
    in_proj: nn.Conv2d
    transformer_blocks: nn.ModuleList
    out_proj: nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ) -> None:
        super(SpatialTransformer, self).__init__()

        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.norm = nn.GroupNorm(
            num_channels=in_channels, num_groups=32, eps=1e-6, affine=True
        )
        self.in_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=inner_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=inner_dim,
                    n_heads=n_heads,
                    d_head=d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                )
                for d in range(depth)
            ]
        )
        self.out_proj = nn.Conv2d(
            in_channels=inner_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_in = x
        x = self.norm(x)
        x = self.in_proj(x)
        x = rearrange(x, "bchw -> b(hw)c")

        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b(hw)c -> bchw")
        out = self.out_proj(x)
        return out + x_in


class TransformerBlock(nn.Module):

    checkpoint: bool
    attn_1: CrossAttention
    ff: FeedForward
    attn_2: CrossAttention
    norm_1: nn.LayerNorm
    norm_2: nn.LayerNorm
    norm_3: nn.LayerNorm

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
    ) -> None:
        super(TransformerBlock, self).__init__()

        self.attn_1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn_2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.norm_3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.attn_1(self.norm_1(x)) + x
        x = self.attn_2(self.norm_2(x), context=context) + x
        out = self.ff(self.norm_3(x)) + x
        return out
