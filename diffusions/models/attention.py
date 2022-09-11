import math
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusions.utils.jax import dynamic_slice, map_, scan
from einops import rearrange
from torch.utils.checkpoint import checkpoint


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
        in_proj = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU() if not glu else GEGLU(dim, inner_dim),
        )

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            in_proj,
            nn.Dropout(dropout),
            nn.LayerNorm(inner_dim),
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
    rescale_output_factor: float
    norm: nn.GroupNorm
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    o_proj: nn.Sequential

    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super(AttentionBlock, self).__init__()
        self.channels = channels

        self.num_heads = (
            channels // num_head_channels if num_head_channels is not None else 1
        )
        self.norm = nn.GroupNorm(
            num_channels=channels, num_groups=num_groups, eps=eps, affine=True
        )

        # q, k, v projection
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        # self.o_proj = nn.Linear(channels, channels)
        self.o_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(p=dropout),
            nn.LayerNorm(channels, eps=eps),
        )  # dropout and layer norm

        self.checkpoint = use_checkpoint

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.checkpoint:
            return checkpoint(self._forward, hidden_states)
        return self._forward(hidden_states)

    def _forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        B, C, H, W = hidden_states.size()

        # norm
        hidden_states = self.norm(hidden_states)
        hidden_states = einops.rearrange(hidden_states, "b c h w -> b (h w) c")
        assert hidden_states.size() == (B, H * W, C), hidden_states.size()

        # proj to q, k, v
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # split heads
        query = einops.rearrange(query, "b t (h d) -> b h t d", h=self.num_heads)
        key = einops.rearrange(key, "b t (h d) -> b h t d", h=self.num_heads)
        value = einops.rearrange(value, "b t (h d) -> b h t d", h=self.num_heads)

        # attention score
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attention_scores = torch.einsum(
            "bhid, bhjd -> bhij", query * scale, key
        )  # i = j = t

        attention_probs = F.softmax(
            attention_scores.float(), dim=-1, dtype=torch.float32
        ).type(attention_scores.dtype)

        # attention output
        context_states = torch.einsum(
            "bhij, bhjd -> bhid", attention_probs, value
        )  # i = j = t

        # bhid = bhtd -> bt(hd) = b(hw)c  -porj-> b(hw)c -> bchw
        context_states = einops.rearrange(context_states, "b h t d -> b t (h d)")
        hidden_states = self.o_proj(context_states)
        hidden_states = einops.rearrange(
            hidden_states, "b (h w) c -> b c h w", c=C, h=H, w=W
        )
        assert (
            hidden_states.size() == residual.size()
        ), f"{hidden_states.size()} != {residual.size()}"

        # residual connection
        hidden_states = (hidden_states + residual) / self.rescale_output_factor

        return hidden_states


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

        self.norm = nn.LayerNorm(query_dim)
        self.norm_context = (
            nn.LayerNorm(context_dim) if context_dim is not None else nn.Identity()
        )

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)

        self.out_proj = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(p=dropout),
            nn.LayerNorm(query_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.size()

        x = self.norm(x)
        if context is not None:
            context = self.norm_context(context)

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

        sim = torch.einsum("bhid, bhjd -> bhij", query * self.scale, key)

        if mask is not None:
            # TODO
            # implement value.masked_fill_(~mask, 0.0)
            mask = mask.reshape(B, -1)
            big_neg = -torch.finfo(sim.dtype).max
            mask = mask[:, None, :].repeat(self.heads, 1, 1)
            sim.masked_fill_(~mask, big_neg)

        attention_weights = F.softmax(sim, dim=-1, dtype=torch.float32).to(sim.dtype)

        out = torch.einsum("bhij, bhjd -> bhid", attention_weights, value)
        out = rearrange(out, "bhid -> bi(hd)")
        return self.out_proj(out)


class MemoryEfficientAttention(nn.Module):
    """Memory Efficient Attention
    `Self-attention Does Not Need $O(n^2)$ Memory <https://arxiv.org/ags/2112.05682>`
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`
    """

    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
        dropout: float = 0.0,
        query_chunk_size: int = 1024,
        key_chunk_size: int = 4096,  # 4096
    ) -> None:
        super(MemoryEfficientAttention, self).__init__()
        self.channels = channels

        self.num_heads = (
            channels // num_head_channels if num_head_channels is not None else 1
        )
        self.head_dim = num_head_channels if num_head_channels is not None else 1
        self.norm = nn.GroupNorm(
            num_channels=channels, num_groups=num_groups, eps=eps, affine=True
        )

        # q, k, v projection
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        # self.o_proj = nn.Linear(channels, channels)
        self.o_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(p=dropout),
            nn.LayerNorm(channels),
        )  # dropout and layer norm

        self.query_chunk_size = query_chunk_size
        self.key_chunk_size = key_chunk_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        B, C, H, W = hidden_states.size()
        num_q = H * W

        # norm
        hidden_states = self.norm(hidden_states)
        hidden_states = einops.rearrange(hidden_states, "b c h w -> b (h w) c")

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query_chunk_size = min(self.query_chunk_size, num_q)

        # split head
        query = einops.rearrange(query, "b t (h d) -> b t h d", h=self.num_heads)
        key = einops.rearrange(key, "b t (h d) -> b t h d", h=self.num_heads)
        value = einops.rearrange(value, "b t (h d) -> b t h d", h=self.num_heads)

        def _chunk_scanner(chunk_idx, _):
            query_chunk = dynamic_slice(
                query,
                (0, chunk_idx, 0, 0),
                sizes=(B, query_chunk_size, self.num_heads, self.head_dim),
            )
            return (
                chunk_idx + self.query_chunk_size,
                self._query_chunk_attention(query_chunk, key, value),
            )

        _, res = scan(
            _chunk_scanner,
            init=0,
            xs=None,
            length=math.ceil(num_q / self.query_chunk_size),
        )

        hidden_states = einops.rearrange(res, "c b t h d -> b (c t) (h d)")
        hidden_states = self.o_proj(hidden_states)
        hidden_states = einops.rearrange(
            hidden_states, "b (h w) c -> b c h w", h=H, w=W
        )

        hidden_states = (hidden_states + residual) / self.rescale_output_factor

        return hidden_states

    def _query_chunk_attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        B, num_kv, num_heads, k_features = key.size()
        v_features = value.size(-1)
        key_chunk_size = min(self.key_chunk_size, num_kv)
        query = query / torch.sqrt(torch.tensor(k_features))

        # with checkpointing
        def summarize_chunk(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ):
            attn_weights = torch.einsum("...qhd, ...khd -> ...qhk", query, key)
            max_score, _ = torch.max(attn_weights, dim=-1, keepdim=True)
            max_score = max_score.detach()
            exp_weights = torch.exp(attn_weights - max_score)
            exp_values = torch.einsum("...vhf, ...qhv -> ...qhf", value, exp_weights)
            max_score = torch.einsum("...qhk -> ...qh", max_score)
            return exp_values, exp_weights.sum(dim=-1), max_score

        def chunk_scanner(chunk_idx):
            key_chunk = dynamic_slice(
                key,
                (0, chunk_idx, 0, 0),
                sizes=(B, key_chunk_size, num_heads, k_features),
            )
            value_chunk = dynamic_slice(
                value,
                (0, chunk_idx, 0, 0),
                sizes=(B, key_chunk_size, num_heads, v_features),
            )

            return checkpoint(summarize_chunk, query, key_chunk, value_chunk)

        chunk_values, chunk_weights, chunk_max = map_(
            chunk_scanner, torch.arange(0, num_kv, key_chunk_size)
        )

        global_max, _ = torch.max(chunk_max, dim=0, keepdim=True)
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values *= torch.unsqueeze(max_diffs, dim=-1)
        chunk_weights *= max_diffs

        all_values = chunk_values.sum(dim=0)
        all_weights = torch.unsqueeze(chunk_weights, dim=-1).sum(dim=0)
        return all_values / all_weights


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


if __name__ == "__main__":
    inp = torch.randn((2, 64, 128, 128))
    print(128 * 128)
    model = MemoryEfficientAttention(64, num_head_channels=32)
    out = model(inp)
    print(out.size())
