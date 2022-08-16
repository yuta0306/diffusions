from typing import Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusions.models.activation import Swish


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        *,
        channel: int,
        dim: int,
        activation: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish], None] = nn.SiLU,
        **factory_kwargs,
    ) -> None:
        super(TimestepEmbedding, self).__init__()

        self.activation = (
            activation(**factory_kwargs) if activation is not None else None
        )
        self.linear_1 = nn.Linear(channel, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)

        if self.activation is not None:
            sample = self.activation(sample)

        return self.linear_2(sample)


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1,
        scale: float = 1,
        max_period: int = 10000,
    ) -> None:
        super(Timesteps, self).__init__()

        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        assert (
            timesteps.ndim == 1
        ), f"`timesteps` should be be a 1D tensor, but {timesteps.ndim}D tensor is given"

        half_dim = self.num_channels // 2

        coeff = -np.log(self.max_period) / (half_dim - self.downscale_freq_shift)
        emb = torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        emb = torch.exp(emb * coeff)
        emb = timesteps[:, None].float() * emb[None, :]

        emb = self.scale * emb

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))

        return emb


class GaussianFourierProjection(nn.Module):
    def __init__(
        self,
        embedding_size: int = 256,
        scale: float = 1,
    ) -> None:
        super(GaussianFourierProjection, self).__init__()

        self.weight = nn.parameter.Parameter(
            torch.randn(embedding_size) * scale, requires_grad=False
        )

    def forward(self, x):
        x = torch.log(x)
        x_proj = x[:, ...] * self.weight[..., :] * 2 * np.pi
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out
