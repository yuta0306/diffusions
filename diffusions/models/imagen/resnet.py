from typing import Optional, Type, Union

import torch
import torch.nn as nn
from diffusions.models.activation import Swish


class EfficientResNetBlock(nn.Module):

    in_channels: int
    out_channels: int
    groups: int
    groups_out: int
    output_scale_factor: float
    norm1: nn.GroupNorm
    norm2: nn.GroupNorm
    act1: Union[Swish, nn.SiLU, nn.Mish]
    act2: Union[Swish, nn.SiLU, nn.Mish]
    conv1: nn.Conv2d
    conv2: nn.Conv2d
    conv_shortcut: nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        groups: int = 8,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        output_scale_factor: float = 1.0,
        **factory_kwargs,
    ) -> None:
        super(EfficientResNetBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.groups = groups
        if groups_out is None:
            self.groups_out = groups

        self.norm1 = nn.GroupNorm(
            num_groups=groups,
            num_channels=self.groups_out,
            eps=eps,
            affine=True,
        )

        if isinstance(non_linearity, Swish):
            beta = factory_kwargs.get("beta")
            self.act1 = non_linearity(beta=beta)
        else:
            self.act1 = non_linearity()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=groups,
            num_channels=self.groups_out,
            eps=eps,
            affine=True,
        )

        if isinstance(non_linearity, Swish):
            beta = factory_kwargs.get("beta")
            self.act2 = non_linearity(beta=beta)
        else:
            self.act2 = non_linearity()

        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
        )

        self.conv_shortcut = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
        )

        self.output_scale_factor = output_scale_factor

    def forward(
        self, x: torch.Tensor, scale_shift: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = x

        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        if scale_shift is not None:
            scale, shift = scale_shift
            h = h * (h + scale) + shift
        h = self.act2(h)
        h = self.conv2(h)

        x = self.conv_shortcut(x)

        # h = (h + x) / self.output_scale_factor

        return h
