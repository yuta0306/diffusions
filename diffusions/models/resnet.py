from typing import Any, Optional, Type, Union

import torch
import torch.nn as nn
from diffusions.models.activation import Swish


class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        # conv: Optional[Any] = None,
        out_channels: Optional[int] = None,
        **factory_kwargs: Any,
    ) -> None:
        super(Upsample2D, self).__init__()

        self.channels = channels
        self.out_channels = channels
        if out_channels is not None:
            self.out_channels = out_channels

        # if isinstance(conv, nn.ConvTranspose2d):
        self.conv = nn.ConvTranspose2d(
            in_channels=self.channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.channels, f"{x.size(1)} != {self.channels}"
        out = self.conv(x)
        return out


class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        padding: int = 1,
        **factory_kwargs: Any,
    ) -> None:
        super(Downsample2D, self).__init__()

        self.channels = channels
        self.out_channels = channels
        if out_channels is not None:
            self.out_channels = out_channels
        self.padding = padding

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=2,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.channels, f"{x.size(1)} != {self.channels}"
        out = self.conv(x)
        return out


class ResnetBlock(nn.Module):

    nonlinearity: Union[Swish, nn.SiLU, nn.Mish]

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = False,
        eps: float = 1e-6,
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        time_embedding_norm: str = "default",
        kernel: Optional[str] = None,
        output_scale_factor: float = 1.0,
        use_nin_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        **factory_kwargs,
    ) -> None:
        super(ResnetBlock, self).__init__()

        self.pre_norm = pre_norm
        self.in_channles = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        # self.conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
        )
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, self.out_channels)
        else:
            self.time_emb_proj = nn.Identity()

        self.norm2 = nn.GroupNorm(
            num_groups=groups_out,
            num_channels=self.out_channels,
            eps=eps,
            affine=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if isinstance(non_linearity, Swish):
            beta = factory_kwargs.get("beta")
            self.nonlinearity = non_linearity(beta=beta)
        else:
            self.nonlinearity = non_linearity()

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample2D(channels=in_channels)
        if self.down:
            self.downsample = Downsample2D(channels=in_channels, padding=1)

        self.use_nin_shortcut = (
            self.in_channles != self.out_channels
            if use_nin_shortcut is None
            else use_nin_shortcut
        )

        self.conv_shortcut = None
        if self.use_nin_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        hey: bool = False,
    ) -> torch.Tensor:
        h = x

        h = self.norm1(h)
        h = self.nonlinearity(h)

        if self.upsample is not None:
            x = self.upsample(x)
            h = self.upsample(h)
        elif self.downsample is not None:
            x = self.downsample(x)
            h = self.downsample(h)

        h = self.conv1(h)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            h = h + temb

        h = self.norm2(h)
        h = self.nonlinearity(h)

        h = self.dropout(h)
        h = self.conv2(h)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        out = (x + h) / self.output_scale_factor

        return out

    def load_weight_from_resnet(self, resnet):
        self.norm1.weight.data = resnet.norm1.weight.data
        self.norm1.bias.data = resnet.norm1.bias.data

        self.conv1.weight.data = resnet.conv1.weight.data
        self.conv1.bias.data = resnet.conv1.bias.data

        if self.time_emb_proj is not None:
            self.time_emb_proj.weight.data = resnet.temb_proj.weight.data
            self.time_emb_proj.bias.data = resnet.temb_proj.bias.data

        self.norm2.weight.data = resnet.norm2.weight.data
        self.norm2.bias.data = resnet.norm2.bias.data

        self.conv2.weight.data = resnet.conv2.weight.data
        self.conv2.bias.data = resnet.conv2.bias.data

        if self.use_nin_shortcut:
            self.conv_shortcut.weight.data = resnet.nin_shortcut.weight.data
            self.conv_shortcut.bias.data = resnet.nin_shortcut.bias.data
