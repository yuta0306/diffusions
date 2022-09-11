from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from diffusions.models.activation import Swish
from diffusions.models.attention import (
    AttentionBlock,
    MemoryEfficientAttention,
    SpatialTransformer,
)
from diffusions.models.resnet import Downsample2D, ResnetBlock, Upsample2D


class UNetMidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        eps: float = 1e-6,
        time_scale_shift: str = "default",
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        groups: int = 32,
        pre_norm: bool = True,
        num_head_channels: int = 1,
        output_scale_factor: float = 1.0,
        **factory_kwargs,
    ) -> None:
        super(UNetMidBlock, self).__init__()

        attentions = [
            AttentionBlock(
                channels=in_channels,
                num_head_channels=num_head_channels,
                rescale_output_factor=output_scale_factor,
                eps=eps,
                num_groups=groups,
            )
            for _ in range(num_layers)
        ]

        resnets = [
            ResnetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=eps,
                groups=groups,
                dropout=dropout,
                time_embedding_norm=time_scale_shift,
                non_linearity=non_linearity,
                output_scale_factor=output_scale_factor,
                pre_norm=pre_norm,
                **factory_kwargs,
            )
            for _ in range(num_layers + 1)
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if isinstance(attn, AttentionBlock):
                hidden_states = attn(hidden_states)
            else:
                hidden_states = attn(hidden_states, encoder_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNetMidCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        eps: float = 1e-6,
        time_scale_shift: str = "default",
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        groups: int = 32,
        pre_norm: bool = True,
        num_head_channels: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        **factory_kwargs,
    ) -> None:
        super(UNetMidCrossAttentionBlock, self).__init__()

        attentions = [
            SpatialTransformer(
                in_channels=in_channels,
                n_heads=num_head_channels,
                d_head=in_channels // num_head_channels,
                depth=1,
                context_dim=cross_attention_dim,
            )
            for _ in range(num_layers)
        ]

        resnets = [
            ResnetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=eps,
                groups=groups,
                dropout=dropout,
                time_embedding_norm=time_scale_shift,
                non_linearity=non_linearity,
                output_scale_factor=output_scale_factor,
                pre_norm=pre_norm,
                **factory_kwargs,
            )
            for _ in range(num_layers + 1)
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        eps: float = 1e-6,
        time_scale_shift: str = "default",
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        groups: int = 32,
        pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        **factory_kwargs,
    ) -> None:
        super(DownBlock, self).__init__()

        resnets = [
            ResnetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=eps,
                groups=groups,
                dropout=dropout,
                time_embedding_norm=time_scale_shift,
                non_linearity=non_linearity,
                output_scale_factor=output_scale_factor,
                pre_norm=pre_norm,
                **factory_kwargs,
            )
            for i in range(num_layers)
        ]
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        channels=out_channels if num_layers > 1 else in_channels,
                        out_channels=out_channels,
                        padding=downsample_padding,
                    )
                ]
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        states: Tuple[torch.Tensor, ...] = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            states += (hidden_states,)

        return hidden_states, states


class AttnDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        eps: float = 1e-6,
        time_scale_shift: str = "default",
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        groups: int = 32,
        pre_norm: bool = True,
        num_head_channels: int = 1,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        memory_efficient: bool = False,
        **factory_kwargs,
    ) -> None:
        super(AttnDownBlock, self).__init__()

        resnets = [
            ResnetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=eps,
                groups=groups,
                dropout=dropout,
                time_embedding_norm=time_scale_shift,
                non_linearity=non_linearity,
                output_scale_factor=output_scale_factor,
                pre_norm=pre_norm,
                **factory_kwargs,
            )
            for i in range(num_layers)
        ]
        attentions = [
            MemoryEfficientAttention(
                channels=out_channels,
                num_head_channels=num_head_channels,
                num_groups=groups,
                rescale_output_factor=output_scale_factor,
                eps=eps,
                dropout=dropout,
            )
            if memory_efficient
            else AttentionBlock(
                channels=out_channels,
                num_head_channels=num_head_channels,
                num_groups=groups,
                rescale_output_factor=output_scale_factor,
                eps=eps,
                dropout=dropout,
                use_checkpoint=True,
            )
            for _ in range(num_layers)
        ]

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        channels=out_channels if num_layers > 1 else in_channels,
                        out_channels=out_channels,
                        padding=downsample_padding,
                    )
                ]
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        states: Tuple[torch.Tensor, ...] = ()

        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            states += (hidden_states,)

        return hidden_states, states


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        eps: float = 1e-6,
        time_scale_shift: str = "default",
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        groups: int = 32,
        pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        **factory_kwargs,
    ) -> None:
        super(UpBlock, self).__init__()

        resnets = []
        for i in range(num_layers):
            skip_channels = in_channels if (i == num_layers - 1) else out_channels
            in_channels = prev_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=in_channels + skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=eps,
                    groups=groups,
                    dropout=dropout,
                    time_embedding_norm=time_scale_shift,
                    non_linearity=non_linearity,
                    output_scale_factor=output_scale_factor,
                    pre_norm=pre_norm,
                    **factory_kwargs,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        channels=out_channels,
                        out_channels=out_channels,
                    )
                ]
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = torch.cat([hidden_states, res_hidden_states[-1]], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class AttnUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        eps: float = 1e-6,
        time_scale_shift: str = "default",
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        groups: int = 32,
        pre_norm: bool = True,
        num_head_channels: int = 1,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        memory_efficient: bool = False,
        **factory_kwargs,
    ) -> None:
        super(AttnUpBlock, self).__init__()

        resnets = []
        attentions = []
        for i in range(num_layers):

            resnets.append(
                ResnetBlock(
                    in_channels=(prev_channels if i == 0 else out_channels)
                    + (in_channels if (i == num_layers - 1) else out_channels),
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=eps,
                    groups=groups,
                    dropout=dropout,
                    time_embedding_norm=time_scale_shift,
                    non_linearity=non_linearity,
                    output_scale_factor=output_scale_factor,
                    pre_norm=pre_norm,
                    **factory_kwargs,
                )
            )
            attentions.append(
                MemoryEfficientAttention(
                    channels=out_channels,
                    num_head_channels=num_head_channels,
                    num_groups=groups,
                    rescale_output_factor=output_scale_factor,
                    eps=eps,
                    dropout=dropout,
                )
                if memory_efficient
                else AttentionBlock(
                    channels=out_channels,
                    num_head_channels=num_head_channels,
                    num_groups=groups,
                    rescale_output_factor=output_scale_factor,
                    eps=eps,
                    dropout=dropout,
                    use_checkpoint=True,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        channels=out_channels,
                        out_channels=out_channels,
                    )
                ]
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for attn, resnet in zip(self.attentions, self.resnets):
            res_states = res_hidden_states[-1]
            res_hidden_states = res_hidden_states[:-1]
            hidden_states = torch.cat([hidden_states, res_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
