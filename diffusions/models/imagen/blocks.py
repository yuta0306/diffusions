from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from diffusions.models.activation import Swish
from diffusions.models.attention import AttentionBlock
from diffusions.models.imagen.resnet import EfficientResNetBlock


class EfficientUNetMidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        groups: int = 8,
        add_attention: bool = False,
        num_heads: int = 8,
        output_scale_factor: float = 1.0,
        **factory_kwargs,
    ) -> None:
        super(EfficientUNetMidBlock, self).__init__()

        self.mid_1 = EfficientResNetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            groups=groups,
            eps=eps,
            non_linearity=non_linearity,
            output_scale_factor=output_scale_factor,
            **factory_kwargs,
        )
        self.mid_2 = EfficientResNetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            groups=groups,
            eps=eps,
            non_linearity=non_linearity,
            output_scale_factor=output_scale_factor,
            **factory_kwargs,
        )

        self.attention = None
        if add_attention:
            self.attention = AttentionBlock(
                channels=in_channels,
                num_head_channels=in_channels // num_heads,
                num_groups=groups,
                rescale_output_factor=output_scale_factor,
                eps=eps,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.mid_1(hidden_states, temb)

        if self.attention is not None:
            hidden_states = self.attention(hidden_states)

        hidden_states = self.mid_2(hidden_states, temb)

        return hidden_states


class EfficientDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: Optional[int] = None,
        stride: Optional[Tuple[int, int]] = None,
        num_layers: int = 1,
        eps: float = 1e-6,
        time_scale_shift: str = "default",
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        groups: int = 8,
        output_scale_factor: float = 1.0,
        add_downsampling: bool = True,
        add_attention: bool = False,
        num_heads: int = 8,
        **factory_kwargs,
    ) -> None:
        super(EfficientDownBlock, self).__init__()

        self.downsampler = None
        if add_downsampling:
            self.downsampler = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride if stride is not None else 2,
                padding=1,
            )

        resnets = [
            EfficientResNetBlock(
                in_channels=in_channels
                if i == 0 and not add_downsampling
                else out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                groups=groups,
                groups_out=groups,
                eps=eps,
                non_linearity=non_linearity,
                output_scale_factor=output_scale_factor,
                **factory_kwargs,
            )
            for i in range(num_layers)
        ]
        self.resnets = nn.ModuleList(resnets)

        self.attention = None
        if add_attention:
            self.attention = AttentionBlock(
                channels=out_channels,
                num_head_channels=out_channels // num_heads,
                num_groups=groups,
                rescale_output_factor=output_scale_factor,
                eps=eps,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,  # not implemented
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        states: Tuple[torch.Tensor, ...] = ()
        # TODO
        if context is not None:
            raise NotImplementedError

        # downsampling
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
            # states += (hidden_states,)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            states += (hidden_states,)

        if self.attention is not None:
            hidden_states = self.attention(hidden_states)

            states += (hidden_states,)

        return hidden_states, states


class EfficientUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_channels: int,
        out_channels: int,
        temb_channels: Optional[int] = None,
        stride: Optional[Tuple[int, int]] = None,
        num_layers: int = 1,
        eps: float = 1e-6,
        time_scale_shift: str = "default",
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        groups: int = 8,
        output_scale_factor: float = 1.0,
        add_upsampling: bool = True,
        add_attention: bool = False,
        num_heads: int = 8,
        **factory_kwargs,
    ) -> None:
        super(EfficientUpBlock, self).__init__()

        self.comb_embs = None

        resnets = []
        for i in range(num_layers):
            resnets.append(
                EfficientResNetBlock(
                    in_channels=in_channels
                    + (prev_channels if i == 0 else out_channels),
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=eps,
                    groups=groups,
                    time_embedding_norm=time_scale_shift,
                    non_linearity=non_linearity,
                    output_scale_factor=output_scale_factor,
                    **factory_kwargs,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.attention = None
        if add_attention:
            self.attention = AttentionBlock(
                channels=out_channels,
                num_head_channels=out_channels // num_heads,
                num_groups=groups,
                rescale_output_factor=output_scale_factor,
                eps=eps,
            )

        self.upsampler = None
        if add_upsampling:
            self.upsampler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if stride is not None else 1,
                    padding=1,
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,  # not implemented
    ) -> torch.Tensor:
        # TODO
        if context is not None:
            raise NotImplementedError

        for resnet in self.resnets:
            # residual
            res_states = res_hidden_states[-1]
            res_hidden_states = res_hidden_states[:-1]
            hidden_states = torch.cat([hidden_states, res_states], dim=1)
            # print(hidden_states.size(), end="->")
            hidden_states = resnet(hidden_states, temb)
            # print(hidden_states.size())

        if self.attention is not None:
            hidden_states = self.attention(hidden_states)

        # upsampling
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states
