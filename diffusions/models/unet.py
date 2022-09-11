from typing import Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from diffusions.models.activation import Swish
from diffusions.models.blocks import (
    AttnDownBlock,
    AttnUpBlock,
    DownBlock,
    UNetMidBlock,
    UpBlock,
)
from diffusions.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)


class UNet(nn.Module):

    conv_in: nn.Conv2d
    time_proj: Union[GaussianFourierProjection, Timesteps]
    time_embedding: TimestepEmbedding
    down_blocks: nn.ModuleList

    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[Union[Type[DownBlock], Type[AttnDownBlock]], ...] = (
            DownBlock,
            AttnDownBlock,
            AttnDownBlock,
            AttnDownBlock,
        ),
        up_block_types: Tuple[Union[Type[UpBlock], Type[AttnUpBlock]], ...] = (
            UpBlock,
            AttnUpBlock,
            AttnUpBlock,
            AttnUpBlock,
        ),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: Union[int, Tuple[int, ...]] = 2,
        mid_block_scale_factor: int = 1,
        downsample_padding: int = 1,
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        head_dim: int = 8,
        groups: int = 32,
        eps: float = 1e-6,
        dropout: float = 0.0,
        memory_efficient: bool = False,
        **factory_kwargs,
    ) -> None:
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=block_out_channels[0],
            kernel_size=3,
            padding=(1, 1),
        )

        # time embeddings
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=block_out_channels[0], scale=16
            )
            timestep_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                num_channels=block_out_channels[0],
                flip_sin_to_cos=flip_sin_to_cos,
                downscale_freq_shift=freq_shift,
            )
            timestep_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            channel=timestep_dim, dim=time_embed_dim
        )

        if isinstance(layers_per_block, int):
            self.layers_per_block = (layers_per_block,) * len(block_out_channels)
        else:
            self.layers_per_block = layers_per_block

        # downsampling
        down_blocks = []
        out_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            in_channel = out_channel
            out_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_blocks.append(
                down_block_type(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    temb_channels=time_embed_dim,
                    num_layers=self.layers_per_block[i],
                    eps=eps,
                    dropout=dropout,
                    non_linearity=non_linearity,
                    num_head_channels=head_dim,
                    add_downsample=not is_final_block,
                    downsample_padding=downsample_padding,
                    memory_efficient=memory_efficient,
                    **factory_kwargs,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid layer
        self.mid_block = UNetMidBlock(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            eps=eps,
            dropout=dropout,
            non_linearity=non_linearity,
            output_scale_factor=mid_block_scale_factor,
            time_scale_shift="default",
            num_head_channels=head_dim,
            groups=groups,
            **factory_kwargs,
        )

        # upsampling
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_layers_per_block = tuple(reversed(self.layers_per_block))
        out_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_channel = out_channel
            out_channel = reversed_block_out_channels[i]
            in_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            up_blocks.append(
                up_block_type(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    prev_channels=prev_channel,
                    temb_channels=time_embed_dim,
                    num_layers=reversed_layers_per_block[i] + 1,
                    eps=eps,
                    dropout=dropout,
                    non_linearity=non_linearity,
                    num_head_channels=head_dim,
                    add_upsample=not (i == len(block_out_channels) - 1),
                    memory_efficient=memory_efficient,
                    **factory_kwargs,
                )
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        # final layer
        self.conv_norm_out = nn.GroupNorm(
            num_groups=groups,
            num_channels=block_out_channels[0],
            eps=eps,
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.center_input_sample = center_input_sample

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
    ) -> Dict[str, torch.Tensor]:

        if self.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        if isinstance(timestep, torch.Tensor):
            timesteps = timestep
        else:
            timesteps = torch.tensor([timestep], dtype=torch.long, device=sample.device)

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        # preprocess
        sample = self.conv_in(sample)

        # downsampling
        res_samples: Tuple[torch.Tensor, ...] = (sample,)
        for down_block in self.down_blocks:
            sample, res = down_block(hidden_states=sample, temb=emb)
            res_samples += res

        # mid
        sample = self.mid_block(sample, emb)

        # upsampling
        down_block_res_samples = res_samples
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(up_block.resnets)]
            sample = up_block(
                sample,
                res_samples,
                emb,
            )

        # postprocessing
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if isinstance(self.time_embedding, GaussianFourierProjection):
            timesteps = timesteps.reshape(
                (sample.shape[0], *([1] * len(sample.shape[1:])))
            )

        return {"sample": sample}


if __name__ == "__main__":
    inp = torch.randn(2, 3, 128, 128)
    timesteps = torch.randint(
        0,
        1000,
        (inp.size(0),),
        device=inp.device,
    ).long()
    model = UNet(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=3,
        block_out_channels=(32, 64, 128, 256),
        down_block_types=(
            DownBlock,
            AttnDownBlock,
            AttnDownBlock,
            AttnDownBlock,
        ),
        up_block_types=(
            AttnUpBlock,
            AttnUpBlock,
            AttnUpBlock,
            UpBlock,
        ),
        memory_efficient=True,
    )
    print(model)

    out = model(inp, timestep=timesteps)
    print(out)
