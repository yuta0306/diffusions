from typing import Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from diffusions.models.activation import Swish
from diffusions.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusions.models.imagen.blocks import (
    EfficientDownBlock,
    EfficientUNetMidBlock,
    EfficientUpBlock,
)


class UnconditionalEfficientUnet(nn.Module):

    time_proj: Union[GaussianFourierProjection, Timesteps]
    num_heads: Tuple[Union[int, None], ...]

    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        block_out_channels: Tuple[int, ...] = (32, 64, 128),
        layers_per_block: Union[int, Tuple[int, ...]] = 3,
        mid_block_scale_factor: float = 2 ** (-0.5),
        downsampling_padding: int = 1,
        non_linearity: Union[Type[Swish], Type[nn.SiLU], Type[nn.Mish]] = Swish,
        num_heads: Union[int, Tuple[Union[int, None], ...]] = 8,
        groups: int = 8,
        eps: float = 1e-6,
        res_scale: float = 2**-0.5,
        **factory_kwargs,
    ) -> None:
        super(UnconditionalEfficientUnet, self).__init__()

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

        # time embedding
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=block_out_channels[0],
                scale=16,
            )
            timestep_dim = block_out_channels[0] * 2
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                num_channels=block_out_channels[0],
                flip_sin_to_cos=flip_sin_to_cos,
                downscale_freq_shift=freq_shift,
            )
            timestep_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            channel=timestep_dim,
            dim=time_embed_dim,
        )

        if isinstance(layers_per_block, int):
            self.layers_per_block = (layers_per_block,) * len(block_out_channels)
        else:
            self.layers_per_block = layers_per_block

        if isinstance(num_heads, int):
            self.num_heads = (num_heads,) * len(block_out_channels)
        else:
            self.num_heads = num_heads

        down_blocks = []
        out_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            in_channel = out_channel
            out_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            n_heads = self.num_heads[i]
            add_attention = n_heads is not None
            if n_heads is None:  # to avoid mypy error
                n_heads = 1

            down_blocks.append(
                EfficientDownBlock(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    temb_channels=time_embed_dim,
                    stride=(2, 2),
                    num_layers=self.layers_per_block[i],
                    eps=eps,
                    non_linearity=non_linearity,
                    groups=groups,
                    add_attention=add_attention,
                    num_heads=n_heads,
                    add_downsampling=not is_final_block,
                    **factory_kwargs,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        self.mid_block = EfficientUNetMidBlock(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            eps=eps,
            non_linearity=non_linearity,
            groups=groups,
            add_attention=False,
            output_scale_factor=mid_block_scale_factor,
            **factory_kwargs,
        )

        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_layers_per_block = tuple(reversed(self.layers_per_block))
        reversed_num_heads = tuple(reversed(self.num_heads))
        for i in range(len(block_out_channels)):
            in_channel = reversed_block_out_channels[i]
            prev_channel = reversed_block_out_channels[i]
            out_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]
            # is_final_block = i == len(block_out_channels) - 1
            is_first_block = i == 0

            n_heads = reversed_num_heads[i]
            add_attention = n_heads is not None
            if n_heads is None:  # to avoid mypy error
                n_heads = 1

            up_blocks.append(
                EfficientUpBlock(
                    in_channels=in_channel,
                    prev_channels=prev_channel,
                    out_channels=out_channel,
                    stride=(1, 1),
                    num_layers=reversed_layers_per_block[i],
                    eps=eps,
                    non_linearity=non_linearity,
                    groups=groups,
                    add_attention=add_attention,
                    num_heads=n_heads,
                    add_upsampling=not is_first_block,
                    **factory_kwargs,
                )
            )
            self.up_blocks = nn.ModuleList(up_blocks)

            # self.dense = nn.Linear(block_out_channels[0], out_channels)
            self.dense = nn.Sequential(
                nn.GroupNorm(
                    num_groups=groups,
                    num_channels=block_out_channels[0],
                    eps=eps,
                ),
                nn.SiLU(),
                nn.Conv2d(
                    in_channels=block_out_channels[0],
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
            )

        self.res_scale = res_scale

    def forward(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> Dict[str, torch.Tensor]:

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
        res_samples: Tuple[torch.Tensor, ...] = ()
        for down_block in self.down_blocks:
            sample, res = down_block(hidden_states=sample, temb=emb)
            res_samples += res

        # mid
        sample = self.mid_block(sample, emb)

        # upsampling
        down_block_res_samples = tuple(
            res * self.res_scale for res in res_samples
        )  # scaling
        for i, up_block in enumerate(self.up_blocks, 1):
            layers = (
                len(up_block.resnets)
                + (1 if up_block.attention is not None else 0)
                # + (1 if up_block.upsampler is not None else 0)
            )
            res_samples = down_block_res_samples[-layers:]
            down_block_res_samples = down_block_res_samples[:-layers]
            sample = up_block(sample, res_samples, emb)

        # postprocess
        sample = self.dense(sample)

        return {"sample": sample}


if __name__ == "__main__":
    model = UnconditionalEfficientUnet(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        block_out_channels=(32, 64, 128, 256),
        layers_per_block=3,
        num_heads=(None, 1, 2, 4),
    )
    # print(model)

    inp = torch.randn((2, 3, 64, 64))
    print(inp.size())

    out = model(inp, 0)
    print(out["sample"].size())
    print(out["sample"])
