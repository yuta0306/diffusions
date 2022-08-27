from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusions.models.imagen.unet import UnconditionalEfficientUnet


class UnconditionalImagen(nn.Module):
    def __init__(
        self,
        unets: Tuple[UnconditionalEfficientUnet, ...],
        sample_sizes: Tuple[int, ...],
        device: str = "cpu",
    ) -> None:
        super(UnconditionalImagen, self).__init__()

        self.unets = unets
        self.sample_sizes = sample_sizes
        self.device = device

    def __repr__(self) -> str:
        return nn.ModuleList(self.unets).__repr__()

    def to(self, device):
        self.device = device
        return self

    def _set_device(self, index: int):
        self.unets = tuple(
            [
                unet.to(self.device) if i == index else unet.cpu()
                for i, unet in enumerate(self.unets)
            ]
        )
        return self

    def resize_to(self, sample: torch.Tensor, index: int) -> torch.Tensor:
        return F.interpolate(sample, self.sample_sizes[index], mode="nearest")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        index: int,
    ) -> Dict[str, torch.Tensor]:
        self._set_device(index)
        unet = self.unets[index]

        out = unet(sample=sample, timestep=timestep)

        return out


if __name__ == "__main__":
    from diffusions.models.imagen import UnconditionalEfficientUnet

    unet = UnconditionalEfficientUnet(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        block_out_channels=(32, 64, 128, 256),
        layers_per_block=3,
        num_heads=(None, 1, 2, 4),
    )

    sr_unet = UnconditionalEfficientUnet(
        sample_size=128,
        in_channels=3,
        out_channels=3,
        block_out_channels=(32, 64, 128),
        layers_per_block=(2, 4, 8),
        num_heads=(None, None, 4),
    )

    imagen = UnconditionalImagen(unets=(unet, sr_unet), sample_sizes=(64, 128))

    sample = torch.randn((2, 3, 64, 64))
    sample_denoise = imagen(sample, 0, 0)["sample"]
    print(sample_denoise.size())
    sample_denoise = imagen.resize_to(sample_denoise, 1)
    print("resized:", sample_denoise.size())
    noise = torch.randn((2, 3, 128, 128))
    sample_denoise = sample_denoise + noise
    sample_sr = imagen(sample_denoise, 0, 1)["sample"]
    print(sample_sr.size())
