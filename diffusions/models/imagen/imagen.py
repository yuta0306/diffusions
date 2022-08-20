from typing import Tuple

import torch
import torch.nn as nn
from diffusions.models.imagen.unet import UnconditionalEfficientUnet


class UnconditionalImagen(nn.Module):
    def __init__(
        self,
        unets: Tuple[UnconditionalEfficientUnet, ...],
        image_sizes: Tuple[int, ...],
        device: str = "cpu",
    ) -> None:
        super(UnconditionalImagen, self).__init__()

        self.unets = unets
        self.image_sizes = image_sizes
        self.device = device

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
