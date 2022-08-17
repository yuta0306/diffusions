from typing import Dict, Optional

import torch
import torch.nn as nn
from diffusions.models import UNet
from diffusions.schedulers import DDPM
from tqdm.auto import tqdm


class DDPMPipeline(nn.Module):
    def __init__(self, unet: UNet, scheduler: DDPM) -> None:
        super(DDPMPipeline, self).__init__()
        self.unet = unet
        self.scheduler = scheduler

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        device: Optional[str] = None,
        timesteps: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.unet.to(device)

        # gaussian noise
        images = torch.randn(
            (
                batch_size,
                self.unet.in_channels,
                self.unet.sample_size if self.unet.sample_size is not None else 128,
                self.unet.sample_size if self.unet.sample_size is not None else 128,
            ),
            generator=generator,
        )

        images = images.to(device)

        self.scheduler.set_timesteps(num_inference_steps=timesteps)

        for t in tqdm(self.scheduler.timesteps):
            model_output = self.unet(images, t)["sample"]

            images = self.scheduler.step(
                model_output=model_output, timestep=t, sample=images
            )["prev_sample"]

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu()

        return {"sample": images}

    @staticmethod
    def tensor_to_pil(self, images: torch.Tensor):
        raise NotImplementedError
