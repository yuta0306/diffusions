from typing import Callable, Dict, List, Optional, Union

import einops
import numpy as np
import torch
import torch.nn as nn
from diffusions.models import UNet
from diffusions.schedulers.ddim import DDIM
from PIL import Image
from tqdm.auto import tqdm


class DDIMPipeline(nn.Module):
    def __init__(self, unet: UNet, scheduler: DDIM) -> None:
        super(DDIMPipeline, self).__init__()
        self.unet = unet
        self.scheduler = scheduler

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        device: Optional[str] = None,
        eta: float = 0.0,
        timesteps: int = 50,
        p: float = 99.5,
        return_type: str = "pt",
        apply_func: Optional[Callable] = None,
        *,
        out: Optional[list] = None,
    ) -> Dict[str, Union[torch.Tensor, np.ndarray, Image.Image]]:
        assert return_type in ("pt", "np", "pil")

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

        for t in tqdm(self.scheduler.timesteps, leave=False):
            model_output = self.unet(images, t)["sample"]

            images = self.scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=images,
                eta=eta,
                p=p,
            )["prev_sample"]

            if out is not None:
                processed = images.clone()
                if apply_func is not None:
                    processed = apply_func(processed)

                if return_type == "np":
                    out.append(processed.cpu().numpy())
                elif return_type == "pil":
                    out.append(self.tensor_to_pil(processed))
                else:
                    out.append(processed)

        # images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu()

        if return_type == "pt":
            pass
        elif return_type == "np":
            images = images.numpy()
        elif return_type == "pil":
            pass

        return {"sample": images}

    def tensor_to_pil(
        self, images: torch.Tensor
    ) -> Union[Image.Image, List[Image.Image]]:
        if images.ndim == 4:
            images_processed = (
                einops.rearrange((images * 255).round(), "b c h w -> b h w c")
                .cpu()
                .numpy()
                .astype("uint8")
            )
            outs = []
            for i in range(images_processed.shape[0]):
                outs.append(Image.fromarray(images_processed[i], mode="RGB"))

        elif images.ndim == 3:
            images_processed = (
                einops.rearrange((images * 255).round(), "c h w -> h w c")
                .cpu()
                .numpy()
                .astype("uint8")
            )
            outs = Image.fromarray(images_processed, mode="RGB")

        else:
            raise ValueError

        return outs
