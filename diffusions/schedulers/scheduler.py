from typing import Iterable, Optional

import numpy as np
import torch


class NoiseScheduler:
    def __init__(
        self,
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
        scheduler_type: str,
        scale_beta: bool,
        betas: Optional[Iterable],
        clip_sample: bool,
        **kwargs,
    ) -> None:
        if betas is not None:
            self.betas = np.asarray(betas)
        elif scheduler_type == "linear":
            if scale_beta:
                self.betas = np.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=np.float32,
                )
            else:
                self.betas = np.linspace(
                    beta_start, beta_end, num_train_timesteps, dtype=np.float32
                )
        elif scheduler_type == "cosine":
            """
            cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            """
            s = kwargs.get("s", 0.008)
            steps = num_train_timesteps + 1
            x = np.linspace(0, num_train_timesteps, steps, dtype=np.float32)
            alphas_cumprod = (
                np.cos(((x / num_train_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod.astype(np.float32)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = np.clip(beta, 0.0001, 0.9999)
        else:
            raise NotImplementedError

        self.num_train_steps = num_train_timesteps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        self.num_inference_steps = -1
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

        self.clip_sample = clip_sample

        self.convert_tensor()

    def convert_tensor(self) -> None:
        # numpy to tensor
        for key, value in vars(self).items():
            if isinstance(value, np.ndarray):
                setattr(self, key, torch.from_numpy(value))

    def clip(
        self,
        inputs: torch.Tensor,
        min: Optional[float],
        max: Optional[float],
    ) -> torch.Tensor:
        return torch.clamp(inputs, min, max)

    def dynamic_clip(self, inputs: torch.Tensor, p: float) -> torch.Tensor:
        """
        Dynamic threshold proposed at [Imagen](https://arxiv.org/pdf/2205.11487.pdf)

        TODO
        implemented on raw PyTorch
        """
        samples = inputs.detach().cpu().numpy()
        s = np.percentile(
            np.abs(samples),
            p,
            axis=tuple(
                range(1, inputs.ndim),
            ),
        ).astype(np.float32)

        s = np.maximum(s, 1.0)

        return torch.from_numpy(np.clip(samples, -s, s) / s).to(device=inputs.device)

    def _broadcast_to(
        self,
        values: torch.Tensor,
        broadcast: torch.Tensor,
    ) -> torch.Tensor:
        values = values.flatten()

        while len(values.shape) < len(broadcast.shape):
            values = values[..., None]
        values = values.to(broadcast.device)

        return values

    def add_noise(
        self,
        original_sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps,
    ) -> torch.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        # broadcast
        sqrt_alpha_prod = self._broadcast_to(sqrt_alpha_prod, original_sample)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = self._broadcast_to(
            sqrt_one_minus_alpha_prod, original_sample
        )

        noisy_samples = (
            sqrt_alpha_prod * original_sample + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.num_inference_steps = min(self.num_train_steps, num_inference_steps)
        self.timesteps = np.arange(
            0, self.num_train_steps, self.num_train_steps // self.num_inference_steps
        )[::-1].copy()

    def __len__(self) -> int:
        return self.num_train_steps
