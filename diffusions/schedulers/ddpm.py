from typing import Dict, Iterable, Optional

import numpy as np
import torch


class DDPM:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        scheduler_type: str = "linear",
        scale_beta: bool = False,
        betas: Optional[Iterable] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
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
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.num_train_steps = num_train_timesteps

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.one = np.array(1.0)

        self.num_inference_steps = -1
        self.timesteps = np.arange(0, num_train_timesteps)[::-1].copy()

        self.variance_type = variance_type
        self.clip_sample = clip_sample

        # numpy to tensor
        for key, value in vars(self).items():
            if isinstance(value, np.ndarray):
                setattr(self, key, torch.from_numpy(value))

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.num_inference_steps = min(self.num_train_steps, num_inference_steps)
        self.timesteps = np.arange(
            0, self.num_train_steps, self.num_train_steps // self.num_inference_steps
        )[::-1].copy()

    def __len__(self):
        return self.num_train_steps

    def _broadcast_to(
        self, values: torch.Tensor, broadcast: torch.Tensor
    ) -> torch.Tensor:
        values = values.flatten()

        while len(values.shape) < len(broadcast.shape):
            values = values[..., None]
        values = values.to(broadcast.device)

        return values

    def _get_variance(
        self,
        t: int,
        predicted_variance: Optional[torch.Tensor] = None,
        variance_type: Optional[str] = None,
    ) -> torch.Tensor:
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]

        if variance_type is None:
            variance_type = self.variance_type

        if variance_type == "fixed_small":
            variance = torch.clamp(variance, min=1e-20)
        elif variance_type == "fixed_large":
            variance = self.betas[t]
        elif variance_type == "learned":
            if predicted_variance is None:
                raise ValueError
            return predicted_variance
        elif variance_type == "learned_range":
            if predicted_variance is None:
                raise ValueError
            min_log = variance
            max_log = self.betas[t]
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log
        else:
            raise NotImplementedError

        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        predict_eps: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        t = timestep

        if model_output.shape[1] == sample.shape[1] * 2:
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1
            )
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else self.one
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev

        if predict_eps:
            pred_original_sample = (
                sample - beta_prod_t**0.5 * model_output
            ) / alpha_prod_t**0.5
        else:
            pred_original_sample = model_output

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        pred_original_sample_coeff = (
            alpha_prod_t_prev**0.5 * self.betas[t]
        ) / beta_prod_t
        current_sample_coeff = self.alphas[t] ** 0.5 * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # add noise
        variance = 0
        if t > 0:
            noise = torch.randn(
                model_output.shape, layout=model_output.layout, generator=generator
            ).to(model_output.device)
            variance = (
                self._get_variance(t, predicted_variance=predicted_variance) ** 0.5
            ) * noise

        pred_prev_sample = pred_prev_sample + variance

        return {"prev_sample": pred_prev_sample}

    def add_noise(
        self, original_sample: torch.Tensor, noise: torch.Tensor, timesteps
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
