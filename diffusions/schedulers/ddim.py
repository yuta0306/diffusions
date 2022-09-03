from typing import Dict, Iterable, Optional

import numpy as np
import torch
from diffusions.schedulers.scheduler import NoiseScheduler


class DDIM(NoiseScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        scheduler_type: str = "linear",
        scale_beta: bool = True,
        betas: Optional[Iterable] = None,
        clip_sample: bool = True,
        dynamic_threshold: bool = False,  # proposed at Imagen by Google
        alpha_to_one: bool = True,
    ) -> None:
        super().__init__(
            num_train_timesteps,
            beta_start,
            beta_end,
            scheduler_type,
            scale_beta,
            betas,
            clip_sample,
        )

        self.final_alpha_cumprod = (
            np.array(1.0, dtype=np.float32) if alpha_to_one else self.alphas_cumprod[0]
        )
        self.dynamic_threshold = dynamic_threshold

        self.convert_tensor()

    def _get_variance(self, t: int, prev_t: int) -> torch.Tensor:
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        p: float = 99.5,
        use_clipped_model_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        t = timestep
        prev_t = t - self.num_train_steps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5

        if self.clip_sample:
            if self.dynamic_threshold:
                assert 0 <= p <= 100, "0 <= p <= 100"
                pred_original_sample = self.dynamic_clip(pred_original_sample, p=p)
            else:
                pred_original_sample = self.clip(pred_original_sample, -1, 1)

        variance = self._get_variance(t, prev_t)
        std_dev_t = eta * variance**0.5

        if use_clipped_model_output:
            raise NotImplementedError

        pred_sample_direction = (
            1 - alpha_prod_t_prev - std_dev_t**2
        ) ** 0.5 * model_output

        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )

        if eta > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=generator).to(
                device=device
            )
            variance = self._get_variance(t, prev_t) ** 0.5 * eta * noise

            prev_sample = prev_sample + variance

        return {"prev_sample": prev_sample}
