import copy
from typing import Optional

import torch
import torch.nn as nn


class EMAModel:
    """
    Exponential Moving Average
    """

    def __init__(
        self,
        model: nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
        max_value: float = 0.9999,
        device: Optional[torch.device] = None,
    ) -> None:
        self.averaged_model = copy.deepcopy(model).eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        if device is not None:
            self.averaged_model = self.averaged_model.to(device=device)

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int):
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: nn.Module):
        ema_state_dict = {}
        ema_params = self.averaged_model.state_dict()

        self.decay = self.get_decay(self.optimization_step)

        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue
            try:
                ema_param = ema_params[key]
            except KeyError:
                ema_param = (
                    param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
                )
                ema_params[key] = ema_param

            if not param.requires_grad:
                ema_params[key].copy_(param.to(dtype=ema_param.dtype).data)
                ema_param = ema_params[key]
            else:
                ema_param.mul_(self.decay)
                ema_param.add_(
                    param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay
                )

            ema_state_dict[key] = ema_param

        for name, buf in new_model.named_buffers():
            ema_state_dict[name] = buf

        self.averaged_model.load_state_dict(ema_state_dict, strict=False)
        self.optimization_step += 1
