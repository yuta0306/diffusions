from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def resize_image_to(
    image: torch.Tensor,
    target_size: Union[int, Tuple[int, int]],
    min: Optional[float] = None,
    max: Optional[float] = None,
) -> torch.Tensor:
    resized = F.interpolate(image, target_size, mode="nearest")
    resized = torch.clamp(resized, min=min, max=max)

    return resized
