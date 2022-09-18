from typing import Optional

import einops
import torch
import torch.nn as nn


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss
    `Focal Frequency Loss for Image Reconstruction and Synthesis <https://arxiv.org/abs/2012.12821>`
    """

    def __init__(
        self,
        alpha: float = 1.0,
        patch_factor: int = 1,
        average_spectrum: bool = False,
        use_log: bool = False,
        use_batch_stat: bool = False,
    ) -> None:
        super(FocalFrequencyLoss, self).__init__()
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.average_spectrum = average_spectrum
        self.use_log = use_log
        self.use_batch_stat = use_batch_stat

    def _get_freq(self, x: torch.Tensor) -> torch.Tensor:
        patch = einops.rearrange(
            x,
            "b c (ph h) (pw w) -> b (ph pw) c h w",
            ph=self.patch_factor,
            pw=self.patch_factor,
        )

        freq = torch.fft.fft2(patch, norm="ortho")
        freq = torch.stack([freq.real, freq.imag], dim=-1)
        return freq

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_freq = self._get_freq(pred)
        target_freq = self._get_freq(target)

        if self.average_spectrum:
            pred_freq = torch.mean(pred_freq, dim=0, keepdim=True)
            target_freq = torch.mean(target_freq, dim=0, keepdim=True)

        if isinstance(matrix, torch.Tensor):
            weight_matrix = matrix.detach()
        else:
            weight_matrix = (pred_freq - target_freq) ** 2
            weight_matrix = (
                torch.sqrt(weight_matrix[..., 0] + weight_matrix[..., 1]) ** self.alpha
            )

            if self.use_log:
                weight_matrix = torch.log(weight_matrix + 1.0)

            if self.use_batch_stat:
                weight_matrix = weight_matrix / weight_matrix.max()
            else:
                weight_matrix = (
                    weight_matrix
                    / weight_matrix.max(dim=-1)
                    .values.max(dim=-1)
                    .values[:, :, :, None, None]
                )

            weight_matrix[torch.isnan(weight_matrix)] = 0.0
            weight_matrix = torch.clamp(weight_matrix, min=0.0, max=1.0).detach()

        dist = (pred_freq - target_freq) ** 2
        freq_dist = dist[..., 0] + dist[..., 1]

        loss = weight_matrix * freq_dist
        return loss.mean()


if __name__ == "__main__":
    criterion = FocalFrequencyLoss(patch_factor=2)
    inp = torch.randn(4, 3, 16, 16)
    target = torch.randn(4, 3, 16, 16)

    print(criterion(inp, target))
