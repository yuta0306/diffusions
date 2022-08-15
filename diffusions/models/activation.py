from typing import Optional, Union

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish
    f(x) = x * sigmoid(beta * x)

    when beta is 1, this function is the same as SiLU
    """

    beta: Union[nn.Parameter, float]

    def __init__(self, beta: Optional[float] = None) -> None:
        """Swish

        Parameters
        ----------
        beta : float, optional
            if beta is given, beta will be constant, otherwise beta will be trainable variable
        """
        super(Swish, self).__init__()

        self.beta = (
            nn.Parameter(torch.tensor(1.0), requires_grad=True)
            if beta is None
            else beta
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * nn.Sigmoid()(self.beta * x)
