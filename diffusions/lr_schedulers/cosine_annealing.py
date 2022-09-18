import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_warmup: int = 0,
        T_multi: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        assert T_0 >= 0 and isinstance(T_0, int)
        assert T_multi > 0 and isinstance(T_multi, int)

        self.T_0 = T_0
        self.T_warmup = T_warmup
        self.T_multi = T_multi
        self.eta_min = eta_min
        self.cos_anneal = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_multi, eta_min=eta_min, last_epoch=last_epoch
        )
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            print("warmup")
            return [
                self.eta_min
                + (base_lr - self.eta_min) * (self.last_epoch / self.T_warmup)
                for base_lr in self.base_lrs
            ]

        self.cos_anneal.step()
        print("cos stepped")
        return self.cos_anneal.get_lr()


if __name__ == "__main__":
    net = torch.nn.Linear(10, 10)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_0=50, T_warmup=5, T_multi=2)

    for step in range(100):
        scheduler.step()
        print(step, scheduler.get_last_lr())
