from torch import Tensor, nn
from typing import Sequence


class ScaledLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, aim_scale: float = 1) -> None:
        super(ScaledLoss, self).__init__()
        self._loss = base_loss
        self._aim_scale = aim_scale

    def forward(self, Y_pred: Tensor, Y_target: Tensor) -> Tensor:
        maxes, _ = Y_target.view(Y_target.size(0), -1).max(dim=1)
        shape = -1, *tuple(1 for _ in Y_target.shape[1:])
        scales = (self._aim_scale / maxes).reshape(shape)

        Y_pred = Y_pred * scales
        Y_target = Y_target * scales

        return self._loss(Y_pred, Y_target)
