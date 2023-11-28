from abc import ABC, abstractmethod
from torch import Tensor, nn


SpatialTemporal = tuple[Tensor, Tensor]


class SpatialTemporalNet(nn.Module, ABC):
    def __init__(self) -> None:
        super(SpatialTemporalNet, self).__init__()

    @abstractmethod
    def forward(self, X_T: SpatialTemporal) -> Tensor:
        pass
