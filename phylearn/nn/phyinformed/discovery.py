from phylearn.domain import DiffEquation
from phylearn.nn.phyinformed.params import PhyParamsModelNet
from phylearn.nn.phyinformed.utils import TimeContinuousNet

from torch import Tensor, nn
from typing import Sequence


class PhyDiscoveryModelNet(PhyParamsModelNet):
    def __init__(
        self,
        discovery_net: TimeContinuousNet,
        equation: DiffEquation[Tensor, Tensor],
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__(equation, encoder)
        self._discover = discovery_net

    def pde_params(self, observation: Tensor, time: Tensor) -> Sequence[Tensor]:
        X_encoded = self._encode(observation)
        return self._discover((X_encoded, time))
