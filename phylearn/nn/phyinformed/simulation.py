from phylearn.domain import PhySimulationModel
from phylearn.nn.phyinformed.abstract import SpatialTemporal, SpatialTemporalNet
from phylearn.nn.phyinformed.utils import TimeContinuousNet, TimeDiscreteNet

from torch import Tensor, nn, no_grad


class PhySimulationModelNet(SpatialTemporalNet, PhySimulationModel[Tensor, Tensor]):
    def __init__(
        self,
        simulation_net: TimeContinuousNet | TimeDiscreteNet,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self._simulate = simulation_net
        self._encoder = nn.Identity() if encoder is None else encoder
        self._decoder = nn.Identity() if decoder is None else decoder

    def _encode(self, X: Tensor) -> Tensor:  # type: ignore
        # with no_grad():
        return self._encoder(X)

    def _decode(self, X: Tensor) -> Tensor:
        # with no_grad():
        return self._decoder(X)

    def forward(self, X_T: SpatialTemporal) -> Tensor:
        X, T = X_T
        X_encoded = self._encode(X)
        X_simulated = self._simulate((X_encoded, T))
        return self._decode(X_simulated)

    def model(self, observation: Tensor, time: Tensor) -> Tensor:
        return self.forward((observation, time))
