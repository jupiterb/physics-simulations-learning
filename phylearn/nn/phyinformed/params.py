from abc import ABC

from phylearn.domain import PhyParamsModel, DiffEquation
from phylearn.nn.phyinformed.abstract import SpatialTemporal, SpatialTemporalNet

from torch import Tensor, nn, no_grad


class _DiffEquationNetModule(nn.Module):
    def __init__(
        self, equation: DiffEquation[Tensor, Tensor], time_step: float = 1
    ) -> None:
        super(_DiffEquationNetModule, self).__init__()
        self._equation = equation
        self._time_step = time_step

    def forward(self, X_T: SpatialTemporal, params: Tensor) -> Tensor:
        X, T = X_T

        X_prim = X.clone()
        steps = int(T.max().item() / self._time_step)

        for i in range(steps):
            mask: Tensor = T > i * self._time_step
            mask = mask.squeeze() if mask.dim() > 1 else mask
            X_prim[mask] += self._equation(X_prim[mask], params[mask])  # type: ignore

        return X_prim


class PhyParamsModelNet(
    SpatialTemporalNet, PhyParamsModel[Tensor, Tensor, Tensor], ABC
):
    def __init__(
        self,
        equation: DiffEquation[Tensor, Tensor],
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self._equation = _DiffEquationNetModule(equation)
        self._encoder = nn.Identity() if encoder is None else encoder

    def _encode(self, X: Tensor) -> Tensor:  # type: ignore
        with no_grad():
            return self._encoder(X)

    def forward(self, X_T: SpatialTemporal) -> Tensor:
        X, T = X_T
        params = self.pde_params(X, T)
        return self._equation((X, T), params)

    def model(self, observation: Tensor, time: Tensor) -> Tensor:
        return self.forward((observation, time))
