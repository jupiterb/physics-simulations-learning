from phylearn.nn.phyinformed.abstract import SpatialTemporal, SpatialTemporalNet

from torch import Tensor, nn, cat, empty, clone


class TimeContinuousNet(SpatialTemporalNet):
    def __init__(
        self, base: nn.Module, feature_extractor: nn.Module | None = None
    ) -> None:
        super().__init__()
        self._eval = base
        self._extract = (
            nn.Identity() if feature_extractor is None else feature_extractor
        )

    def forward(self, X_T: SpatialTemporal) -> Tensor:
        X, T = X_T
        X = self._extract(X)

        T_expanded = empty((len(T), 1, *X.shape[2:]))

        for i, t in enumerate(T):
            T_expanded[i] = t

        XT = cat((X, T_expanded), dim=1)
        return self._eval(XT)


class TimeDiscreteNet(SpatialTemporalNet):
    def __init__(
        self,
        base: nn.Module,
        time_step: float = 1,
        feature_extractor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self._eval = base
        self._time_step = time_step
        self._extract = (
            nn.Identity() if feature_extractor is None else feature_extractor
        )

    def forward(self, X_T: SpatialTemporal) -> Tensor:
        X, T = X_T
        X = self._extract(X)

        X_prim = clone(X)
        steps = int(T.max().item() / self._time_step)

        for i in range(steps):
            mask: Tensor = T > i * self._time_step
            mask = mask.squeeze() if mask.dim() > 1 else mask
            X_prim[mask] += self._eval(X_prim[mask])

        return X_prim
