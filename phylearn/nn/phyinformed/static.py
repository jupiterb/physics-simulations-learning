from phylearn.domain import DiffEquation
from phylearn.nn.phyinformed.params import PhyParamsModelNet

from torch import Tensor, nn, stack
from typing import Sequence


class _ParamsLike(nn.Module):
    def __init__(self, params: Sequence[float]) -> None:
        super(_ParamsLike, self).__init__()
        self._params = params

    def forward(self, observation: Tensor) -> Tensor:
        return stack([Tensor(self._params)] * observation.shape[0])


class PhyStaticModel(PhyParamsModelNet):
    def __init__(
        self, equation: DiffEquation[Tensor, Tensor], params: Sequence[float]
    ) -> None:
        super().__init__(equation)
        self._get_params = _ParamsLike(params)

    def pde_params(self, observation: Tensor, time: Tensor) -> Tensor:
        return self._get_params(observation)
