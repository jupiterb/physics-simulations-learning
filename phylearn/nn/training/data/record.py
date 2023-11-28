from phylearn.domain import Record
from phylearn.nn.phyinformed.abstract import SpatialTemporalNet

from torch import Tensor, cat, no_grad
from typing import Sequence


class TensorRecord(Record[Tensor]):
    def __init__(self, observations: Tensor, time_steps: Tensor) -> None:
        assert len(observations) == len(time_steps)
        assert len(observations.shape) >= 1
        assert len(time_steps.shape) == 1

        self._observations = observations
        self._time_steps = time_steps

    @property
    def observations(self) -> Sequence[Tensor]:
        return self._observations  # type: ignore - observations Tensor is sequence of Tensors as well

    @property
    def time_steps(self) -> Sequence[float]:
        return self._time_steps  # type: ignore - time_steps Tensor is sequence of floats as well

    def __len__(self) -> int:
        return len(self._time_steps)


class RecordsGenerator:
    def __init__(self, physics: SpatialTemporalNet) -> None:
        self._physics = physics

    def __call__(
        self, initial_states: Sequence[Tensor], record_len: int, time_step: int = 1
    ) -> Sequence[TensorRecord]:
        return [
            self._eval_record(initial_state, record_len, time_step)
            for initial_state in initial_states
        ]

    def _eval_record(
        self, initial_state: Tensor, record_len: int, time_step: int
    ) -> TensorRecord:
        observation = initial_state.unsqueeze(0)
        time_step_tensor = Tensor([time_step])
        t = 0

        observations = observation
        time_steps = Tensor([t])

        with no_grad():
            for _ in range(record_len - 1):
                t += time_step
                observation = self._physics((observation, time_step_tensor))
                observations = cat((observations, observation))
                time_steps = cat((time_steps, Tensor([t])))

        return TensorRecord(observations, time_steps)
