from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Sequence, Sized


ObservationT = TypeVar("ObservationT")


class Record(Generic[ObservationT], Sized, ABC):
    """
    This class represents an ordered series of observations
    along with the time step at which they were recorded.
    """

    @property
    @abstractmethod
    def observations(self) -> Sequence[ObservationT]:
        pass

    @property
    @abstractmethod
    def time_steps(self) -> Sequence[float]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


EquationParamsT = TypeVar("EquationParamsT")


class DiffEquation(Generic[ObservationT, EquationParamsT], ABC):
    """
    This class represents a differential equation with respect to time,
    a callable, for a given observation and parameters
    returns a new observation.
    """

    @abstractmethod
    def __call__(
        self, observation: ObservationT, params: EquationParamsT
    ) -> ObservationT:
        pass


TimeT = TypeVar("TimeT")


class PhySimulationModel(Generic[ObservationT, TimeT], ABC):
    """
    This class represents a model of a physical phenomenon,
    for given observation it simulates
    a physical phenomenon after given time.
    """

    @abstractmethod
    def model(self, observation: ObservationT, time: TimeT) -> ObservationT:
        pass


class PhyParamsModel(Generic[ObservationT, EquationParamsT, TimeT], ABC):
    """
    This class represents a model of a physical phenomenon,
    for given observation it returns discovered parameters
    of differential equation which model physical phenomenon.
    """

    @abstractmethod
    def pde_params(self, observation: ObservationT, time: TimeT) -> EquationParamsT:
        pass

    @abstractmethod
    def model(self, observation: ObservationT, time: TimeT) -> ObservationT:
        pass
