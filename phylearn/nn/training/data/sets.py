from abc import ABC, abstractmethod

from phylearn.domain import Record
from phylearn.nn.phyinformed.abstract import SpatialTemporal, SpatialTemporalNet

from torch import Tensor, no_grad, cat
from torch.utils.data import Dataset

from typing import Sequence, Sized


class SpatialTemporalDataset(Dataset, Sized, ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index) -> tuple[SpatialTemporal, Tensor]:
        pass


class RecordDataset(SpatialTemporalDataset):
    def __init__(self, records: Sequence[Record[Tensor]], offset: int = 1) -> None:
        self._records = records
        self._offset = offset

    def __len__(self) -> int:
        return sum(len(record) - self._offset for record in self._records)

    def __getitem__(self, index: int) -> tuple[SpatialTemporal, Tensor]:
        for record in self._records:
            n_items = len(record) - self._offset
            if index < n_items:
                T = Tensor(
                    [record.time_steps[index + self._offset] - record.time_steps[index]]
                )
                X = record.observations[index]
                Y = record.observations[index + self._offset]
                return (X, T), Y
            else:
                index -= n_items
        raise IndexError(
            f"Index out of range, should be lower than size of dataset ({len(self)})."
        )


class ManyOffsetsRecordDataset(SpatialTemporalDataset):
    def __init__(
        self, records: Sequence[Record[Tensor]], offsets: Sequence[int]
    ) -> None:
        self._sub_datasets = [RecordDataset(records, offset) for offset in offsets]

    def __len__(self) -> int:
        return sum([len(dataset) for dataset in self._sub_datasets])

    def __getitem__(self, index) -> tuple[SpatialTemporal, Tensor]:
        for dataset in self._sub_datasets:
            n_items = len(dataset)
            if index < n_items:
                return dataset[index]
            else:
                index -= n_items
        raise IndexError(
            f"Index out of range, should be lower than size of dataset ({len(self)})."
        )


class DatasetWithPhysics(SpatialTemporalDataset):
    def __init__(
        self,
        dataset: SpatialTemporalDataset,
        physics: SpatialTemporalNet,
        physics_weight: float,
    ) -> None:
        self._sub_dataset = dataset
        self._physics_w = physics_weight

        Y_phys = []

        with no_grad():
            for i in range(len(self._sub_dataset)):
                (X, T), _ = self._sub_dataset[i]
                Y_phy = physics((X.unsqueeze(0), T))
                Y_phys.append(Y_phy)

        self._Y_phy = cat(Y_phys)

    def __len__(self) -> int:
        return len(self._sub_dataset)

    def __getitem__(self, index: int) -> tuple[SpatialTemporal, Tensor]:
        (X, T), Y = self._sub_dataset[index]

        diff = self._Y_phy[index]
        Y_phy = X + diff

        Y_mean = (1 - self._physics_w) * Y + Y_phy * self._physics_w

        return (X, T), Y_mean
