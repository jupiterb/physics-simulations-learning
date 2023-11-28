from __future__ import annotations
from typing import Callable, Type, TypeVar

from torch import Tensor, nn, optim, no_grad, nan
from torch.utils.data import DataLoader, Dataset


OptimizerClass = Type[optim.Adam] | Type[optim.AdamW] | Type[optim.SGD]

ModelT = TypeVar("ModelT", bound=nn.Module)

_DEFAULT_EPOCHS = 10
_DEFAULT_LR = 0.001
_DEFAULT_BATCH_SIZE = 32


class Trainer:
    def __init__(self) -> None:
        self._get_optimizer: Callable[
            [nn.Module], optim.Optimizer
        ] = lambda net: optim.Adam(net.parameters(), lr=_DEFAULT_LR)

        self._loss_func: nn.Module = nn.MSELoss()

        self._train_dl: DataLoader | None = None
        self._test_dl: DataLoader | None = None

        self._epochs = _DEFAULT_EPOCHS

    def optimize_with(
        self, optimizer_cls: OptimizerClass, lr: float = _DEFAULT_LR
    ) -> Trainer:
        self._get_optimizer = lambda net: optimizer_cls(net.parameters(), lr=lr)
        return self

    def to_minimize(self, loss_func: nn.Module) -> Trainer:
        self._loss_func = loss_func
        return self

    def with_training_on(
        self, train_data: Dataset, batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> Trainer:
        self._train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return self

    def with_validation_on(
        self, test_data: Dataset, batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> Trainer:
        self._test_dl = DataLoader(test_data, batch_size=batch_size)
        return self

    def fit(self, net: ModelT, epochs: int = _DEFAULT_EPOCHS) -> ModelT:
        optimizer = self._get_optimizer(net)
        self._epochs = epochs
        for epoch in range(self._epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            net = self._run_epoch(net, optimizer)
        return net

    def _run_epoch(self, net: ModelT, optimizer: optim.Optimizer) -> ModelT:
        net = self._train(net, optimizer)
        test_loss = self._validate(net)
        print(f"    test loss: {test_loss}")
        return net

    def _train(self, net: ModelT, optimizer: optim.Optimizer) -> ModelT:
        net.train()

        if self._train_dl is None:
            return net

        n_batches = len(self._train_dl)

        for i, (X, Y) in enumerate(self._train_dl):
            loss = self._loss(net, X, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"    batch {i+1}/{n_batches}: train loss: {loss}")

        return net

    def _validate(self, net: nn.Module) -> float:
        net.eval()

        if self._test_dl is None:
            return nan

        with no_grad():
            losses = [self._loss(net, X, Y).item() for (X, Y) in self._test_dl]
            return sum(losses) / len(losses)

    def _loss(self, net: nn.Module, X: Tensor, Y: Tensor) -> Tensor:
        prediction = net(X)
        return self._loss_func(prediction, Y)
