import pytest

from torch import Tensor, no_grad, nn

from phylearn.nn.training.losses import ScaledLoss


@pytest.fixture
def prediction() -> Tensor:
    return Tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )


@pytest.fixture
def target(prediction) -> Tensor:
    return prediction * 2


def test_scaled_loss(prediction, target) -> None:
    loss = ScaledLoss(nn.MSELoss())
    with no_grad():
        assert loss(prediction, target) == loss(
            prediction * 100, target * 100
        ), "Scaled loss does not depend on the range of values."
