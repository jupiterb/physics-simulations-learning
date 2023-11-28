import pytest

from torch import Tensor


@pytest.mark.parametrize(
    "dataset_fixture",
    ["record_dataset", "many_offsets_dataset", "physics_dataset"],
)
def test_dataset(dataset_fixture, request):
    dataset = request.getfixturevalue(dataset_fixture)

    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]

    for i in range(len(dataset)):
        X_T, Y = dataset[i]
        assert len(X_T) == 2  # spatial and temporal

        X, T = X_T
        assert T.shape[0] == 1 and len(T.shape) == 1

        assert X.shape == Y.shape
