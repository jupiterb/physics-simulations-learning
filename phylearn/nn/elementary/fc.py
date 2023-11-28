from torch import Tensor, nn

from typing import Sequence


class FCNet(nn.Module):
    def __init__(self, layer_sizes: Sequence[int]) -> None:
        super(FCNet, self).__init__()
        self._fc = nn.Sequential()
        input_size = layer_sizes[0]
        for output_size in layer_sizes[1:]:
            self._fc.append(nn.Linear(input_size, output_size))
            self._fc.append(nn.ReLU())
            input_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return self._fc(x)
