from torch import Tensor, nn

from typing import Sequence


class Convblock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        max_pooling: int | None,
    ) -> None:
        super(Convblock, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
        )
        if max_pooling is not None and max_pooling > 1:
            self._conv.append(nn.MaxPool2d(kernel_size=max_pooling, stride=max_pooling))

    def forward(self, x: Tensor) -> Tensor:
        return self._conv(x)


class TransposedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        upsampling: int | None,
    ) -> None:
        super(TransposedConvBlock, self).__init__()
        self._conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
        )
        if upsampling is not None and upsampling > 1:
            self._conv.append(nn.Upsample(scale_factor=upsampling, mode="nearest"))

    def forward(self, x: Tensor) -> Tensor:
        return self._conv(x)


class ConvNet(nn.Module):
    def __init__(
        self,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int] | None = None,
        paddings: Sequence[int] | None = None,
        rescales: Sequence[int] | None = None,
        transposed=False,
    ) -> None:
        super(ConvNet, self).__init__()

        sequence_of = lambda x: [1 for _ in kernel_sizes]
        strides = sequence_of(1) if strides is None else strides
        paddings = sequence_of(1) if paddings is None else paddings
        rescales_ = sequence_of(None) if rescales is None else rescales

        conv_block_class = TransposedConvBlock if transposed else Convblock
        self._conv = nn.Sequential()

        in_channels = channels[0]
        for out_channels, kernel_size, stride, padding, rescale in zip(
            channels[1:], kernel_sizes, strides, paddings, rescales_
        ):
            block = conv_block_class(
                in_channels, out_channels, kernel_size, stride, padding, rescale
            )
            self._conv.append(block)
            in_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        return self._conv(x)
