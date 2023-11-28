import pytest

from torch import no_grad
from functools import reduce

from phylearn.nn.elementary import ConvNet, FCNet


def test_fc_net_output(fc_net_layer_sizes, fc_net_input) -> None:
    fc_net = FCNet(fc_net_layer_sizes)
    with no_grad():
        fc_net_output_size = fc_net.forward(fc_net_input).shape[1]
        assert fc_net_output_size == fc_net_layer_sizes[-1]


@pytest.mark.parametrize(
    "transposed,total_scale",
    [
        (False, lambda scales: reduce(lambda a, b: a * b, scales)),
        (True, lambda scales: reduce(lambda a, b: a / b, reversed(scales))),
    ],
)
def test_conv_net_output(
    transposed,
    total_scale,
    conv_net_channels,
    conv_net_kernel_sizes,
    conv_net_rescales,
    conv_net_input,
) -> None:
    conv_net = ConvNet(
        conv_net_channels,
        conv_net_kernel_sizes,
        rescales=conv_net_rescales,
        transposed=transposed,
    )  # strides and paddings are default so they do not influance on the output shape
    with no_grad():
        conv_net_output = conv_net.forward(conv_net_input)
        assert conv_net_output.shape[1] == conv_net_channels[-1]

        size_change = total_scale(conv_net_rescales)
        assert conv_net_output.shape[2] == conv_net_input.shape[2] / size_change
        assert conv_net_output.shape[3] == conv_net_input.shape[3] / size_change
