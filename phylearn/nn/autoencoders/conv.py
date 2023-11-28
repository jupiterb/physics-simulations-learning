from phylearn.nn.autoencoders.abstract import AutoEncoder, Shape
from phylearn.nn.elementary import ConvNet

from typing import Sequence

from torch import nn, zeros


class ConvAutoencoder(AutoEncoder):
    def __init__(
        self,
        input_shape: Shape,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        rescales: Sequence[int] | None = None,
        sub_autoencoder: AutoEncoder | None = None,
        flatten_latent=True,
    ) -> None:
        encoder = ConvNet(channels, kernel_sizes, rescales=rescales)
        decoder = ConvNet(
            list(reversed(channels)), kernel_sizes, rescales=rescales, transposed=True
        )

        if flatten_latent:
            encode_output_shape = encoder(zeros((1, *input_shape))).shape[1:]
            encoder = nn.Sequential(encoder, nn.Flatten())
            decoder = nn.Sequential(
                nn.Unflatten(dim=1, unflattened_size=encode_output_shape), decoder
            )

        super().__init__(input_shape, encoder, decoder, sub_autoencoder)
