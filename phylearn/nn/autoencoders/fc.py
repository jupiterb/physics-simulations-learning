from phylearn.nn.autoencoders.abstract import AutoEncoder
from phylearn.nn.elementary import FCNet

from typing import Sequence


class FCAutoencoder(AutoEncoder):
    def __init__(
        self,
        layer_sizes: Sequence[int],
        sub_autoencoder: AutoEncoder | None = None,
    ) -> None:
        input_shape = (layer_sizes[0],)
        encoder = FCNet(layer_sizes)
        decoder = FCNet(list(reversed(layer_sizes)))
        super().__init__(input_shape, encoder, decoder, sub_autoencoder)
