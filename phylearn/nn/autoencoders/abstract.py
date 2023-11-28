from __future__ import annotations
from typing import Sequence

from torch import Tensor, nn, zeros


Shape = Sequence[int]


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Shape,
        encoder: nn.Module,
        decoder: nn.Module,
        sub_autoencoder: AutoEncoder | None = None,
    ) -> None:
        super(AutoEncoder, self).__init__()
        self._input_shape = input_shape
        self._encoder = (
            encoder
            if sub_autoencoder is None
            else nn.Sequential(encoder, sub_autoencoder.encoder)
        )
        self._decoder = (
            decoder
            if sub_autoencoder is None
            else nn.Sequential(sub_autoencoder.decoder, decoder)
        )

    def forward(self, x: Tensor) -> Tensor:
        latent = self._encoder(x)
        return self._decoder(latent)

    @property
    def input_shape(self) -> Shape:
        return self._input_shape

    @property
    def latent_shape(self) -> Shape:
        x = zeros((1, *self._input_shape))
        latent = self._encoder(x)
        return latent.shape[1:]

    @property
    def encoder(self) -> nn.Module:
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        return self._decoder
