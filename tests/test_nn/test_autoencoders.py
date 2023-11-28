import pytest

from torch import no_grad


@pytest.mark.parametrize(
    "autoencoder_fixture,input_fixture",
    [
        ("fc_autoencoder", "fc_net_input"),
        ("conv_autoencoder", "conv_net_input"),
        ("conv_autoencoder_with_sub_autoencdoer", "conv_net_input"),
    ],
)
def test_autoencoder(autoencoder_fixture, input_fixture, request) -> None:
    autoencoder = request.getfixturevalue(autoencoder_fixture)
    input = request.getfixturevalue(input_fixture)

    assert autoencoder.input_shape == input.shape[1:]

    with no_grad():
        encoder_output = autoencoder.encoder.forward(input)
        assert encoder_output.shape[1:] == autoencoder.latent_shape

        decoder_output = autoencoder.decoder.forward(encoder_output)
        assert decoder_output.shape[1:] == autoencoder.input_shape

        assert decoder_output.shape == autoencoder.forward(input).shape
