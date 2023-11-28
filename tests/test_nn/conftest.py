import pytest

from torch import Tensor, zeros

from typing import Sequence
from functools import reduce

from phylearn.domain import DiffEquation

from phylearn.nn.elementary import FCNet

from phylearn.nn.autoencoders import ConvAutoencoder, FCAutoencoder

from phylearn.nn.phyinformed import (
    PhySimulationModelNet,
    PhyDiscoveryModelNet,
    PhyStaticModel,
    TimeContinuousNet,
    TimeDiscreteNet,
)

from phylearn.nn.training.data.record import TensorRecord, RecordsGenerator
from phylearn.nn.training.data.sets import (
    RecordDataset,
    DatasetWithPhysics,
    ManyOffsetsRecordDataset,
)


@pytest.fixture
def fc_net_layer_sizes() -> Sequence[int]:
    return [10, 8, 4]


@pytest.fixture
def fc_net_input(fc_net_layer_sizes) -> Tensor:
    return zeros((1, fc_net_layer_sizes[0]))


@pytest.fixture
def conv_net_channels() -> Sequence[int]:
    return [3, 10, 10, 1]


@pytest.fixture
def conv_net_kernel_sizes() -> Sequence[int]:
    return [3, 3, 3]


@pytest.fixture
def conv_net_rescales() -> Sequence[int]:
    return [2, 2, 1]


@pytest.fixture
def conv_net_input(conv_net_channels) -> Tensor:
    input_shape = (1, conv_net_channels[0], 16, 16)
    return zeros(input_shape)


@pytest.fixture
def fc_autoencoder(fc_net_layer_sizes) -> FCAutoencoder:
    return FCAutoencoder(fc_net_layer_sizes)


@pytest.fixture
def conv_autoencoder_input_shape(conv_net_input) -> Sequence[int]:
    return conv_net_input.shape[1:]


@pytest.fixture
def conv_autoencoder(
    conv_autoencoder_input_shape,
    conv_net_channels,
    conv_net_kernel_sizes,
    conv_net_rescales,
) -> ConvAutoencoder:
    return ConvAutoencoder(
        conv_autoencoder_input_shape,
        conv_net_channels,
        conv_net_kernel_sizes,
        rescales=conv_net_rescales,
    )


@pytest.fixture
def conv_autoencoder_with_sub_autoencdoer(
    conv_autoencoder_input_shape,
    conv_net_channels,
    conv_net_kernel_sizes,
    conv_net_rescales,
    fc_net_layer_sizes,
) -> ConvAutoencoder:
    multiply = lambda a, b: a * b
    scale = reduce(multiply, conv_net_rescales)
    output_image_size = [size // scale for size in conv_autoencoder_input_shape[1:]]
    output_size = reduce(multiply, output_image_size) * conv_net_channels[-1]

    fc_layer_sizes = [output_size] + fc_net_layer_sizes

    return ConvAutoencoder(
        conv_autoencoder_input_shape,
        conv_net_channels,
        conv_net_kernel_sizes,
        rescales=conv_net_rescales,
        sub_autoencoder=FCAutoencoder(fc_layer_sizes),
    )


@pytest.fixture
def time_continuous_simulation_model(
    conv_autoencoder_with_sub_autoencdoer,
) -> PhySimulationModelNet:
    encoder = conv_autoencoder_with_sub_autoencdoer.encoder
    decoder = conv_autoencoder_with_sub_autoencdoer.decoder
    latent_size = conv_autoencoder_with_sub_autoencdoer.latent_shape[0]

    base = FCNet([latent_size + 1, 10, latent_size])  # + 1 for temporal input
    simulation_net = TimeContinuousNet(base)
    phy_simulation_net = PhySimulationModelNet(simulation_net, encoder, decoder)

    return phy_simulation_net


@pytest.fixture
def time_discrete_simulation_model(
    conv_autoencoder_with_sub_autoencdoer,
) -> PhySimulationModelNet:
    encoder = conv_autoencoder_with_sub_autoencdoer.encoder
    decoder = conv_autoencoder_with_sub_autoencdoer.decoder
    latent_size = conv_autoencoder_with_sub_autoencdoer.latent_shape[0]

    base = FCNet([latent_size, 10, latent_size])
    simulation_net = TimeDiscreteNet(base)
    phy_simulation_net = PhySimulationModelNet(simulation_net, encoder, decoder)

    return phy_simulation_net


@pytest.fixture
def diff_equation_num_params() -> int:
    return 2


@pytest.fixture
def diff_equation(diff_equation_num_params) -> DiffEquation[Tensor, Tensor]:
    class TestDiffEquation(DiffEquation[Tensor, Tensor]):
        def __call__(self, observation: Tensor, params: Tensor) -> Tensor:
            return observation + sum(params[:, diff_equation_num_params - 1])

    return TestDiffEquation()


@pytest.fixture
def discovery_model(
    conv_autoencoder_with_sub_autoencdoer, diff_equation, diff_equation_num_params
) -> PhyDiscoveryModelNet:
    encoder = conv_autoencoder_with_sub_autoencdoer.encoder
    latent_size = conv_autoencoder_with_sub_autoencdoer.latent_shape[0]

    base = FCNet([latent_size + 1, 10, diff_equation_num_params])
    discovery_net = TimeContinuousNet(base)
    phy_inverse_net = PhyDiscoveryModelNet(discovery_net, diff_equation, encoder)

    return phy_inverse_net


@pytest.fixture
def static_model(diff_equation, diff_equation_num_params) -> PhyStaticModel:
    params = [i for i in range(diff_equation_num_params)]
    return PhyStaticModel(diff_equation, params)


@pytest.fixture
def records_len() -> int:
    return 10


@pytest.fixture
def records(conv_net_input, static_model, records_len) -> Sequence[TensorRecord]:
    n_initial_states = 10

    initial_states = [conv_net_input[0] for _ in range(n_initial_states)]
    generate = RecordsGenerator(static_model)
    return generate(initial_states, records_len)


@pytest.fixture
def record_dataset(records) -> RecordDataset:
    return RecordDataset(records)


@pytest.fixture
def many_offsets_dataset(records) -> ManyOffsetsRecordDataset:
    return ManyOffsetsRecordDataset(records, [1, 2, 3])


@pytest.fixture
def physics_dataset(record_dataset, static_model) -> DatasetWithPhysics:
    return DatasetWithPhysics(record_dataset, static_model, 0.5)
