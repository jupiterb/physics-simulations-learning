import pytest

from torch import ones, no_grad


@pytest.mark.parametrize(
    "model_fixture,input_fixture",
    [
        ("time_continuous_simulation_model", "conv_net_input"),
        ("time_discrete_simulation_model", "conv_net_input"),
        ("discovery_model", "conv_net_input"),
        ("static_model", "conv_net_input"),
    ],
)
def test_phy_model_output(model_fixture, input_fixture, request):
    model = request.getfixturevalue(model_fixture)
    input = request.getfixturevalue(input_fixture)
    time = ones([len(input)]) * 5

    with no_grad():
        output = model((input, time))
        assert output.shape == input.shape


@pytest.mark.parametrize(
    "model_fixture,input_fixture",
    [
        ("discovery_model", "conv_net_input"),
        ("static_model", "conv_net_input"),
    ],
)
def test_phy_params_model(
    model_fixture, input_fixture, diff_equation_num_params, request
):
    model = request.getfixturevalue(model_fixture)
    input = request.getfixturevalue(input_fixture)

    input_size = len(input)
    time = ones([input_size]) * 5

    with no_grad():
        params = model.pde_params(input, time)
        assert params.shape == (input_size, diff_equation_num_params)
