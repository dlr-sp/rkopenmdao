"""Tests operator component for the flux around an internal boundary of split heat
equations."""

# pylint: disable=duplicate-code
import itertools
import pytest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.integration_control import IntegrationControl

from ..flux_integral_operator_component import (
    FluxIntegralOperatorComponent,
)


implicit_euler = ButcherTableau(np.array([[0.5]]), np.array([1.0]), np.array([0.5]))

gamma = (2.0 - np.sqrt(2.0)) / 2.0
two_stage_dirk = ButcherTableau(
    np.array(
        [
            [gamma, 0.0],
            [1 - gamma, gamma],
        ]
    ),
    np.array([1 - gamma, gamma]),
    np.array([gamma, 1.0]),
)

runge_kutta_four = ButcherTableau(
    np.array(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.167, 0.5, 0.0, 0.0],
            [-0.5, 0.5, 0.5, 0.0],
            [1.5, -1.5, 0.5, 0.5],
        ]
    ),
    np.array([1.5, -1.5, 0.5, 0.5]),
    np.array([0.5, 0.667, 0.5, 1.0]),
)


@pytest.mark.heatequ
@pytest.mark.heatequ_flux_integral_operator_comp
@pytest.mark.parametrize(
    "delta, shape, delta_t, butcher_tableau",
    itertools.product(
        [1e-1, 2e-2, 5e-1],
        [11, 21, 51],
        [1e-2, 1e-4, 1e-4],
        [implicit_euler, two_stage_dirk, runge_kutta_four],
    ),
)
def test_flux_component_partials(
    delta, shape, delta_t, butcher_tableau: ButcherTableau
):
    """Tests partials of component."""
    test_prob = om.Problem()

    integration_control = IntegrationControl(0.0, 1, 1, delta_t)

    flux_int_comp = FluxIntegralOperatorComponent(
        delta=delta, shape=shape, integration_control=integration_control
    )
    test_prob.model.add_subsystem("flux_comp", flux_int_comp)
    test_prob.setup()
    for stage in range(butcher_tableau.number_of_stages()):
        integration_control.butcher_diagonal_element = butcher_tableau.butcher_matrix[
            stage, stage
        ]
        test_prob.run_model()
        test_data = test_prob.check_partials(step=1e-1)
        assert_check_partials(test_data)
