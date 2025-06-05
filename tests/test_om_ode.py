"""
Tests the wrapper for turning an OpenMDAO problem into a discretized ODE.
"""

import openmdao.api as om
import numpy as np
import pytest

from rkopenmdao.discretized_ode.openmdao_ode import OpenMDAOODE
from rkopenmdao.integration_control import StepTerminationIntegrationControl

from .test_components import TestComp1

DELTA_T = 0.1
test_integration_control = StepTerminationIntegrationControl(DELTA_T, 1, 0.0)
test_problem = om.Problem()
test_problem.model.add_subsystem(
    "comp", TestComp1(integration_control=test_integration_control), promotes=["*"]
)
test_problem.setup()
test_problem.final_setup()
test_time_integration_quantities = ["x"]
test_independent_input_quantities = ["b"]
ODE = OpenMDAOODE(
    test_problem,
    test_integration_control,
    test_time_integration_quantities,
    test_independent_input_quantities,
)


def test_ode_attributes():
    """
    Tests that all public attributes of the OpenMDAO ODE are accessible.
    """
    assert hasattr(ODE, "time_integration_metadata")


def test_compute_update():
    """
    Tests the compute_update method of the OpenMDAO ODE.
    """
    update, _, _ = ODE.compute_update(
        np.ones(1), np.ones(1), np.ones(1), 0.0, DELTA_T, 0.0
    )
    assert update == pytest.approx(1 + DELTA_T)


def test_compute_update_derivative():
    """
    Tests the compute_update_derivative method of the OpenMDAO ODE.
    """
    _, _, _ = ODE.compute_update(np.ones(1), np.ones(1), np.ones(1), 0.0, DELTA_T, 0.0)

    update_pert, _, _ = ODE.compute_update_derivative(
        np.ones(1), np.ones(1), np.ones(1), 0.0, DELTA_T, 0.0
    )

    assert update_pert == pytest.approx(2 + 2 * DELTA_T)


def test_compute_update_adjoint_derivative():
    """
    Tests the compute_update_adjoint_derivative method of the OpenMDAO ODE.
    """
    _, _, _ = ODE.compute_update(np.ones(1), np.ones(1), np.ones(1), 0.0, DELTA_T, 0.0)

    step_input_pert, stage_input_pert, independent_input_pert, _ = (
        ODE.compute_update_adjoint_derivative(
            np.ones(1), np.zeros(1), np.zeros(1), DELTA_T, 0.0
        )
    )

    assert step_input_pert == pytest.approx(1.0)
    assert stage_input_pert == pytest.approx(DELTA_T)
    assert independent_input_pert == pytest.approx(1 + DELTA_T)


def test_reimport():
    """
    Tests that exporting and reimporting a cached linearization state provides the
    correct results.
    """
    _, _, _ = ODE.compute_update(np.ones(1), np.ones(1), np.ones(1), 0.0, DELTA_T, 0.0)
    cache = ODE.export_linearization()
    update_pert, _, _ = ODE.compute_update_derivative(
        np.ones(1), np.ones(1), np.ones(1), 0.0, DELTA_T, 0.0
    )

    ODE.compute_update(np.zeros(1), np.zeros(1), np.zeros(1), 0.0, DELTA_T, 0.0)

    ODE.import_linearization(cache)
    update_pert_new, _, _ = ODE.compute_update_derivative(
        np.ones(1), np.ones(1), np.ones(1), 0.0, DELTA_T, 0.0
    )
    assert update_pert_new == update_pert
