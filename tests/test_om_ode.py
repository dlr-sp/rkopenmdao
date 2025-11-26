"""
Tests the wrapper for turning an OpenMDAO problem into a discretized ODE.
"""

import openmdao.api as om
import numpy as np
import pytest

from rkopenmdao.discretized_ode.discretized_ode import (
    DiscretizedODEInputState,
    DiscretizedODEResultState,
)
from rkopenmdao.discretized_ode.openmdao_ode import OpenMDAOODE
from rkopenmdao.integration_control import StepTerminationIntegrationControl

from .test_components import TestComp1
from .distributed_variables_test import Test1Component1

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
    update = ODE.compute_update(
        DiscretizedODEInputState(np.ones(1), np.ones(1), np.ones(1), 0.0), DELTA_T, 0.0
    ).stage_update
    assert update == pytest.approx(1 + DELTA_T)


def test_compute_update_derivative():
    """
    Tests the compute_update_derivative method of the OpenMDAO ODE.
    """
    ODE.compute_update(
        DiscretizedODEInputState(np.ones(1), np.ones(1), np.ones(1), 0.0), DELTA_T, 0.0
    )

    update_pert = ODE.compute_update_derivative(
        DiscretizedODEInputState(np.ones(1), np.ones(1), np.ones(1), 0.0), DELTA_T, 0.0
    ).stage_update

    assert update_pert == pytest.approx(2 + 2 * DELTA_T)


def test_compute_update_adjoint_derivative():
    """
    Tests the compute_update_adjoint_derivative method of the OpenMDAO ODE.
    """
    ODE.compute_update(
        DiscretizedODEInputState(np.ones(1), np.ones(1), np.ones(1), 0.0), DELTA_T, 0.0
    )

    stage_input_perturbation = ODE.compute_update_adjoint_derivative(
        DiscretizedODEResultState(np.ones(1), np.zeros(1), np.zeros(1)), DELTA_T, 0.0
    )
    assert stage_input_perturbation == pytest.approx(
        DiscretizedODEInputState(
            np.ones(1), np.full(1, DELTA_T), np.full(1, 1 + DELTA_T), 0.0
        )
    )


def test_reimport():
    """
    Tests that exporting and reimporting a cached linearization state provides the
    correct results.
    """
    ODE.compute_update(
        DiscretizedODEInputState(np.ones(1), np.ones(1), np.ones(1), 0.0), DELTA_T, 0.0
    )
    cache = ODE.get_linearization_point()
    update_pert = ODE.compute_update_derivative(
        DiscretizedODEInputState(np.ones(1), np.ones(1), np.ones(1), 0.0), DELTA_T, 0.0
    ).stage_update

    ODE.compute_update(
        DiscretizedODEInputState(np.zeros(1), np.zeros(1), np.zeros(1), 0.0),
        DELTA_T,
        0.0,
    )

    ODE.set_linearization_point(cache)
    update_pert_new = ODE.compute_update_derivative(
        DiscretizedODEInputState(np.ones(1), np.ones(1), np.ones(1), 0.0), DELTA_T, 0.0
    ).stage_update
    assert update_pert_new == update_pert


def test_compute_state_norm():
    state = DiscretizedODEResultState(np.zeros(0), np.full(1, 2.0), np.zeros(0))
    norm = ODE.compute_state_norm(state)
    assert norm == 2.0


@pytest.mark.mpi
def test_compute_state_norm_parallel():
    parallel_problem = om.Problem()
    parallel_problem.model.add_subsystem(
        "comp",
        Test1Component1(integration_control=test_integration_control),
        promotes=["*"],
    )
    par_ivc = om.IndepVarComp()
    par_ivc.add_output("x_old", shape=1, distributed=True)
    par_ivc.add_output("s_i", shape=1, distributed=True)
    parallel_problem.model.add_subsystem("ivc", par_ivc, promotes=["*"])
    parallel_problem.setup()
    parallel_problem.final_setup()
    parallel_problem.run_model()
    parallel_test_time_integration_quantities = ["x"]
    parallel_ode = OpenMDAOODE(
        parallel_problem,
        test_integration_control,
        parallel_test_time_integration_quantities,
    )
    result = parallel_ode.compute_state_norm(
        DiscretizedODEResultState(
            np.zeros(0),
            np.full(1, 2.0) if parallel_problem.comm.rank == 0 else np.full(1, 3.0),
            np.zeros(0),
        )
    )
    assert result == pytest.approx(13**0.5)
