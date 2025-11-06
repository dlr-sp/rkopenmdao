"""
Tests the wrapper for turning an OpenMDAO problem into a discretized ODE.
"""

from __future__ import annotations
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
from .distributed_variables_test import Test2Component1, Test2Component2

DELTA_T = 0.1


@pytest.fixture(name="identity_openmdao_ode")
def fixture_identity_openmdao_ode() -> OpenMDAOODE:
    """Creates an OpenMDAO ODE for the identity ODE x'=x."""
    test_integration_control = StepTerminationIntegrationControl(DELTA_T, 1, 0.0)
    test_problem = om.Problem()
    test_problem.model.add_subsystem(
        "comp", TestComp1(integration_control=test_integration_control), promotes=["*"]
    )
    test_problem.setup()
    test_problem.final_setup()
    test_time_integration_quantities = ["x"]
    test_independent_input_quantities = ["b"]
    ode = OpenMDAOODE(
        test_problem,
        test_integration_control,
        test_time_integration_quantities,
        test_independent_input_quantities,
    )
    return ode


@pytest.fixture(name="compute_update_output")
def fixture_compute_update_output(
    identity_openmdao_ode,
) -> DiscretizedODEResultState:
    """Solution of the ODE with export of linearization points."""
    return identity_openmdao_ode.compute_update(
        DiscretizedODEInputState(np.ones(1), np.ones(1), np.ones(1), 0.0),
        DELTA_T,
        0.0,
    )


def test_ode_attributes(identity_openmdao_ode):
    """
    Tests that all public attributes of the OpenMDAO ODE are accessible.
    """
    assert hasattr(identity_openmdao_ode, "time_integration_metadata")


def test_compute_update_stage_update(compute_update_output):
    """
    Tests the stage_update member of the compute_update method of the OpenMDAO ODE.
    """
    assert compute_update_output.stage_update == pytest.approx(np.array([1 + DELTA_T]))


@pytest.mark.parametrize(
    "result, expected_linearization",
    [
        (
            "compute_update_output",
            np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1 + DELTA_T]),
        ),
    ],
)
def test_compute_update_linearization_point(result, expected_linearization, request):
    """Test that the linearization point is correctly (not) exported."""
    result_state = request.getfixturevalue(result)
    assert result_state.linearization_point == pytest.approx(expected_linearization)


def test_compute_update_derivative(identity_openmdao_ode, compute_update_output):
    """
    Tests the compute_update_derivative method of the OpenMDAO ODE.
    """
    update_pert = identity_openmdao_ode.compute_update_derivative(
        DiscretizedODEInputState(
            np.ones(1),
            np.ones(1),
            np.ones(1),
            0.0,
            compute_update_output.linearization_point,
        ),
        DELTA_T,
        0.0,
    ).stage_update

    assert update_pert == pytest.approx(2 + 2 * DELTA_T)


def test_compute_update_adjoint_derivative(
    identity_openmdao_ode, compute_update_output
):
    """
    Tests the compute_update_adjoint_derivative method of the OpenMDAO ODE.
    """
    stage_input_perturbation = identity_openmdao_ode.compute_update_adjoint_derivative(
        DiscretizedODEResultState(
            np.ones(1),
            np.zeros(1),
            np.zeros(1),
            compute_update_output.linearization_point,
        ),
        DELTA_T,
        0.0,
    )
    assert stage_input_perturbation == pytest.approx(
        DiscretizedODEInputState(
            np.ones(1), np.full(1, DELTA_T), np.full(1, 1 + DELTA_T), 0.0
        )
    )


def test_reimport(identity_openmdao_ode):
    """
    Tests that exporting and reimporting a cached linearization state provides the
    correct results.
    """
    ode_result = identity_openmdao_ode.compute_update(
        DiscretizedODEInputState(np.ones(1), np.ones(1), np.ones(1), 0.0),
        DELTA_T,
        0.0,
    )
    cache = ode_result.linearization_point.copy()
    update_pert = identity_openmdao_ode.compute_update_derivative(
        DiscretizedODEInputState(np.ones(1), np.ones(1), np.ones(1), 0.0, cache),
        DELTA_T,
        0.0,
    ).stage_update

    identity_openmdao_ode.compute_update(
        DiscretizedODEInputState(np.zeros(1), np.zeros(1), np.zeros(1), 0.0),
        DELTA_T,
        0.0,
    )
    update_pert_new = identity_openmdao_ode.compute_update_derivative(
        DiscretizedODEInputState(np.ones(1), np.ones(1), np.ones(1), 0.0, cache),
        DELTA_T,
        0.0,
    ).stage_update
    assert update_pert_new == update_pert


@pytest.mark.parametrize(
    "state, expected_norm",
    [
        (
            DiscretizedODEResultState(
                np.zeros(1),
                np.ones(1),
                np.zeros(1),
            ),
            1.0,
        ),
        (
            DiscretizedODEResultState(
                np.zeros(1),
                np.full(1, -2),
                np.zeros(1),
            ),
            2.0,
        ),
    ],
)
def test_sequential_norm(identity_openmdao_ode, state, expected_norm):
    """Tests the norm calculation for strictly nondistributed variables."""
    norm = identity_openmdao_ode.compute_state_norm(state)
    assert norm == pytest.approx(expected_norm)


@pytest.fixture(name="distributed_openmdao_ode")
def fixture_complicated_openmdao_ode() -> OpenMDAOODE:
    """More complicated OpenMDAO ODE with a mix of distributed and non-distributed
    variables. Is used here only for the calculation of the norm."""
    test_integration_control = StepTerminationIntegrationControl(DELTA_T, 1, 0.0)
    test_problem = om.Problem()
    test_problem.model.add_subsystem(
        "comp", TestComp1(integration_control=test_integration_control), promotes=["*"]
    )
    test_problem.model.add_subsystem(
        "comp_1",
        Test2Component1(integration_control=test_integration_control),
        promotes=["*"],
    )
    test_problem.model.add_subsystem(
        "comp_2",
        Test2Component2(integration_control=test_integration_control),
        promotes=["*"],
    )
    ivc = om.IndepVarComp()
    ivc.add_output("x12_old", shape=1, distributed=True)
    ivc.add_output("s12_i", shape=1, distributed=True)
    ivc.add_output("x43_old", shape=1, distributed=True)
    ivc.add_output("s43_i", shape=1, distributed=True)
    test_problem.model.add_subsystem("ivc", ivc, promotes=["*"])
    test_problem.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=False, iprint=-1
    )
    test_problem.model.linear_solver = om.PETScKrylov(atol=1e-12, rtol=1e-12, iprint=-1)

    test_problem.setup()
    test_problem.final_setup()
    test_time_integration_quantities = ["x12", "x43", "x"]
    ode = OpenMDAOODE(
        test_problem,
        test_integration_control,
        test_time_integration_quantities,
    )
    return ode


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "state, solution_base",
    [
        (
            DiscretizedODEResultState(
                np.zeros(1),
                np.ones(3),
                np.zeros(1),
            ),
            1.0,
        ),
        (
            DiscretizedODEResultState(
                np.zeros(1),
                np.full(3, -2),
                np.zeros(1),
            ),
            2.0,
        ),
    ],
)
@pytest.mark.parametrize("excluded_vars", (["x"], ["x12"], ["x", "x43"]))
@pytest.mark.parametrize("order", [1.0, 2.0, 3.0, 4.0])
def test_parallel_norm(
    distributed_openmdao_ode, state, solution_base, excluded_vars, order
):
    """Tests the parallel compution of the norm"""
    # pylint: disable=protected-access
    # Set norm and its exclusion here after construction, since creating multiple
    # fixtures that is annoying.
    distributed_openmdao_ode._norm_order = order
    distributed_openmdao_ode._norm_exclusions = excluded_vars
    norm = distributed_openmdao_ode.compute_state_norm(state)
    num_vals = 5
    if "x" in excluded_vars:
        num_vals -= 1
    if "x12" in excluded_vars:
        num_vals -= 2
    if "x43" in excluded_vars:
        num_vals -= 2
    expected_norm = num_vals ** (1 / order) * solution_base
    assert norm == pytest.approx(expected_norm)
