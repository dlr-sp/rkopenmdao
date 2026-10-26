"""
Tests the time integration of the ODE system x' = y, y' = x modeled by OpenMDAO
components, both in a single and in two split components.
"""

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
import pytest

from om_components import (
    ODE2dUnified,
    ODE2dSplit1,
    ODE2dSplit2,
    ode2d_analytical_solution,
)

from rkopenmdao.butcher_tableaux import embedded_fourth_order_five_stage_sdirk
from rkopenmdao.checkpointed_time_integration.pyrevolve_time_integration import (
    PyrevolveTimeIntegration,
)
from rkopenmdao.discretized_ode.openmdao_ode import OpenMDAOODE
from rkopenmdao.openmdao_time_stepping import OpenMDAOTimeStepping
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.termination_criterion import PredefinedNumberOfSteps
from rkopenmdao.time_discretization.stage_ordered_runge_kutta_discretization import (
    StageOrderedRungeKuttaDiscretization,
)


@pytest.fixture(name="single_component_problem")
def single_component_problem_fixture():
    """Provide an OpenMDAO problem with a single ODE2dUnified component.

    The problem models the ODE system x' = y, y' = x in one component and
    is set up already, i.e. its final_setup() method has been called.
    """
    problem = om.Problem()
    problem.model.add_subsystem("ode_component", ODE2dUnified(), promotes=["*"])
    problem.setup()
    problem.final_setup()
    return problem


@pytest.fixture(name="split_component_problem")
def split_component_problem_fixture():
    """Provide an OpenMDAO problem with the ODE2dSplit1 and ODE2dSplit2
    components.

    The problem models the ODE system x' = y, y' = x in two split
    components, solved with a Newton solver, and is set up already, i.e.
    its final_setup() method has been called.
    """
    problem = om.Problem()
    problem.model.add_subsystem("ode_component_1", ODE2dSplit1(), promotes=["*"])
    problem.model.add_subsystem("ode_component_2", ODE2dSplit2(), promotes=["*"])
    problem.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=-1)
    problem.model.linear_solver = om.ScipyKrylov()
    problem.setup()
    problem.final_setup()
    return problem


@pytest.fixture(name="single_component_om_time_integrator")
def single_component_om_time_integrator_fixture(single_component_problem):
    """Provide an OpenMDAO problem time integrating the single component ODE.

    Wraps ``single_component_problem`` into an OpenMDAOODE, which is
    integrated by a PyrevolveTimeIntegration with the embedded fourth
    order SDIRK scheme for 100 steps of size 0.01, and embeds the
    integration into a new OpenMDAO problem as an OpenMDAOTimeStepping
    component.
    """
    time_integration = PyrevolveTimeIntegration(
        ode=OpenMDAOODE(single_component_problem, ["x"]),
        time_discretization_scheme=StageOrderedRungeKuttaDiscretization(
            embedded_fourth_order_five_stage_sdirk
        ),
        time_integration_config=IntegrationConfig(
            False, PredefinedNumberOfSteps(100), 0.01
        ),
    )
    problem = om.Problem()
    problem.model.add_subsystem(
        "time_integration",
        OpenMDAOTimeStepping(time_integrator=time_integration),
        promotes=["*"],
    )
    problem.setup()
    return problem


@pytest.fixture(name="split_component_om_time_integrator")
def split_component_om_time_integrator_fixture(split_component_problem):
    """Provide an OpenMDAO problem time integrating the split component ODE.

    Wraps ``split_component_problem`` into an OpenMDAOODE, which is
    integrated by a PyrevolveTimeIntegration with the embedded fourth
    order SDIRK scheme for 100 steps of size 0.01, and embeds the
    integration into a new OpenMDAO problem as an OpenMDAOTimeStepping
    component.
    """
    time_integration = PyrevolveTimeIntegration(
        ode=OpenMDAOODE(split_component_problem, ["x", "y"]),
        time_discretization_scheme=StageOrderedRungeKuttaDiscretization(
            embedded_fourth_order_five_stage_sdirk
        ),
        time_integration_config=IntegrationConfig(
            False, PredefinedNumberOfSteps(100), 0.01
        ),
    )
    problem = om.Problem()
    problem.model.add_subsystem(
        "time_integration",
        OpenMDAOTimeStepping(time_integrator=time_integration),
        promotes=["*"],
    )
    problem.setup()
    return problem


def test_single_component_partials(single_component_problem):
    """Check the partial derivatives of the single component ODE problem."""
    single_component_problem.run_model()
    partials_data = single_component_problem.check_partials()
    assert_check_partials(partials_data)


def test_split_component_partials(split_component_problem):
    """Check the partial derivatives of the split component ODE problem."""
    split_component_problem.run_model()
    partials_data = split_component_problem.check_partials()
    assert_check_partials(partials_data)


def test_single_component_time_integration(single_component_om_time_integrator):
    """Integrate the single component ODE and compare the final state to the
    analytical solution at time 1 for the initial value (1, 1)."""
    single_component_om_time_integrator.run_model()
    assert single_component_om_time_integrator["x_final"] == pytest.approx(
        ode2d_analytical_solution(1, np.ones(2), 0.0)
    )


def test_split_component_time_integration(split_component_om_time_integrator):
    """Integrate the split component ODE and compare the final state (x, y) to
    the analytical solution at time 1 for the initial value (1, 1)."""
    split_component_om_time_integrator.run_model()
    assert np.array(
        [
            split_component_om_time_integrator["x_final"][0],
            split_component_om_time_integrator["y_final"][0],
        ]
    ) == pytest.approx(ode2d_analytical_solution(1, np.ones(2), 0.0))


def test_single_component_time_integration_partials(
    single_component_om_time_integrator,
):
    """Check the partial derivatives of the time integrated single component
    ODE problem."""
    single_component_om_time_integrator.run_model()
    partials_data = single_component_om_time_integrator.check_partials()
    assert_check_partials(partials_data)


def test_split_component_time_integration_partials(split_component_om_time_integrator):
    """Check the partial derivatives of the time integrated split component
    ODE problem."""
    split_component_om_time_integrator.run_model()
    partials_data = split_component_om_time_integrator.check_partials()
    assert_check_partials(partials_data)
