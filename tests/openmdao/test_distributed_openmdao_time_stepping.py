"""Test to make sure that rkopenmdao works with problems containing distributed
variables."""

import openmdao.api as om

from openmdao.utils.assert_utils import assert_check_totals
import numpy as np
import pytest

from om_components import (
    ODE4dDistributedSplit1,
    ODE4dDistributedSplit2,
    ode4d_analytical_solution,
)

from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk,
)
from rkopenmdao.checkpointed_time_integration.all_checkpoint_time_integration import (
    AllCheckpointTimeIntegration,
)
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


@pytest.fixture(params=["fwd", "rev"], name="distributed_problem")
def distributed_problem_fixture(request):
    """Provide an OpenMDAO problem with the ODE4dDistributedSplit components.

    The problem models the ODE system x1' = x4, x2' = x1, x3' = x2,
    x4' = x3 with two components on distributed variables, solved with a
    Newton solver, in the mode given by the fixture parameter, and is set
    up already, i.e. its final_setup() method has been called.
    """
    problem = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output("x12_old", shape=1, distributed=True)
    ivc.add_output("s12_i", shape=1, distributed=True)
    ivc.add_output("x43_old", shape=1, distributed=True)
    ivc.add_output("s43_i", shape=1, distributed=True)
    problem.model.add_subsystem("ivc", ivc, promotes=["*"])
    problem.model.add_subsystem(
        "ode_component_1", ODE4dDistributedSplit1(), promotes=["*"]
    )
    problem.model.add_subsystem(
        "ode_component_2", ODE4dDistributedSplit2(), promotes=["*"]
    )
    problem.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, iprint=-1)
    problem.model.linear_solver = om.PETScKrylov(atol=1e-12, rtol=1e-12, iprint=-1)
    problem.setup(mode=request.param)
    problem.final_setup()
    return problem


@pytest.fixture(
    params=[AllCheckpointTimeIntegration, PyrevolveTimeIntegration],
    name="distributed_om_time_integration",
)
def distributed_om_time_integration_fixture(distributed_problem, request):
    """Provide an OpenMDAO problem time integrating the distributed ODE.

    Wraps ``distributed_problem`` into an OpenMDAOODE, which is integrated
    by the checkpointed time integration implementation selected by the
    fixture parameter with the embedded second order SDIRK scheme for 10
    steps of size 0.01, and embeds the integration into a new OpenMDAO
    problem as an OpenMDAOTimeStepping component with distributed initial
    values.
    """
    time_integration = request.param(
        ode=OpenMDAOODE(distributed_problem, ["x12", "x43"]),
        time_discretization_scheme=StageOrderedRungeKuttaDiscretization(
            embedded_second_order_two_stage_sdirk
        ),
        time_integration_config=IntegrationConfig(
            False, PredefinedNumberOfSteps(10), 0.01
        ),
    )
    problem = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output("x12_initial", shape=1, distributed=True)
    ivc.add_output("x43_initial", shape=1, distributed=True)
    problem.model.add_subsystem("ivc", ivc, promotes=["*"])
    problem.model.add_subsystem(
        "time_integration",
        OpenMDAOTimeStepping(time_integrator=time_integration),
        promotes=["*"],
    )
    problem.setup()
    problem.final_setup()
    return problem


def test_distributed_time_integration(distributed_om_time_integration):
    """Integrate the distributed ODE and compare the final state of this
    rank to the analytical solution at time 0.1 for the initial value
    (1, 1, 1, 1)."""
    distributed_om_time_integration.run_model()

    analytical_solution = ode4d_analytical_solution(0.1, np.ones(4))
    numerical_solution = np.array(
        [
            distributed_om_time_integration.get_val("x12_final", get_remote=False),
            distributed_om_time_integration.get_val("x43_final", get_remote=False),
        ]
    ).flatten()
    assert numerical_solution == pytest.approx(
        analytical_solution[
            [
                distributed_om_time_integration.comm.rank,
                3 - distributed_om_time_integration.comm.rank,
            ]
        ]
    )


def test_distributed_time_intetgration_totals(distributed_om_time_integration):
    """Run the time integration and check that the total derivatives of the
    final state with respect to the initial values are correct."""
    distributed_om_time_integration.run_model()
    if distributed_om_time_integration.comm.rank > 0:
        data = distributed_om_time_integration.check_totals(
            of=["x12_final", "x43_final"],
            wrt=["x12_initial", "x43_initial"],
            abs_err_tol=1e-4,
            rel_err_tol=1e-4,
            out_stream=None,
        )
    else:
        data = distributed_om_time_integration.check_totals(
            of=["x12_final", "x43_final"],
            wrt=["x12_initial", "x43_initial"],
            abs_err_tol=1e-4,
            rel_err_tol=1e-4,
        )
    assert_check_totals(data, atol=1e-4, rtol=1e-4)
