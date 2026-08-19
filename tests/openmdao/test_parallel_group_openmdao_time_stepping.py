"""Test to make sure that rkopenmdao works with problems containing parallel groups."""

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals
import pytest

from om_components import (
    FirstParallelGroupChain,
    SecondParallelGroupChain1,
    SecondParallelGroupChain2,
    ThirdParallelGroupChain,
    parallel_group_chain_solution,
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


@pytest.fixture(params=["fwd", "rev"], name="parallel_group_problem")
def parallel_group_problem_fixture(request):
    """TODO"""
    problem = om.Problem()
    problem.model.add_subsystem("First", FirstParallelGroupChain())
    par_group = om.ParallelGroup()
    group_1 = om.Group()
    ivc21 = om.IndepVarComp()
    ivc21.add_output("c_old")
    ivc21.add_output("c_accumulated_stages")
    group_1.add_subsystem("ivc21", ivc21, promotes=["*"])
    group_1.add_subsystem(
        "comp_21",
        SecondParallelGroupChain1(),
        promotes=["*"],
    )
    par_group.add_subsystem("Second_1", group_1)
    group_2 = om.Group()
    ivc22 = om.IndepVarComp()
    ivc22.add_output("b_old")
    ivc22.add_output("b_accumulated_stages")
    group_1.add_subsystem("ivc22", ivc22, promotes=["*"])
    group_2.add_subsystem(
        "comp_22",
        SecondParallelGroupChain2(),
        promotes=["*"],
    )
    par_group.add_subsystem("Second_2", group_2)
    problem.model.add_subsystem("Second", par_group)
    problem.model.add_subsystem("Third", ThirdParallelGroupChain())
    problem.model.connect("First.d_state", "Second.Second_1.d")
    problem.model.connect("First.d_state", "Second.Second_2.d")
    problem.model.connect("Second.Second_1.c_state", "Third.c")
    problem.model.connect("Second.Second_2.b_state", "Third.b")
    problem.setup(mode=request.param)
    problem.final_setup()
    return problem


@pytest.fixture(
    params=[AllCheckpointTimeIntegration, PyrevolveTimeIntegration],
    name="parallel_group_om_time_integration",
)
def parallel_group_om_time_integration_fixture(parallel_group_problem, request):
    """TODO"""
    time_integration = request.param(
        ode=OpenMDAOODE(parallel_group_problem, ["a", "b", "c", "d"]),
        time_discretization_scheme=StageOrderedRungeKuttaDiscretization(
            embedded_second_order_two_stage_sdirk
        ),
        time_integration_config=IntegrationConfig(
            False, PredefinedNumberOfSteps(100), 0.001
        ),
    )
    problem = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output("b_initial", shape_by_conn=True, distributed=True)
    ivc.add_output("c_initial", shape_by_conn=True, distributed=True)
    problem.model.add_subsystem("ivc", ivc, promotes=["*"])
    problem.model.add_subsystem(
        "time_integration",
        OpenMDAOTimeStepping(time_integrator=time_integration),
        promotes=["*"],
    )
    problem.setup()
    return problem


def test_parallel_group_time_integration(parallel_group_om_time_integration):
    """TODO"""
    parallel_group_om_time_integration.run_model()
    analytical_solution = parallel_group_chain_solution(0.1, np.ones(4))[
        [0, 1 if parallel_group_om_time_integration.comm.rank == 0 else 2, 3]
    ]
    numerical_solution = np.array(
        [
            parallel_group_om_time_integration["d_final"][:],
            parallel_group_om_time_integration.get_val(
                (
                    "c_final"
                    if parallel_group_om_time_integration.comm.rank == 0
                    else "b_final"
                ),
                get_remote=False,
            )[:],
            parallel_group_om_time_integration["a_final"][:],
        ]
    ).flatten()
    assert numerical_solution == pytest.approx(analytical_solution)


def test_parallel_group_time_integration_totals(parallel_group_om_time_integration):
    """TODO"""
    parallel_group_om_time_integration.run_model()
    if parallel_group_om_time_integration.comm.rank == 0:
        data = parallel_group_om_time_integration.check_totals(
            ["a_final", "d_final"],
            [
                "a_initial",
                "d_initial",
            ],
        )
    else:
        data = parallel_group_om_time_integration.check_totals(
            ["a_final", "d_final"],
            [
                "a_initial",
                "d_initial",
            ],
            out_stream=None,
        )
    assert_check_totals(data)
