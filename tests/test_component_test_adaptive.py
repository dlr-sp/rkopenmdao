"""Tests the adaptive time integration with the various components of
test_components.py"""

# pylint: disable=duplicate-code

from itertools import product

import openmdao.api as om
import pytest
import numpy as np

from rkopenmdao.error_controller import ErrorControllerConfig

from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as two_stage_dirk,
    embedded_second_order_three_stage_esdirk as three_stage_esdirk,
    embedded_heun_euler as heun_euler,
    embedded_third_order_four_stage_esdirk as four_stage_esdirk,
    embedded_fourth_order_five_stage_esdirk as five_stage_esdirk,
)
from rkopenmdao.checkpoint_interface.no_checkpoint_time_integration import (
    NoCheckpointer,
)
from rkopenmdao.checkpoint_interface.all_checkpoint_time_integration import (
    AllCheckpointer,
)
from rkopenmdao.error_controllers import (
    integral,
    h_211,
    h0_211,
    pid,
    h_312,
    h0_312,
    ppid,
    h_321,
    h0_321,
)
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.termination_criterion import PredefinedFinalTime

from .test_components import (
    TestComp1,
    TestComp2,
    TestComp3,
    TestComp4,
    Testcomp51,
    Testcomp52,
    TestComp6,
    TestComp6a,
    TestComp7,
)

from .test_components import (
    solution_test1,
    solution_test2,
    solution_test3,
    solution_test4,
    solution_test6,
    solution_test7,
)

test_comp_class_list = [
    TestComp1,
    TestComp2,
    TestComp3,
    TestComp4,
    Testcomp51,
    Testcomp52,
    TestComp6,
    TestComp6a,
    TestComp7,
]

error_controller_list = [
    integral,
    h_211,
    h0_211,
    pid,
    h_312,
    h0_312,
    ppid,
    h_321,
    h0_321,
]

times = np.linspace(1.0, 9.0, 3)
butcher_diagonal_elements = np.linspace(0.0, 1.0, 3)


@pytest.mark.parametrize(
    "test_class, test_functor, initial_time, initial_values",
    (
        [TestComp1, solution_test1, 0.0, np.array([1.0])],
        [TestComp1, solution_test1, 1.0, np.array([1.0])],
        [TestComp2, solution_test2, 0.0, np.array([1.0])],
        [TestComp2, solution_test2, 1.0, np.array([1.0])],
        [TestComp3, solution_test3, 0.0, np.array([1.0])],
        [TestComp3, solution_test3, 1.0, np.array([1.0])],
        [TestComp4, solution_test4, 0.0, np.array([[1.0, 1.0]])],
        [TestComp4, solution_test4, 1.0, np.array([[1.0, 1.0]])],
        [TestComp6, solution_test6, 0.0, np.array([1.0])],
        [TestComp6, solution_test6, 1.0, np.array([1.0])],
        [TestComp6a, solution_test6, 0.0, np.array([1.0])],
        [TestComp6a, solution_test6, 1.0, np.array([1.0])],
        [TestComp7, solution_test7, 1.0, np.array([1.0])],
        [TestComp7, solution_test7, 2.0, np.array([1.0])],
    ),
)
@pytest.mark.parametrize(
    "butcher_tableau",
    [
        heun_euler,
        two_stage_dirk,
        three_stage_esdirk,
        four_stage_esdirk,
        five_stage_esdirk,
    ],
)
@pytest.mark.parametrize("quantities", [["x"]])
@pytest.mark.parametrize(
    "test_measurer", [SimpleErrorMeasurer(), ImprovedErrorMeasurer()]
)
@pytest.mark.parametrize("test_controller", error_controller_list)
def test_component_integration(
    test_class,
    test_functor,
    initial_time,
    initial_values,
    butcher_tableau,
    quantities,
    test_measurer,
    test_controller,
):
    """Tests the time integration of the different components."""
    integration_config = IntegrationConfig(
        use_adaptive_time_stepping=True,
        termination_criterion=PredefinedFinalTime(initial_time + 0.01),
        initial_step_size=0.01,
    )
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem("test_comp", test_class())
    time_integration_prob.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True
    )
    time_integration_prob.model.linear_solver = om.ScipyKrylov()

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_config=integration_config,
            time_integration_quantities=quantities,
            error_controller=[test_controller, integral],
            error_measurer=test_measurer,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["time_initial"] = initial_time
    for i, quantity in enumerate(quantities):
        runge_kutta_prob[quantity + "_initial"] = initial_values[i]

    runge_kutta_prob.run_model()

    result = np.zeros_like(initial_values)
    for i, quantity in enumerate(quantities):
        result[i] = runge_kutta_prob[quantity + "_final"][0]

    # relatively coarse, but this isn't supposed to test the accuracy,
    # it's just to make sure the solution is in the right region
    assert test_functor(
        initial_time + 0.01,
        initial_values,
        initial_time,
    ) == pytest.approx(result, rel=1e-4)


@pytest.mark.parametrize(
    "test_class, initial_time",
    list(
        product(
            [TestComp1, TestComp2, TestComp3, TestComp4, TestComp6, TestComp6a],
            [0.0, 1.0],
        )
    )
    + [[TestComp7, 1.0]]
    + [[TestComp7, 2.0]],
)
@pytest.mark.parametrize(
    "butcher_tableau",
    [
        heun_euler,
        two_stage_dirk,
        three_stage_esdirk,
        four_stage_esdirk,
        five_stage_esdirk,
    ],
)
@pytest.mark.parametrize(
    "checkpointing_implementation",
    [
        NoCheckpointer,
        AllCheckpointer,
    ],
)
def test_time_integration_partials(
    test_class,
    initial_time,
    butcher_tableau,
    checkpointing_implementation,
):
    """Tests the partials of the time integration of the different components."""
    integration_config = IntegrationConfig(
        use_adaptive_time_stepping=True,
        termination_criterion=PredefinedFinalTime(initial_time + 0.01),
        initial_step_size=0.01,
    )
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem("test_comp", test_class())

    time_integration_prob.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True
    )
    time_integration_prob.model.linear_solver = om.ScipyKrylov()

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_config=integration_config,
            time_integration_quantities=["x"],
            checkpointing_type=checkpointing_implementation,
            error_controller=[integral],
            error_controller_options={"config": ErrorControllerConfig(tol=1e-6)},
            error_measurer=SimpleErrorMeasurer(),
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["time_initial"] = initial_time

    runge_kutta_prob.run_model()
    if checkpointing_implementation == NoCheckpointer:
        with pytest.raises(NotImplementedError):
            runge_kutta_prob.check_partials()

    else:
        data = runge_kutta_prob.check_partials()
        check_partials_wo_fd(data)


def check_partials_wo_fd(jac_data: dict, tol: float = 1e-6):
    """
    Since FD by the Openmdao and fwd/rev are not comparable for adaptive schemes, a
    function excluding fd is necessary. The fd of OpenMDAO perturbs the inputs/initial
    values, which lead the error estimation to also be differentiated.
    The implementation of fwd and rev mode explicitely excludes the error estimation
    from the derivatives, because else that would introduce errors
    (see 1. https://doi.org/10.1016/j.cam.2009.08.109 and
    2. http://dx.doi.org/10.1090/S0025-5718-99-01027-3
    """
    for pair in jac_data["rk_integrator"]:
        fwd = jac_data["rk_integrator"][pair]["J_fwd"]
        rev = jac_data["rk_integrator"][pair]["J_rev"]
        # Absolute :
        assert rev == pytest.approx(fwd, rel=tol, abs=tol)
