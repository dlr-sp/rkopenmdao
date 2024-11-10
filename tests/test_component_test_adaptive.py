from itertools import product

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
import pytest
import numpy as np

from rkopenmdao.integration_control import IntegrationControl, TerminationCriterion
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as two_stage_dirk,
    embedded_second_order_three_stage_esdirk as three_stage_esdirk,
    embedded_heun_euler as heun_euler,
    embedded_third_order_four_stage_esdirk as four_stage_esdirk,
    embedded_fourth_order_five_stage_esdirk as five_stage_esdirk,
)
from rkopenmdao.checkpoint_interface.no_checkpointer import NoCheckpointer
from rkopenmdao.checkpoint_interface.all_checkpointer import AllCheckpointer
from rkopenmdao.error_controllers import *
from rkopenmdao.error_estimator import *
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
times = np.linspace(1.0, 9.0, 3)
butcher_diagonal_elements = np.linspace(0.0, 1.0, 3)

# -----------------------------------------------------------------------------------------------
# Test 1

@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize("test_class", test_comp_class_list)
@pytest.mark.parametrize("time", times)
@pytest.mark.parametrize("butcher_diagonal_element", butcher_diagonal_elements)
def test_component_partials(test_class, time, butcher_diagonal_element):
    termination_criterion = TerminationCriterion('end_time', 0.1)
    integration_control = IntegrationControl(0.0, termination_criterion, 0.1)
    integration_control.stage_time = time
    integration_control.butcher_diagonal_element = butcher_diagonal_element
    test_prob = om.Problem()
    test_prob.model.add_subsystem(
        "test_comp", test_class(integration_control=integration_control)
    )

    test_prob.setup()
    test_prob.run_model()
    data = test_prob.check_partials()
    assert_check_partials(data)

# -----------------------------------------------------------------------------------------------
# Test 2

@pytest.mark.rk
@pytest.mark.rk_openmdao
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
    "butcher_tableau", [heun_euler, two_stage_dirk, three_stage_esdirk, four_stage_esdirk, five_stage_esdirk]
)
@pytest.mark.parametrize("quantities", [["x"]])
@pytest.mark.parametrize(
    "test_estimator",
    [SimpleErrorEstimator, ImprovedErrorEstimator])
@pytest.mark.parametrize(
    "test_controller",
    [Integral, H0_110, PC]
)
def test_component_integration(
    test_class, test_functor, initial_time, initial_values,
    butcher_tableau, quantities, test_estimator, test_controller
):
    """Tests the time integration of the different components."""
    termination_criterion = TerminationCriterion('end_time', initial_time + 0.01)
    integration_control = IntegrationControl(initial_time, termination_criterion, 0.001)
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", test_class(integration_control=integration_control)
    )
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
            integration_control=integration_control,
            time_integration_quantities=quantities,
            error_controller=test_controller,
            error_estimator_type=test_estimator,
            adaptive_time_stepping=True,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    for i, quantity in enumerate(quantities):
        runge_kutta_prob[quantity + "_initial"] = initial_values[i]

    runge_kutta_prob.run_model()

    result = np.zeros_like(initial_values)
    for i, quantity in enumerate(quantities):
        result[i] = runge_kutta_prob[quantity + "_final"]

    # relatively coarse, but this isn't supposed to test the accuracy,
    # it's just to make sure the solution is in the right region
    assert test_functor(
        initial_time + 0.01,
        initial_values,
        initial_time,
    ) == pytest.approx(result, rel=1e-4)


@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize(
    "test_class, test_functor, initial_time, initial_values",
    (
        [TestComp1, solution_test1, 0.0, np.array([1.0])],
        [TestComp1, solution_test1, 1.0, np.array([1.0])],
    ),
)
@pytest.mark.parametrize("parameter", [-1.0, 0.0, 2.0])
@pytest.mark.parametrize(
    "butcher_tableau", 
    [heun_euler, two_stage_dirk, three_stage_esdirk, four_stage_esdirk, five_stage_esdirk]
)
@pytest.mark.parametrize("quantities", [["x"]])
@pytest.mark.parametrize(
    "test_estimator",
    [SimpleErrorEstimator, ImprovedErrorEstimator])
@pytest.mark.parametrize(
    "test_controller",
    [Integral, H0_110, PC]
)
def test_component_integration_with_parameter(
    test_class,
    test_functor,
    initial_time,
    initial_values,
    parameter,
    butcher_tableau,
    quantities,
    test_estimator,
    test_controller
):
    """Tests the time integration of the different components."""
    termination_criterion = TerminationCriterion('end_time', initial_time + 0.01)
    integration_control = IntegrationControl(initial_time, termination_criterion, 0.001)
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", test_class(integration_control=integration_control)
    )

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
            integration_control=integration_control,
            time_integration_quantities=quantities,
            time_independent_input_quantities=["b"],
            error_controller=test_controller,
            error_estimator_type=test_estimator,
            adaptive_time_stepping=True,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    for i, quantity in enumerate(quantities):
        runge_kutta_prob[quantity + "_initial"] = initial_values[i]
    runge_kutta_prob["b"] = parameter
    runge_kutta_prob.run_model()

    result = np.zeros_like(initial_values)
    for i, quantity in enumerate(quantities):
        result[i] = runge_kutta_prob[quantity + "_final"]

    # relatively coarse, but this isn't supposed to test the accuracy,
    # it's just to make sure the solution is in the right region
    assert test_functor(
        initial_time + 0.01, initial_values, initial_time, parameter
    ) == pytest.approx(result, rel=1e-4)



@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize("initial_time", [0.0, 1.0])
@pytest.mark.parametrize("initial_values", [[1.0, 1.0]])
@pytest.mark.parametrize(
    "butcher_tableau", 
    [heun_euler, two_stage_dirk, three_stage_esdirk, four_stage_esdirk, five_stage_esdirk]
)
@pytest.mark.parametrize(
    "test_estimator",
    [SimpleErrorEstimator, ImprovedErrorEstimator])
@pytest.mark.parametrize(
    "test_controller",
    [Integral, H0_110, PC]
)
def test_component_splitting(initial_time, initial_values, butcher_tableau, test_estimator, test_controller):
    """Tests the time integration of the problem that is split over multiple
    components."""
    termination_criterion = TerminationCriterion('end_time', initial_time + 0.01)
    integration_control_1 = IntegrationControl(initial_time, termination_criterion, 0.001)
    integration_control_2 = IntegrationControl(initial_time, termination_criterion, 0.001)

    time_integration_prob_1 = om.Problem()
    time_integration_prob_1.model.add_subsystem(
        "single_comp", TestComp4(integration_control=integration_control_1)
    )

    time_integration_prob_2 = om.Problem()
    time_integration_prob_2.model.add_subsystem(
        "first_comp",
        Testcomp51(integration_control=integration_control_2),
        promotes=["x_stage", "y_stage"],
    )
    time_integration_prob_2.model.add_subsystem(
        "second_comp",
        Testcomp52(integration_control=integration_control_2),
        promotes=["x_stage", "y_stage"],
    )

    runge_kutta_prob_1 = om.Problem()
    runge_kutta_prob_1.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob_1,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control_1,
            time_integration_quantities=["x"],
            error_controller=test_controller,
            error_estimator_type=test_estimator,
            adaptive_time_stepping=True,
        ),
        promotes=["*"],
    )
    runge_kutta_prob_1.setup()
    runge_kutta_prob_1["x_initial"] = initial_values
    runge_kutta_prob_1.run_model()

    result_1 = runge_kutta_prob_1["x_final"]

    runge_kutta_prob_2 = om.Problem()
    runge_kutta_prob_2.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob_2,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control_2,
            time_integration_quantities=["x", "y"],
            error_controller=test_controller,
            error_estimator_type=test_estimator,
            adaptive_time_stepping=True,
        ),
        promotes=["*"],
    )

    runge_kutta_prob_2.setup()
    runge_kutta_prob_2["x_initial"] = initial_values[0]
    runge_kutta_prob_2["y_initial"] = initial_values[1]
    runge_kutta_prob_2.run_model()
    result_2 = np.zeros_like(initial_values)
    result_2[0] = runge_kutta_prob_2["x_final"]
    result_2[1] = runge_kutta_prob_2["y_final"]

    assert result_1 == pytest.approx(result_2)



@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize(
    "test_class, initial_time",
    list(
        product(
            [TestComp1],
            [0.0],
        )
    )
)
@pytest.mark.parametrize(
    "butcher_tableau", 
    [
        heun_euler, 
        #two_stage_dirk, 
        #three_stage_esdirk, 
        #four_stage_esdirk, 
        #five_stage_esdirk
    ]

)
@pytest.mark.parametrize(
    "checkpointing_implementation",
    [
    #NoCheckpointer, 
    AllCheckpointer,
    #PyrevolveCheckpointer,
    ]
)
@pytest.mark.parametrize(
    "test_estimator",
    [SimpleErrorEstimator, 
    #ImprovedErrorEstimator
    ])
@pytest.mark.parametrize(
    "test_controller",
    [Integral, 
    #H0_110, 
    #PC,
    ]
)
def test_time_integration_partials(
    test_class, initial_time, butcher_tableau, checkpointing_implementation, test_estimator, test_controller
):
    """Tests the partials of the time integration of the different components."""
    print("Test_case: ",test_class)
    print("Checkpointing: ",checkpointing_implementation)
    termination_criterion = TerminationCriterion('end_time', initial_time + 0.01)
    integration_control = IntegrationControl(initial_time, termination_criterion, 0.001)
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", test_class(integration_control=integration_control)
    )

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
            integration_control=integration_control,
            time_integration_quantities=["x"],
            checkpointing_type=checkpointing_implementation,
            error_controller=test_controller,
            error_estimator_type=test_estimator,
            adaptive_time_stepping=True,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()

    runge_kutta_prob.run_model()
    if checkpointing_implementation == NoCheckpointer:
        with pytest.raises(NotImplementedError):
            runge_kutta_prob.check_partials()

    else:
        data = runge_kutta_prob.check_partials()
        assert_check_partials(data)


@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize(
    "test_class, initial_time",
    (
        [TestComp1, 0.0],
        [TestComp1, 1.0],
    ),
)
@pytest.mark.parametrize("parameter", [1.0])
@pytest.mark.parametrize(
    "butcher_tableau",
    [heun_euler, two_stage_dirk, three_stage_esdirk, four_stage_esdirk, five_stage_esdirk]
)
@pytest.mark.parametrize(
    "checkpointing_implementation",
    [
        NoCheckpointer,
        AllCheckpointer,
        #PyrevolveCheckpointer,
    ],
)
@pytest.mark.parametrize(
    "test_estimator",
    [SimpleErrorEstimator, ImprovedErrorEstimator])
@pytest.mark.parametrize(
    "test_controller",
    [Integral, H0_110, PC]
)
def test_time_integration_with_parameter_partials(
    test_class, initial_time, parameter, butcher_tableau, checkpointing_implementation, test_estimator, test_controller
):
    """Tests the partials of the time integration of the different components."""
    termination_criterion = TerminationCriterion('end_time', initial_time + 0.01)
    integration_control = IntegrationControl(initial_time, termination_criterion, 0.001)
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", test_class(integration_control=integration_control)
    )

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
            integration_control=integration_control,
            time_integration_quantities=["x"],
            time_independent_input_quantities=["b"],
            checkpointing_type=checkpointing_implementation,
            error_controller=test_controller,
            error_estimator_type=test_estimator,
            adaptive_time_stepping=True,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["b"] = parameter
    runge_kutta_prob.run_model()
    if checkpointing_implementation == NoCheckpointer:
        with pytest.raises(NotImplementedError):
            runge_kutta_prob.check_partials()
    else:
        data = runge_kutta_prob.check_partials()
        assert_check_partials(data)
