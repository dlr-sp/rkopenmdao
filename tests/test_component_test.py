"""Tests the time integration with the various components of test_components.py"""

from itertools import product

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
import pytest
import numpy as np

from rkopenmdao.integration_control import StepTerminationIntegrationControl
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    embedded_second_order_two_stage_sdirk as two_stage_dirk,
    runge_kutta_four,
)
from rkopenmdao.checkpoint_interface.no_checkpointer import NoCheckpointer
from rkopenmdao.checkpoint_interface.all_checkpointer import AllCheckpointer
from rkopenmdao.checkpoint_interface.pyrevolve_checkpointer import PyrevolveCheckpointer


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


@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize("test_class", test_comp_class_list)
@pytest.mark.parametrize("time", times)
@pytest.mark.parametrize("butcher_diagonal_element", butcher_diagonal_elements)
def test_component_partials(test_class, time, butcher_diagonal_element):
    """Tests whether the components itself produce the right partials"""
    integration_control = StepTerminationIntegrationControl(0.1, 1, 0.0)
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
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
@pytest.mark.parametrize("quantities", [["x"]])
def test_component_integration(
    test_class, test_functor, initial_time, initial_values, butcher_tableau, quantities
):
    """Tests the time integration of the different components."""
    integration_control = StepTerminationIntegrationControl(0.001, 10, initial_time)
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
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
@pytest.mark.parametrize("quantities", [["x"]])
def test_component_integration_with_parameter(
    test_class,
    test_functor,
    initial_time,
    initial_values,
    parameter,
    butcher_tableau,
    quantities,
):
    """Tests the time integration of the different components."""
    integration_control = StepTerminationIntegrationControl(0.001, 10, initial_time)
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
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
def test_component_splitting(initial_time, initial_values, butcher_tableau):
    """Tests the time integration of the problem that is split over multiple
    components."""
    integration_control_1 = StepTerminationIntegrationControl(0.001, 10, initial_time)
    integration_control_2 = StepTerminationIntegrationControl(0.001, 10, initial_time)

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
            [TestComp1, TestComp2, TestComp3, TestComp4, TestComp6, TestComp6a],
            [0.0, 1.0],
        )
    )
    + [[TestComp7, 1.0]]
    + [[TestComp7, 2.0]],
)
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
@pytest.mark.parametrize(
    "checkpointing_implementation",
    [NoCheckpointer, AllCheckpointer, PyrevolveCheckpointer],
)
def test_time_integration_partials(
    test_class, initial_time, butcher_tableau, checkpointing_implementation
):
    """Tests the partials of the time integration of the different components."""
    integration_control = StepTerminationIntegrationControl(0.001, 10, initial_time)
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
    [
        implicit_euler,
        two_stage_dirk,
        runge_kutta_four,
    ],
)
@pytest.mark.parametrize(
    "checkpointing_implementation",
    [
        NoCheckpointer,
        AllCheckpointer,
        PyrevolveCheckpointer,
    ],
)
def test_time_integration_with_parameter_partials(
    test_class, initial_time, parameter, butcher_tableau, checkpointing_implementation
):
    """Tests the partials of the time integration of the different components."""
    integration_control = StepTerminationIntegrationControl(0.001, 10, initial_time)
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


@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize("initial_time", [0.0, 1.0])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
@pytest.mark.parametrize(
    "checkpointing_implementation",
    [AllCheckpointer, PyrevolveCheckpointer],
)
def test_component_splitting_partials(
    initial_time, butcher_tableau, checkpointing_implementation
):
    """Tests the partials of the time integration of the problem that is split into
    multiple components."""
    integration_control_1 = StepTerminationIntegrationControl(0.001, 10, initial_time)
    integration_control_2 = StepTerminationIntegrationControl(0.001, 10, initial_time)

    time_integration_prob_1 = om.Problem()
    time_integration_prob_1.model.add_subsystem(
        "single_comp", TestComp4(integration_control=integration_control_1)
    )

    time_integration_prob_2 = om.Problem()
    time_integration_prob_2.model.add_subsystem(
        "first_comp",
        Testcomp51(integration_control=integration_control_2),
        promotes=["*"],
    )
    time_integration_prob_2.model.add_subsystem(
        "second_comp",
        Testcomp52(integration_control=integration_control_2),
        promotes=["*"],
    )

    time_integration_prob_2.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True
    )
    time_integration_prob_2.model.linear_solver = om.ScipyKrylov()

    runge_kutta_prob_1 = om.Problem()
    runge_kutta_prob_1.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob_1,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control_1,
            time_integration_quantities=["x"],
            checkpointing_type=checkpointing_implementation,
        ),
        promotes=["*"],
    )
    runge_kutta_prob_1.setup()
    runge_kutta_prob_1.run_model()

    data_1 = runge_kutta_prob_1.check_partials()

    runge_kutta_prob_2 = om.Problem()
    runge_kutta_prob_2.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob_2,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control_2,
            time_integration_quantities=["x", "y"],
            checkpointing_type=checkpointing_implementation,
        ),
        promotes=["*"],
    )

    runge_kutta_prob_2.setup()
    runge_kutta_prob_2.run_model()

    data_2 = runge_kutta_prob_2.check_partials()

    compare_split_and_unsplit_jacobian(data_1, data_2)


def compare_split_and_unsplit_jacobian(unsplit_jac_data, split_jac_data):
    """Compares data obtained from check_partials of a time integration between an
    unsplit and split version of the stage problem."""
    # row/column 0 in jacobian of matrix 1 corresponds to quantity "x" in problem 2
    # row/column 1 in jacobian of matrix 1 corresponds to quantity "y" in problem 2
    for i, name_i in enumerate(["x", "y"]):
        for j, name_j in enumerate(["x", "y"]):
            for mode in ["fwd", "rev"]:
                assert unsplit_jac_data["rk_integrator"][("x_final", "x_initial")][
                    f"J_{mode}"
                ][i, j] == pytest.approx(
                    split_jac_data["rk_integrator"][
                        (f"{name_i}_final", f"{name_j}_initial")
                    ][f"J_{mode}"][0, 0]
                )
