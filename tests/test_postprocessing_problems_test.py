"""Tests the postprocessing capabilities of the RungeKuttaIntegrator."""

import pytest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from rkopenmdao.integration_control import (
    IntegrationControl,
    StepTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    embedded_second_order_two_stage_sdirk as two_stage_dirk,
)
from rkopenmdao.checkpoint_interface.all_checkpointer import AllCheckpointer
from rkopenmdao.checkpoint_interface.pyrevolve_checkpointer import PyrevolveCheckpointer

from .test_components import TestComp4, Testcomp51, Testcomp52, TestComp6

from .test_components import solution_test4, solution_test6

from .test_postprocessing_problems import (
    create_negating_problem,
    create_accumulating_problem,
    create_squaring_problem,
    create_phase_problem,
)

from .test_postprocessing_problems import (
    negating_function,
    accumulating_function,
    squaring_function,
    phase_function,
)

# pylint: disable=too-many-arguments, too-many-statements


@pytest.mark.postporc
@pytest.mark.parametrize(
    "problem_creator, quantity_list",
    (
        [create_negating_problem, [("x", 1)]],
        [create_accumulating_problem, [("x", 1)]],
        [create_squaring_problem, [("x", 1)]],
        [create_phase_problem, [("x", 1)]],
        [create_negating_problem, [("x", 2)]],
        [create_accumulating_problem, [("x", 2)]],
        [create_squaring_problem, [("x", 2)]],
        [create_phase_problem, [("x", 2)]],
        [create_negating_problem, [("x", 1), ("y", 1)]],
        [create_accumulating_problem, [("x", 1), ("y", 1)]],
        [create_squaring_problem, [("x", 1), ("y", 1)]],
        [create_phase_problem, [("x", 1), ("y", 1)]],
        [create_negating_problem, [("x", 2), ("y", 1)]],
        [create_accumulating_problem, [("x", 2), ("y", 1)]],
        [create_squaring_problem, [("x", 2), ("y", 1)]],
        [create_phase_problem, [("x", 2), ("y", 1)]],
    ),
)
def test_postprocessing_problem_partials(problem_creator, quantity_list):
    """Tests the partials of the postprocessing problems themselves."""
    postproc_problem = problem_creator(quantity_list)
    postproc_problem.setup()

    postproc_problem.run_model()

    data = postproc_problem.check_partials(step=1e-7)

    assert_check_partials(data)


@pytest.mark.postproc
@pytest.mark.parametrize(
    """postprocessing_problem_creator, postproc_functor, postprocessing_quantity""",
    (
        [create_negating_problem, negating_function, ["negated_x"]],
        [create_accumulating_problem, accumulating_function, ["accumulated"]],
        [create_squaring_problem, squaring_function, ["squared_x"]],
        [create_phase_problem, phase_function, ["phase_x"]],
        [create_negating_problem, negating_function, ["negated_x"]],
        [create_accumulating_problem, accumulating_function, ["accumulated"]],
        [create_squaring_problem, squaring_function, ["squared_x"]],
        [create_phase_problem, phase_function, ["phase_x"]],
    ),
)
@pytest.mark.parametrize("initial_time", [0.0])
@pytest.mark.parametrize(
    "test_class, test_functor, quantity_size, initial_values",
    [
        [TestComp6, solution_test6, 1, np.array([1.0])],
        [TestComp4, solution_test4, 2, np.array([1.0, 1.0])],
    ],
)
@pytest.mark.parametrize("butcher_tableau", [implicit_euler, two_stage_dirk])
def test_postprocessing_after_time_integration(
    postprocessing_problem_creator,
    postproc_functor,
    quantity_size,
    test_class,
    test_functor,
    initial_time,
    initial_values,
    postprocessing_quantity,
    butcher_tableau,
):
    """Tests the postprocessing after time integration."""
    integration_control = StepTerminationIntegrationControl(0.001, 10, 0.0)
    postproc_problem = postprocessing_problem_creator([("x", quantity_size)])

    time_stage_problem = om.Problem()
    time_stage_problem.model.add_subsystem(
        "stage_comp",
        test_class(integration_control=integration_control),
        promotes=["*"],
    )

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_stage_problem,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            postprocessing_problem=postproc_problem,
            postprocessing_quantities=postprocessing_quantity,
        ),
        promotes=[(x + "_final", x) for x in postprocessing_quantity],
    )

    runge_kutta_prob.setup()

    runge_kutta_prob["rk_integrator.x_initial"] = initial_values

    runge_kutta_prob.run_model()

    result = np.array([runge_kutta_prob[postprocessing_quantity[0]]])

    assert postproc_functor(
        test_functor(initial_time + 0.01, np.array([initial_values]), initial_time)
    ) == pytest.approx(result, rel=1e-2)


@pytest.mark.postporc
@pytest.mark.parametrize(
    """postprocessing_problem_creator, postprocessing_quantity_1,
    postprocessing_quantities_2""",
    (
        [create_negating_problem, ["negated_x"], ["negated_x", "negated_y"]],
        [create_accumulating_problem, ["accumulated"], ["accumulated"]],
        [create_squaring_problem, ["squared_x"], ["squared_x", "squared_y"]],
        [create_phase_problem, ["phase_x"], ["phase_x", "phase_y"]],
    ),
)
@pytest.mark.parametrize("initial_values", [np.array([1.0, 1.0])])
@pytest.mark.parametrize("butcher_tableau", [implicit_euler, two_stage_dirk])
def test_postprocessing_after_time_integration_split(
    postprocessing_problem_creator,
    initial_values,
    postprocessing_quantity_1,
    postprocessing_quantities_2,
    butcher_tableau,
):
    """Tests whether postprocessing works the same when split over multiple
    components."""
    integration_control_1 = StepTerminationIntegrationControl(0.001, 10, 0.0)
    integration_control_2 = StepTerminationIntegrationControl(0.001, 10, 0.0)
    postproc_problem_1 = postprocessing_problem_creator([("x", 2)])
    postproc_problem_2 = postprocessing_problem_creator([("x", 1), ("y", 1)])

    time_stage_problem_1 = om.Problem()
    time_stage_problem_1.model.add_subsystem(
        "stage_comp",
        TestComp4(integration_control=integration_control_1),
        promotes=["*"],
    )

    runge_kutta_prob_1 = om.Problem()
    runge_kutta_prob_1.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_stage_problem_1,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control_1,
            time_integration_quantities=["x"],
            postprocessing_problem=postproc_problem_1,
            postprocessing_quantities=postprocessing_quantity_1,
        ),
        promotes=[(x + "_final", x) for x in postprocessing_quantity_1],
    )

    runge_kutta_prob_1.setup()
    runge_kutta_prob_1["rk_integrator.x_initial"] = initial_values

    runge_kutta_prob_1.run_model()

    result_1 = np.array([runge_kutta_prob_1[postprocessing_quantity_1[0]]])

    time_stage_problem_2 = om.Problem()
    time_stage_problem_2.model.add_subsystem(
        "stage_comp_1",
        Testcomp51(integration_control=integration_control_2),
        promotes=["*"],
    )
    time_stage_problem_2.model.add_subsystem(
        "stage_comp_2",
        Testcomp52(integration_control=integration_control_2),
        promotes=["*"],
    )

    time_stage_problem_2.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    time_stage_problem_2.model.linear_solver = om.ScipyKrylov()

    runge_kutta_prob_2 = om.Problem()
    runge_kutta_prob_2.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_stage_problem_2,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control_2,
            time_integration_quantities=["x", "y"],
            postprocessing_problem=postproc_problem_2,
            postprocessing_quantities=postprocessing_quantities_2,
        ),
        promotes=[(x + "_final", x) for x in postprocessing_quantities_2],
    )

    runge_kutta_prob_2.setup()

    runge_kutta_prob_2["rk_integrator.x_initial"] = initial_values[0]
    runge_kutta_prob_2["rk_integrator.y_initial"] = initial_values[1]

    runge_kutta_prob_2.run_model()

    result_2 = np.zeros_like(result_1)
    result_2[0][0] = runge_kutta_prob_2[postprocessing_quantities_2[0]]
    if len(postprocessing_quantities_2) > 1:
        result_2[0][1] = runge_kutta_prob_2[postprocessing_quantities_2[1]]

    assert result_1 == pytest.approx(result_2)


@pytest.mark.postporc
@pytest.mark.parametrize(
    """postprocessing_problem_creator, quantity_list, test_class,
    postprocessing_quantities""",
    (
        [create_negating_problem, [("x", 2)], TestComp4, ["negated_x"]],
        [create_negating_problem, [("x", 1)], TestComp6, ["negated_x"]],
        [create_accumulating_problem, [("x", 2)], TestComp4, ["accumulated"]],
        [create_accumulating_problem, [("x", 1)], TestComp6, ["accumulated"]],
        [create_squaring_problem, [("x", 2)], TestComp4, ["squared_x"]],
        [create_squaring_problem, [("x", 1)], TestComp6, ["squared_x"]],
        [create_phase_problem, [("x", 2)], TestComp4, ["phase_x"]],
        [create_phase_problem, [("x", 1)], TestComp6, ["phase_x"]],
    ),
)
@pytest.mark.parametrize("butcher_tableau", [implicit_euler, two_stage_dirk])
@pytest.mark.parametrize(
    "checkpointing_implementation", [AllCheckpointer, PyrevolveCheckpointer]
)
def test_postprocessing_after_time_integration_partials(
    postprocessing_problem_creator,
    quantity_list,
    test_class,
    postprocessing_quantities,
    butcher_tableau,
    checkpointing_implementation,
):
    """Tests partials of the postprocessing after time integration."""
    integration_control = StepTerminationIntegrationControl(0.001, 10, 0.0)
    postproc_problem = postprocessing_problem_creator(quantity_list)
    quantities = [quantity_tuple[0] for quantity_tuple in quantity_list]

    time_stage_problem = om.Problem()
    time_stage_problem.model.add_subsystem(
        "stage_comp",
        test_class(integration_control=integration_control),
        promotes=["*"],
    )

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_stage_problem,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=quantities,
            postprocessing_problem=postproc_problem,
            postprocessing_quantities=postprocessing_quantities,
            checkpointing_type=checkpointing_implementation,
        ),
    )

    runge_kutta_prob.setup()

    runge_kutta_prob.run_model()

    data = runge_kutta_prob.check_partials(
        step=1e-7
    )  # with default step, some tests don't pass

    assert_check_partials(data)


@pytest.mark.postporc
@pytest.mark.parametrize(
    """postprocessing_problem_creator, postprocessing_quantity_1,
    postprocessing_quantities_2""",
    (
        [create_negating_problem, ["negated_x"], ["negated_x", "negated_y"]],
        [create_accumulating_problem, ["accumulated"], ["accumulated"]],
        [create_squaring_problem, ["squared_x"], ["squared_x", "squared_y"]],
        [create_phase_problem, ["phase_x"], ["phase_x", "phase_y"]],
    ),
)
@pytest.mark.parametrize("butcher_tableau", [implicit_euler, two_stage_dirk])
@pytest.mark.parametrize(
    "checkpointing_implementation", [AllCheckpointer, PyrevolveCheckpointer]
)
def test_postprocessing_after_time_integration_split_partials(
    postprocessing_problem_creator,
    postprocessing_quantity_1,
    postprocessing_quantities_2,
    butcher_tableau,
    checkpointing_implementation,
):
    """Tests whether the partials of the postprocessing are the same for a split and
    unsplit problem."""
    integration_control_1 = StepTerminationIntegrationControl(0.01, 10, 0.0)
    integration_control_2 = StepTerminationIntegrationControl(0.01, 10, 0.0)
    postproc_problem_1 = postprocessing_problem_creator([("x", 2)])
    postproc_problem_2 = postprocessing_problem_creator([("x", 1), ("y", 1)])

    time_stage_problem_1 = om.Problem()
    time_stage_problem_1.model.add_subsystem(
        "stage_comp",
        TestComp4(integration_control=integration_control_1),
        promotes=["*"],
    )

    runge_kutta_prob_1 = om.Problem()
    runge_kutta_prob_1.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_stage_problem_1,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control_1,
            time_integration_quantities=["x"],
            postprocessing_problem=postproc_problem_1,
            postprocessing_quantities=postprocessing_quantity_1,
            checkpointing_type=checkpointing_implementation,
        ),
        promotes=[(x + "_final", x) for x in postprocessing_quantity_1],
    )

    runge_kutta_prob_1.setup()

    runge_kutta_prob_1.run_model()

    data_1 = runge_kutta_prob_1.check_partials()

    time_stage_problem_2 = om.Problem()
    time_stage_problem_2.model.add_subsystem(
        "stage_comp_1",
        Testcomp51(integration_control=integration_control_2),
        promotes=["*"],
    )
    time_stage_problem_2.model.add_subsystem(
        "stage_comp_2",
        Testcomp52(integration_control=integration_control_2),
        promotes=["*"],
    )

    time_stage_problem_2.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    time_stage_problem_2.model.linear_solver = om.ScipyKrylov()

    runge_kutta_prob_2 = om.Problem()
    runge_kutta_prob_2.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_stage_problem_2,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control_2,
            time_integration_quantities=["x", "y"],
            postprocessing_problem=postproc_problem_2,
            postprocessing_quantities=postprocessing_quantities_2,
            checkpointing_type=checkpointing_implementation,
        ),
        promotes=[(x + "_final", x) for x in postprocessing_quantities_2],
    )

    runge_kutta_prob_2.setup()

    runge_kutta_prob_2.run_model()

    data_2 = runge_kutta_prob_2.check_partials()

    compare_split_and_unsplit_jacobian_with_postproc(
        data_1, data_2, postprocessing_quantity_1, postprocessing_quantities_2
    )


def compare_split_and_unsplit_jacobian_with_postproc(
    unsplit_jac_data,
    split_jac_data,
    unsplit_postproc_quantities,
    split_postproc_quantities,
):
    """Compares data obtained from check_partials of a time integration between an
    unsplit and split version of the stage problem, in a case with postprocessing."""
    # row/column 0 in jacobian of matrix 1 corresponds to quantity "x" in problem 2
    # row/column 1 in jacobian of matrix 1 corresponds to quantity "y" in problem 2
    for mode in ["fwd", "rev"]:
        for j, name_j in enumerate(["x", "y"]):
            for i, name_i in enumerate(["x", "y"]):
                assert unsplit_jac_data["rk_integrator"][("x_final", "x_initial")][
                    f"J_{mode}"
                ][i, j] == pytest.approx(
                    split_jac_data["rk_integrator"][
                        (f"{name_i}_final", f"{name_j}_initial")
                    ][f"J_{mode}"][0, 0]
                )
            for i, name_i in enumerate(split_postproc_quantities):
                assert unsplit_jac_data["rk_integrator"][
                    (f"{unsplit_postproc_quantities[0]}_final", "x_initial")
                ][f"J_{mode}"][i, j] == pytest.approx(
                    split_jac_data["rk_integrator"][
                        (f"{name_i}_final", f"{name_j}_initial")
                    ][f"J_{mode}"][0, 0]
                )
