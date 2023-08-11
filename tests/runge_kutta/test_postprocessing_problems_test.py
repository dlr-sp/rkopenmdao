import pytest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl
from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import RungeKuttaIntegrator

from .test_components import TestComp4, TestComp5_1, TestComp5_2, TestComp6

from .test_components import Test4Solution, Test6Solution

from runge_kutta_openmdao.runge_kutta.butcher_tableaus import (
    implicit_euler,
    two_stage_dirk,
    runge_kutta_four,
)


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
    postproc_problem = problem_creator(quantity_list)
    postproc_problem.setup()

    postproc_problem.run_model()

    data = postproc_problem.check_partials(step=1e-7)

    assert_check_partials(data)


@pytest.mark.postproc
@pytest.mark.parametrize(
    """postprocessing_problem_creator, postproc_functor, quantity_size, test_class,
    test_functor, initial_time, initial_values, postprocessing_quantity, butcher_tableau""",
    (
        [
            create_negating_problem,
            negating_function,
            1,
            TestComp6,
            Test6Solution,
            0.0,
            np.array([1.0]),
            ["negated_x"],
            implicit_euler,
        ],
        [
            create_negating_problem,
            negating_function,
            1,
            TestComp6,
            Test6Solution,
            0.0,
            np.array([1.0]),
            ["negated_x"],
            two_stage_dirk,
        ],
        [
            create_accumulating_problem,
            accumulating_function,
            1,
            TestComp6,
            Test6Solution,
            0.0,
            np.array([1.0]),
            ["accumulated"],
            implicit_euler,
        ],
        [
            create_accumulating_problem,
            accumulating_function,
            1,
            TestComp6,
            Test6Solution,
            0.0,
            np.array([1.0]),
            ["accumulated"],
            two_stage_dirk,
        ],
        [
            create_squaring_problem,
            squaring_function,
            1,
            TestComp6,
            Test6Solution,
            0.0,
            np.array([1.0]),
            ["squared_x"],
            implicit_euler,
        ],
        [
            create_squaring_problem,
            squaring_function,
            1,
            TestComp6,
            Test6Solution,
            0.0,
            np.array([1.0]),
            ["squared_x"],
            two_stage_dirk,
        ],
        [
            create_phase_problem,
            phase_function,
            1,
            TestComp6,
            Test6Solution,
            0.0,
            np.array([1.0]),
            ["phase_x"],
            implicit_euler,
        ],
        [
            create_phase_problem,
            phase_function,
            1,
            TestComp6,
            Test6Solution,
            0.0,
            np.array([1.0]),
            ["phase_x"],
            two_stage_dirk,
        ],
        [
            create_negating_problem,
            negating_function,
            2,
            TestComp4,
            Test4Solution,
            0.0,
            np.array([1.0, 1.0]),
            ["negated_x"],
            implicit_euler,
        ],
        [
            create_negating_problem,
            negating_function,
            2,
            TestComp4,
            Test4Solution,
            0.0,
            np.array([1.0, 1.0]),
            ["negated_x"],
            two_stage_dirk,
        ],
        [
            create_accumulating_problem,
            accumulating_function,
            2,
            TestComp4,
            Test4Solution,
            0.0,
            np.array([1.0, 1.0]),
            ["accumulated"],
            implicit_euler,
        ],
        [
            create_accumulating_problem,
            accumulating_function,
            2,
            TestComp4,
            Test4Solution,
            0.0,
            np.array([1.0, 1.0]),
            ["accumulated"],
            two_stage_dirk,
        ],
        [
            create_squaring_problem,
            squaring_function,
            2,
            TestComp4,
            Test4Solution,
            0.0,
            np.array([1.0, 1.0]),
            ["squared_x"],
            implicit_euler,
        ],
        [
            create_squaring_problem,
            squaring_function,
            2,
            TestComp4,
            Test4Solution,
            0.0,
            np.array([1.0, 1.0]),
            ["squared_x"],
            two_stage_dirk,
        ],
        [
            create_phase_problem,
            phase_function,
            2,
            TestComp4,
            Test4Solution,
            0.0,
            np.array([1.0, 1.0]),
            ["phase_x"],
            implicit_euler,
        ],
        [
            create_phase_problem,
            phase_function,
            2,
            TestComp4,
            Test4Solution,
            0.0,
            np.array([1.0, 1.0]),
            ["phase_x"],
            two_stage_dirk,
        ],
    ),
)
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
    integration_control = IntegrationControl(0.0, 1000, 0.001)
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
        test_functor(initial_time + 1, np.array([initial_values]), initial_time)
    ) == pytest.approx(result, rel=1e-2)


@pytest.mark.postporc
@pytest.mark.parametrize(
    """postprocessing_problem_creator, postproc_functor, initial_time, initial_values,
      postprocessing_quantity_1, postprocessing_quantities_2, butcher_tableau""",
    (
        [
            create_negating_problem,
            negating_function,
            0.0,
            np.array([1.0, 1.0]),
            ["negated_x"],
            ["negated_x", "negated_y"],
            implicit_euler,
        ],
        [
            create_negating_problem,
            negating_function,
            0.0,
            np.array([1.0, 1.0]),
            ["negated_x"],
            ["negated_x", "negated_y"],
            two_stage_dirk,
        ],
        [
            create_accumulating_problem,
            accumulating_function,
            0.0,
            np.array([1.0, 1.0]),
            ["accumulated"],
            ["accumulated"],
            implicit_euler,
        ],
        [
            create_accumulating_problem,
            accumulating_function,
            0.0,
            np.array([1.0, 1.0]),
            ["accumulated"],
            ["accumulated"],
            two_stage_dirk,
        ],
        [
            create_squaring_problem,
            squaring_function,
            0.0,
            np.array([1.0, 1.0]),
            ["squared_x"],
            ["squared_x", "squared_y"],
            implicit_euler,
        ],
        [
            create_squaring_problem,
            squaring_function,
            0.0,
            np.array([1.0, 1.0]),
            ["squared_x"],
            ["squared_x", "squared_y"],
            two_stage_dirk,
        ],
        [
            create_phase_problem,
            phase_function,
            0.0,
            np.array([1.0, 1.0]),
            ["phase_x"],
            ["phase_x", "phase_y"],
            implicit_euler,
        ],
        [
            create_phase_problem,
            phase_function,
            0.0,
            np.array([1.0, 1.0]),
            ["phase_x"],
            ["phase_x", "phase_y"],
            two_stage_dirk,
        ],
    ),
)
def test_postprocessing_after_time_integration_split(
    postprocessing_problem_creator,
    postproc_functor,
    initial_time,
    initial_values,
    postprocessing_quantity_1,
    postprocessing_quantities_2,
    butcher_tableau,
):
    integration_control_1 = IntegrationControl(0.0, 1000, 0.001)
    integration_control_2 = IntegrationControl(0.0, 1000, 0.001)
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
        TestComp5_1(integration_control=integration_control_2),
        promotes=["*"],
    )
    time_stage_problem_2.model.add_subsystem(
        "stage_comp_2",
        TestComp5_2(integration_control=integration_control_2),
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
    "postprocessing_problem_creator, quantity_list, test_class, postprocessing_quantities, butcher_tableau",
    (
        [create_negating_problem, [("x", 2)], TestComp4, ["negated_x"], implicit_euler],
        [create_negating_problem, [("x", 1)], TestComp6, ["negated_x"], implicit_euler],
        [create_negating_problem, [("x", 2)], TestComp4, ["negated_x"], two_stage_dirk],
        [create_negating_problem, [("x", 1)], TestComp6, ["negated_x"], two_stage_dirk],
        [
            create_accumulating_problem,
            [("x", 2)],
            TestComp4,
            ["accumulated"],
            implicit_euler,
        ],
        [
            create_accumulating_problem,
            [("x", 1)],
            TestComp6,
            ["accumulated"],
            implicit_euler,
        ],
        [
            create_accumulating_problem,
            [("x", 2)],
            TestComp4,
            ["accumulated"],
            two_stage_dirk,
        ],
        [
            create_accumulating_problem,
            [("x", 1)],
            TestComp6,
            ["accumulated"],
            two_stage_dirk,
        ],
        [create_squaring_problem, [("x", 2)], TestComp4, ["squared_x"], implicit_euler],
        [create_squaring_problem, [("x", 1)], TestComp6, ["squared_x"], implicit_euler],
        [create_squaring_problem, [("x", 2)], TestComp4, ["squared_x"], two_stage_dirk],
        [create_squaring_problem, [("x", 1)], TestComp6, ["squared_x"], two_stage_dirk],
        [create_phase_problem, [("x", 2)], TestComp4, ["phase_x"], implicit_euler],
        [create_phase_problem, [("x", 1)], TestComp6, ["phase_x"], implicit_euler],
        [create_phase_problem, [("x", 2)], TestComp4, ["phase_x"], two_stage_dirk],
        [create_phase_problem, [("x", 1)], TestComp6, ["phase_x"], two_stage_dirk],
    ),
)
def test_postprocessing_after_time_integration_partials(
    postprocessing_problem_creator,
    quantity_list,
    test_class,
    postprocessing_quantities,
    butcher_tableau,
):
    integration_control = IntegrationControl(0.0, 1000, 0.001)
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
        ),
    )

    runge_kutta_prob.setup()

    runge_kutta_prob.run_model()

    data = runge_kutta_prob.check_partials(
        step=1e-7
    )  # with default step, some tests don't pass

    assert_check_partials(data)


# TODO: check whether this test works (by adding test cases


@pytest.mark.postporc
@pytest.mark.parametrize(
    """postprocessing_problem_creator, postprocessing_quantity_1, postprocessing_quantities_2, butcher_tableau""",
    (
        [
            create_negating_problem,
            ["negated_x"],
            ["negated_x", "negated_y"],
            implicit_euler,
        ],
        [
            create_negating_problem,
            ["negated_x"],
            ["negated_x", "negated_y"],
            two_stage_dirk,
        ],
        [
            create_accumulating_problem,
            ["accumulated"],
            ["accumulated"],
            implicit_euler,
        ],
        [
            create_accumulating_problem,
            ["accumulated"],
            ["accumulated"],
            two_stage_dirk,
        ],
        [
            create_squaring_problem,
            ["squared_x"],
            ["squared_x", "squared_y"],
            implicit_euler,
        ],
        [
            create_squaring_problem,
            ["squared_x"],
            ["squared_x", "squared_y"],
            two_stage_dirk,
        ],
        [
            create_phase_problem,
            ["phase_x"],
            ["phase_x", "phase_y"],
            implicit_euler,
        ],
        [
            create_phase_problem,
            ["phase_x"],
            ["phase_x", "phase_y"],
            two_stage_dirk,
        ],
    ),
)
def test_postprocessing_after_time_integration_split_partials(
    postprocessing_problem_creator,
    postprocessing_quantity_1,
    postprocessing_quantities_2,
    butcher_tableau,
):
    integration_control_1 = IntegrationControl(0.0, 100, 0.01)
    integration_control_2 = IntegrationControl(0.0, 100, 0.01)
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

    runge_kutta_prob_1.run_model()

    data_1 = runge_kutta_prob_1.check_partials()

    time_stage_problem_2 = om.Problem()
    time_stage_problem_2.model.add_subsystem(
        "stage_comp_1",
        TestComp5_1(integration_control=integration_control_2),
        promotes=["*"],
    )
    time_stage_problem_2.model.add_subsystem(
        "stage_comp_2",
        TestComp5_2(integration_control=integration_control_2),
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

    runge_kutta_prob_2.run_model()

    data_2 = runge_kutta_prob_2.check_partials()

    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_fwd"][
        0, 0
    ] == pytest.approx(data_2["rk_integrator"][("x_final", "x_initial")]["J_fwd"][0, 0])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_fwd"][
        0, 1
    ] == pytest.approx(data_2["rk_integrator"][("x_final", "y_initial")]["J_fwd"][0, 0])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_fwd"][
        1, 0
    ] == pytest.approx(data_2["rk_integrator"][("y_final", "x_initial")]["J_fwd"][0, 0])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_fwd"][
        1, 1
    ] == pytest.approx(data_2["rk_integrator"][("y_final", "y_initial")]["J_fwd"][0, 0])

    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_rev"][
        0, 0
    ] == pytest.approx(data_2["rk_integrator"][("x_final", "x_initial")]["J_rev"][0, 0])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_rev"][
        0, 1
    ] == pytest.approx(data_2["rk_integrator"][("x_final", "y_initial")]["J_rev"][0, 0])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_rev"][
        1, 0
    ] == pytest.approx(data_2["rk_integrator"][("y_final", "x_initial")]["J_rev"][0, 0])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_rev"][
        1, 1
    ] == pytest.approx(data_2["rk_integrator"][("y_final", "y_initial")]["J_rev"][0, 0])

    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_fwd"][
        0, 0
    ] == pytest.approx(data_2["rk_integrator"][("x_final", "x_initial")]["J_fwd"])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_fwd"][
        0, 1
    ] == pytest.approx(data_2["rk_integrator"][("x_final", "y_initial")]["J_fwd"])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_fwd"][
        1, 0
    ] == pytest.approx(data_2["rk_integrator"][("y_final", "x_initial")]["J_fwd"])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_fwd"][
        1, 1
    ] == pytest.approx(data_2["rk_integrator"][("y_final", "y_initial")]["J_fwd"])

    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_rev"][
        0, 0
    ] == pytest.approx(data_2["rk_integrator"][("x_final", "x_initial")]["J_rev"])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_rev"][
        0, 1
    ] == pytest.approx(data_2["rk_integrator"][("x_final", "y_initial")]["J_rev"])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_rev"][
        1, 0
    ] == pytest.approx(data_2["rk_integrator"][("y_final", "x_initial")]["J_rev"])
    assert data_1["rk_integrator"][("x_final", "x_initial")]["J_rev"][
        1, 1
    ] == pytest.approx(data_2["rk_integrator"][("y_final", "y_initial")]["J_rev"])

    if len(postprocessing_quantities_2) > 1:
        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_fwd"][0, 0] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[0] + "_final", "x_initial")
            ]["J_fwd"]
        )
        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_fwd"][0, 1] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[0] + "_final", "y_initial")
            ]["J_fwd"]
        )
        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_fwd"][1, 0] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[1] + "_final", "x_initial")
            ]["J_fwd"]
        )
        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_fwd"][1, 1] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[1] + "_final", "y_initial")
            ]["J_fwd"]
        )

        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_rev"][0, 0] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[0] + "_final", "x_initial")
            ]["J_rev"]
        )
        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_rev"][0, 1] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[0] + "_final", "y_initial")
            ]["J_rev"]
        )
        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_rev"][1, 0] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[1] + "_final", "x_initial")
            ]["J_rev"]
        )
        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_rev"][1, 1] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[1] + "_final", "y_initial")
            ]["J_rev"]
        )
    else:
        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_fwd"][0, 0] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[0] + "_final", "x_initial")
            ]["J_fwd"]
        )
        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_fwd"][0, 1] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[0] + "_final", "y_initial")
            ]["J_fwd"]
        )

        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_rev"][0, 0] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[0] + "_final", "x_initial")
            ]["J_rev"]
        )
        assert data_1["rk_integrator"][
            (postprocessing_quantity_1[0] + "_final", "x_initial")
        ]["J_rev"][0, 1] == pytest.approx(
            data_2["rk_integrator"][
                (postprocessing_quantities_2[0] + "_final", "y_initial")
            ]["J_rev"]
        )
