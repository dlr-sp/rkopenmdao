import openmdao.api as om
import pytest
import numpy as np

from itertools import product
from openmdao.utils.assert_utils import assert_check_partials

from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl
from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import RungeKuttaIntegrator
from runge_kutta_openmdao.runge_kutta.butcher_tableaus import (
    implicit_euler,
    two_stage_dirk,
    runge_kutta_four,
)


from .test_components import (
    TestComp1,
    TestComp2,
    TestComp3,
    TestComp4,
    TestComp5_1,
    TestComp5_2,
    TestComp6,
    TestComp7,
)

from .test_components import (
    Test1Solution,
    Test2Solution,
    Test3Solution,
    Test4Solution,
    Test6Solution,
    Test7Solution,
)

test_comp_class_list = [
    TestComp1,
    TestComp2,
    TestComp3,
    TestComp4,
    TestComp5_1,
    TestComp5_2,
    TestComp6,
    TestComp7,
]
times = np.linspace(1.0, 9.0, 3)
butcher_diagonal_elements = np.linspace(0.0, 1.0, 5)


@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize(
    "test_class, time, butcher_diagonal_element",
    product(test_comp_class_list, times, butcher_diagonal_elements),
)
def test_component_partials(test_class, time, butcher_diagonal_element):
    integration_control = IntegrationControl(0.0, 1, 0.1)
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
    "test_class, test_functor, initial_time, initial_values, butcher_tableau, quantities",
    (
        [TestComp1, Test1Solution, 0.0, np.array([1.0]), implicit_euler, ["x"]],
        [TestComp1, Test1Solution, 1.0, np.array([1.0]), implicit_euler, ["x"]],
        [TestComp1, Test1Solution, 0.0, np.array([1.0]), two_stage_dirk, ["x"]],
        [TestComp1, Test1Solution, 1.0, np.array([1.0]), two_stage_dirk, ["x"]],
        [TestComp1, Test1Solution, 0.0, np.array([1.0]), runge_kutta_four, ["x"]],
        [TestComp1, Test1Solution, 1.0, np.array([1.0]), runge_kutta_four, ["x"]],
        [TestComp2, Test2Solution, 0.0, np.array([1.0]), implicit_euler, ["x"]],
        [TestComp2, Test2Solution, 1.0, np.array([1.0]), implicit_euler, ["x"]],
        [TestComp2, Test2Solution, 0.0, np.array([1.0]), two_stage_dirk, ["x"]],
        [TestComp2, Test2Solution, 1.0, np.array([1.0]), two_stage_dirk, ["x"]],
        [TestComp2, Test2Solution, 0.0, np.array([1.0]), runge_kutta_four, ["x"]],
        [TestComp2, Test2Solution, 1.0, np.array([1.0]), runge_kutta_four, ["x"]],
        [TestComp3, Test3Solution, 0.0, np.array([1.0]), implicit_euler, ["x"]],
        [TestComp3, Test3Solution, 1.0, np.array([1.0]), implicit_euler, ["x"]],
        [TestComp3, Test3Solution, 0.0, np.array([1.0]), two_stage_dirk, ["x"]],
        [TestComp3, Test3Solution, 1.0, np.array([1.0]), two_stage_dirk, ["x"]],
        [TestComp3, Test3Solution, 0.0, np.array([1.0]), runge_kutta_four, ["x"]],
        [TestComp3, Test3Solution, 1.0, np.array([1.0]), runge_kutta_four, ["x"]],
        [
            TestComp4,
            Test4Solution,
            0.0,
            np.array([[1.0, 1.0]]),
            implicit_euler,
            ["x"],
        ],
        [
            TestComp4,
            Test4Solution,
            1.0,
            np.array([[1.0, 1.0]]),
            implicit_euler,
            ["x"],
        ],
        [
            TestComp4,
            Test4Solution,
            0.0,
            np.array([[1.0, 1.0]]),
            two_stage_dirk,
            ["x"],
        ],
        [
            TestComp4,
            Test4Solution,
            1.0,
            np.array([[1.0, 1.0]]),
            two_stage_dirk,
            ["x"],
        ],
        [
            TestComp4,
            Test4Solution,
            0.0,
            np.array([[1.0, 1.0]]),
            runge_kutta_four,
            ["x"],
        ],
        [
            TestComp4,
            Test4Solution,
            1.0,
            np.array([[1.0, 1.0]]),
            runge_kutta_four,
            ["x"],
        ],
        [TestComp6, Test6Solution, 0.0, np.array([1.0]), implicit_euler, ["x"]],
        [TestComp6, Test6Solution, 1.0, np.array([1.0]), implicit_euler, ["x"]],
        [TestComp6, Test6Solution, 0.0, np.array([1.0]), two_stage_dirk, ["x"]],
        [TestComp6, Test6Solution, 1.0, np.array([1.0]), two_stage_dirk, ["x"]],
        [TestComp6, Test6Solution, 0.0, np.array([1.0]), runge_kutta_four, ["x"]],
        [TestComp6, Test6Solution, 1.0, np.array([1.0]), runge_kutta_four, ["x"]],
        [TestComp7, Test7Solution, 1.0, np.array([1.0]), implicit_euler, ["x"]],
        [TestComp7, Test7Solution, 2.0, np.array([1.0]), implicit_euler, ["x"]],
        [TestComp7, Test7Solution, 1.0, np.array([1.0]), two_stage_dirk, ["x"]],
        [TestComp7, Test7Solution, 2.0, np.array([1.0]), two_stage_dirk, ["x"]],
        [TestComp7, Test7Solution, 1.0, np.array([1.0]), runge_kutta_four, ["x"]],
        [TestComp7, Test7Solution, 2.0, np.array([1.0]), runge_kutta_four, ["x"]],
    ),
)
def test_component_integration(
    test_class, test_functor, initial_time, initial_values, butcher_tableau, quantities
):
    integration_control = IntegrationControl(initial_time, 1000, 0.001)
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", test_class(integration_control=integration_control)
    )

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

    # relatively coarse, but this isn't supposed to test the accuracy
    # and instead just to make sure the solution is in the right region
    assert test_functor(
        initial_time + 1,
        initial_values,
        initial_time,
    ) == pytest.approx(result, rel=1e-2)


@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize(
    "initial_time, initial_values, butcher_tableau",
    (
        [0.0, np.array([1.0, 1.0]), implicit_euler],
        [0.0, np.array([1.0, 1.0]), two_stage_dirk],
        [0.0, np.array([1.0, 1.0]), runge_kutta_four],
        [1.0, np.array([1.0, 1.0]), implicit_euler],
        [1.0, np.array([1.0, 1.0]), two_stage_dirk],
        [1.0, np.array([1.0, 1.0]), runge_kutta_four],
        [0.0, np.array([1.0, 0.0]), implicit_euler],
        [0.0, np.array([1.0, 0.0]), two_stage_dirk],
        [0.0, np.array([1.0, 0.0]), runge_kutta_four],
    ),
)
def test_component_splitting(initial_time, initial_values, butcher_tableau):
    integration_control_1 = IntegrationControl(initial_time, 1000, 0.001)
    integration_control_2 = IntegrationControl(initial_time, 1000, 0.001)

    time_integration_prob_1 = om.Problem()
    time_integration_prob_1.model.add_subsystem(
        "single_comp", TestComp4(integration_control=integration_control_1)
    )

    time_integration_prob_2 = om.Problem()
    time_integration_prob_2.model.add_subsystem(
        "first_comp",
        TestComp5_1(integration_control=integration_control_2),
        promotes=["x_stage", "y_stage"],
    )
    time_integration_prob_2.model.add_subsystem(
        "second_comp",
        TestComp5_2(integration_control=integration_control_2),
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
    "test_class, initial_time, butcher_tableau, quantities",
    (
        [TestComp1, 0.0, implicit_euler, ["x"]],
        [TestComp1, 1.0, implicit_euler, ["x"]],
        [TestComp1, 0.0, two_stage_dirk, ["x"]],
        [TestComp1, 1.0, two_stage_dirk, ["x"]],
        [TestComp1, 0.0, runge_kutta_four, ["x"]],
        [TestComp1, 1.0, runge_kutta_four, ["x"]],
        [TestComp2, 0.0, implicit_euler, ["x"]],
        [TestComp2, 1.0, implicit_euler, ["x"]],
        [TestComp2, 0.0, two_stage_dirk, ["x"]],
        [TestComp2, 1.0, two_stage_dirk, ["x"]],
        [TestComp2, 0.0, runge_kutta_four, ["x"]],
        [TestComp2, 1.0, runge_kutta_four, ["x"]],
        [TestComp3, 0.0, implicit_euler, ["x"]],
        [TestComp3, 1.0, implicit_euler, ["x"]],
        [TestComp3, 0.0, two_stage_dirk, ["x"]],
        [TestComp3, 1.0, two_stage_dirk, ["x"]],
        [TestComp3, 0.0, runge_kutta_four, ["x"]],
        [TestComp3, 1.0, runge_kutta_four, ["x"]],
        [TestComp4, 0.0, implicit_euler, ["x"]],
        [TestComp4, 1.0, implicit_euler, ["x"]],
        [TestComp4, 0.0, two_stage_dirk, ["x"]],
        [TestComp4, 1.0, two_stage_dirk, ["x"]],
        [TestComp4, 0.0, runge_kutta_four, ["x"]],
        [TestComp4, 1.0, runge_kutta_four, ["x"]],
        [TestComp6, 0.0, implicit_euler, ["x"]],
        [TestComp6, 1.0, implicit_euler, ["x"]],
        [TestComp6, 0.0, two_stage_dirk, ["x"]],
        [TestComp6, 1.0, two_stage_dirk, ["x"]],
        [TestComp6, 0.0, runge_kutta_four, ["x"]],
        [TestComp6, 1.0, runge_kutta_four, ["x"]],
        [TestComp7, 1.0, implicit_euler, ["x"]],
        [TestComp7, 2.0, implicit_euler, ["x"]],
        [TestComp7, 1.0, two_stage_dirk, ["x"]],
        [TestComp7, 2.0, two_stage_dirk, ["x"]],
        [TestComp7, 1.0, runge_kutta_four, ["x"]],
        [TestComp7, 2.0, runge_kutta_four, ["x"]],
    ),
)
def test_time_integration_partials(
    test_class, initial_time, butcher_tableau, quantities
):
    integration_control = IntegrationControl(initial_time, 10, 0.001)
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", test_class(integration_control=integration_control)
    )
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

    runge_kutta_prob.run_model()

    data = runge_kutta_prob.check_partials()

    assert_check_partials(data)


@pytest.mark.rk
@pytest.mark.rk_openmdao
@pytest.mark.parametrize(
    "initial_time, butcher_tableau",
    (
        [0.0, implicit_euler],
        [0.0, two_stage_dirk],
        [0.0, runge_kutta_four],
        [1.0, implicit_euler],
        [1.0, two_stage_dirk],
        [1.0, runge_kutta_four],
        [0.0, implicit_euler],
        [0.0, two_stage_dirk],
        [0.0, runge_kutta_four],
    ),
)
def test_component_splitting_partials(initial_time, butcher_tableau):
    integration_control_1 = IntegrationControl(initial_time, 10, 0.001)
    integration_control_2 = IntegrationControl(initial_time, 10, 0.001)

    time_integration_prob_1 = om.Problem()
    time_integration_prob_1.model.add_subsystem(
        "single_comp", TestComp4(integration_control=integration_control_1)
    )

    time_integration_prob_2 = om.Problem()
    time_integration_prob_2.model.add_subsystem(
        "first_comp",
        TestComp5_1(integration_control=integration_control_2),
        promotes=["*"],
    )
    time_integration_prob_2.model.add_subsystem(
        "second_comp",
        TestComp5_2(integration_control=integration_control_2),
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
        ),
        promotes=["*"],
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
