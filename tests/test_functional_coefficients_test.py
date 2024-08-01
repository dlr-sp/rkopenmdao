"""Tests to make sure that FunctionalCoefficients in the RungeKuttaIntegrator work
correctly."""

import pytest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    second_order_two_stage_sdirk as two_stage_dirk,
    runge_kutta_four,
)
from rkopenmdao.functional_coefficients import (
    AverageCoefficients,
    CompositeTrapezoidalCoefficients,
)

from rkopenmdao.checkpoint_interface.all_checkpointer import AllCheckpointer
from rkopenmdao.checkpoint_interface.pyrevolve_checkpointer import PyrevolveCheckpointer


from .test_functional_coefficients import (
    FifthStepOfQuantity,
)
from .test_components import TestComp6
from .test_components import Test6Solution
from .test_postprocessing_problems import (
    create_negating_problem,
)


@pytest.mark.lin_comb
@pytest.mark.parametrize("initial_time", [1.0])
@pytest.mark.parametrize("initial_value", [1.0])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
def test_average_functional(
    initial_time,
    initial_value,
    butcher_tableau,
):
    """Tests the averaging functional with various time integration schemes."""
    integration_control = IntegrationControl(initial_time, 10, 0.001)

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", TestComp6(integration_control=integration_control)
    )

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            functional_coefficients=AverageCoefficients(integration_control, ["x"]),
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["x_initial"] = initial_value

    runge_kutta_prob.run_model()

    result = runge_kutta_prob["x_functional"]

    expected = 0.0
    for i in range(0, 11):
        expected += (
            Test6Solution(initial_time + i / 1000, initial_value, initial_time) / 11
        )
    assert result[0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.lin_comb
@pytest.mark.parametrize("initial_time", [1.0])
@pytest.mark.parametrize("initial_value", [1.0])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
@pytest.mark.parametrize(
    "checkpointing_implementation", [AllCheckpointer, PyrevolveCheckpointer]
)
def test_average_functional_partials(
    initial_time, initial_value, butcher_tableau, checkpointing_implementation
):
    """Tests the partials of the averaging functional with various time
    integration schemes."""
    integration_control = IntegrationControl(initial_time, 10, 0.001)

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", TestComp6(integration_control=integration_control)
    )

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            functional_coefficients=AverageCoefficients(integration_control, ["x"]),
            checkpointing_type=checkpointing_implementation,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["x_initial"] = initial_value

    runge_kutta_prob.run_model()

    data = runge_kutta_prob.check_partials()

    assert_check_partials(data)


@pytest.mark.lin_comb
@pytest.mark.parametrize("initial_time", [1.0])
@pytest.mark.parametrize("initial_value", [1.0])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
def test_composite_trapezoidal_functional(
    initial_time,
    initial_value,
    butcher_tableau,
):
    """Tests the functional for the composite trapezoidal rule with various time
    integration schemes."""
    integration_control = IntegrationControl(initial_time, 1000, 0.001)

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", TestComp6(integration_control=integration_control)
    )

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            functional_coefficients=CompositeTrapezoidalCoefficients(
                integration_control, ["x"]
            ),
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["x_initial"] = initial_value

    runge_kutta_prob.run_model()

    result = runge_kutta_prob["x_functional"]

    expected = 1.5833333333333333333333
    assert result[0] == pytest.approx(expected, abs=1e-3)


@pytest.mark.lin_comb
@pytest.mark.parametrize("initial_time", [1.0])
@pytest.mark.parametrize("initial_value", [1.0])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
@pytest.mark.parametrize(
    "checkpointing_implementation", [AllCheckpointer, PyrevolveCheckpointer]
)
def test_composite_trapezoidal_functional_partials(
    initial_time, initial_value, butcher_tableau, checkpointing_implementation
):
    """Tests the partials of the functional for the composite trapezoidal rule with
    various time integration schemes."""
    integration_control = IntegrationControl(initial_time, 10, 0.001)

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", TestComp6(integration_control=integration_control)
    )

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            functional_coefficients=CompositeTrapezoidalCoefficients(
                integration_control, ["x"]
            ),
            checkpointing_type=checkpointing_implementation,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["x_initial"] = initial_value

    runge_kutta_prob.run_model()

    data = runge_kutta_prob.check_partials()

    assert_check_partials(data)


@pytest.mark.lin_comb
@pytest.mark.parametrize("initial_time", [1.0])
@pytest.mark.parametrize("initial_value", [1.0])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
def test_fifth_value_functional(
    initial_time,
    initial_value,
    butcher_tableau,
):
    """Tests the functional for the fifth step with various time integration schemes."""
    integration_control = IntegrationControl(initial_time, 10, 0.001)

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", TestComp6(integration_control=integration_control)
    )

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            functional_coefficients=FifthStepOfQuantity("x"),
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["x_initial"] = initial_value

    runge_kutta_prob.run_model()

    result = runge_kutta_prob["x_functional"]

    expected = Test6Solution(initial_time + 0.005, initial_value, initial_time)
    assert result[0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.lin_comb
@pytest.mark.parametrize("initial_time", [1.0])
@pytest.mark.parametrize("initial_value", [1.0])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
@pytest.mark.parametrize(
    "checkpointing_implementation", [AllCheckpointer, PyrevolveCheckpointer]
)
def test_fifth_value_functional_partials(
    initial_time, initial_value, butcher_tableau, checkpointing_implementation
):
    """Tests the partials of the functional for the fifth step with various time
    integration schemes."""
    integration_control = IntegrationControl(initial_time, 10, 0.001)

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", TestComp6(integration_control=integration_control)
    )

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            functional_coefficients=FifthStepOfQuantity("x"),
            checkpointing_type=checkpointing_implementation,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["x_initial"] = initial_value

    runge_kutta_prob.run_model()

    data = runge_kutta_prob.check_partials()

    assert_check_partials(data)


@pytest.mark.lin_comb
@pytest.mark.parametrize("initial_time", [1.0])
@pytest.mark.parametrize("initial_value", [1.0])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
def test_average_functional_with_postprocessing(
    initial_time,
    initial_value,
    butcher_tableau,
):
    """Tests the averaging functional with time integration and postprocessing."""
    integration_control = IntegrationControl(initial_time, 10, 0.001)

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", TestComp6(integration_control=integration_control)
    )

    negating_prob = create_negating_problem([("x", 1)])

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            postprocessing_problem=negating_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            postprocessing_quantities=["negated_x"],
            functional_coefficients=AverageCoefficients(
                integration_control, ["x", "negated_x"]
            ),
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["x_initial"] = initial_value

    runge_kutta_prob.run_model()

    result_1 = runge_kutta_prob["x_functional"]
    result_2 = runge_kutta_prob["negated_x_functional"]

    expected_1 = 0.0
    for i in range(0, 11):
        expected_1 += (
            Test6Solution(initial_time + i / 1000, initial_value, initial_time) / 11
        )
    expected_2 = -expected_1
    assert result_1[0] == pytest.approx(expected_1, abs=1e-5)
    assert result_2[0] == pytest.approx(expected_2, abs=1e-5)


@pytest.mark.lin_comb
@pytest.mark.parametrize("initial_time", [1.0])
@pytest.mark.parametrize("initial_value", [1.0])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
@pytest.mark.parametrize(
    "checkpointing_implementation", [AllCheckpointer, PyrevolveCheckpointer]
)
def test_average_functional_with_postprocessing_partials(
    initial_time, initial_value, butcher_tableau, checkpointing_implementation
):
    """Tests the partials of the averaging functional with time integration and
    postprocessing."""
    integration_control = IntegrationControl(initial_time, 10, 0.001)

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", TestComp6(integration_control=integration_control)
    )

    negating_prob = create_negating_problem([("x", 1)])

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            postprocessing_problem=negating_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            postprocessing_quantities=["negated_x"],
            functional_coefficients=AverageCoefficients(
                integration_control, ["x", "negated_x"]
            ),
            checkpointing_type=checkpointing_implementation,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["x_initial"] = initial_value

    runge_kutta_prob.run_model()

    data = runge_kutta_prob.check_partials()

    assert_check_partials(data)


@pytest.mark.lin_comb
@pytest.mark.parametrize("initial_time", [1.0])
@pytest.mark.parametrize("initial_value", [1.0])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
def test_average_functional_with_only_postprocessing(
    initial_time,
    initial_value,
    butcher_tableau,
):
    """Tests the averaging functional with only postprocessing."""
    integration_control = IntegrationControl(initial_time, 10, 0.001)

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", TestComp6(integration_control=integration_control)
    )

    negating_prob = create_negating_problem([("x", 1)])

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            postprocessing_problem=negating_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            postprocessing_quantities=["negated_x"],
            functional_coefficients=AverageCoefficients(
                integration_control, ["negated_x"]
            ),
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["x_initial"] = initial_value

    runge_kutta_prob.run_model()

    result = runge_kutta_prob["negated_x_functional"]

    expected = 0.0
    for i in range(0, 11):
        expected -= (
            Test6Solution(initial_time + i / 1000, initial_value, initial_time) / 11
        )

    assert result[0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.lin_comb
@pytest.mark.parametrize("initial_time", [1.0])
@pytest.mark.parametrize("initial_value", [1.0])
@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, two_stage_dirk, runge_kutta_four]
)
@pytest.mark.parametrize(
    "checkpointing_implementation", [AllCheckpointer, PyrevolveCheckpointer]
)
def test_average_functional_with_only_postprocessing_partials(
    initial_time, initial_value, butcher_tableau, checkpointing_implementation
):
    """Tests the partials of the averaging functional with only postprocessing."""
    integration_control = IntegrationControl(initial_time, 10, 0.001)

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", TestComp6(integration_control=integration_control)
    )

    negating_prob = create_negating_problem([("x", 1)])

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            postprocessing_problem=negating_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            postprocessing_quantities=["negated_x"],
            functional_coefficients=AverageCoefficients(
                integration_control, ["negated_x"]
            ),
            checkpointing_type=checkpointing_implementation,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()
    runge_kutta_prob["x_initial"] = initial_value

    runge_kutta_prob.run_model()

    data = runge_kutta_prob.check_partials()

    assert_check_partials(data)
