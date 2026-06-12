"""
A set of tests that verify whether the runge kutta methods conform their designated
order.
"""

import numpy as np
import pytest
import openmdao.api as om

from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.termination_criterion import PredefinedNumberOfSteps
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator

from rkopenmdao.butcher_tableaux import *  # pylint:disable=unused-wildcard-import,wildcard-import
from rkopenmdao.butcher_tableau import EmbeddedButcherTableau

from .test_components import TestComp1, solution_test1

NON_EMBEDDED_TABLEAUX = [
    explicit_euler,
    implicit_euler,
    implicit_midpoint,
    fifth_order_five_stage_sdirk,
    third_order_two_stage_sdirk,
    runge_kutta_four,
]

EMBEDDED_TABLEAUX = [
    embedded_heun_euler,
    embedded_second_order_two_stage_sdirk,
    embedded_second_order_three_stage_esdirk,
    embedded_third_order_three_stage_sdirk1,
    embedded_third_order_three_stage_sdirk2,
    embedded_third_order_three_stage_esdirk,
    embedded_third_order_four_stage_sdirk,
    embedded_third_order_second_weak_stage_order_four_stage_dirk,
    embedded_third_order_third_weak_stage_order_four_stage_dirk,
    embedded_third_order_four_stage_esdirk,
    embedded_third_order_five_stage_esdirk,
    embedded_fourth_order_four_stage_sdirk,
    embedded_fourth_order_six_stage_esdirk,
    embedded_fourth_order_third_weak_stage_order_six_stage_dirk,
    embedded_fourth_order_five_stage_esdirk,
    embedded_runge_kutta_fehlberg,
    embedded_fifth_order_six_stage_esdirk,
    embedded_fourth_order_five_stage_sdirk,
]


STEP_SIZES = np.array([0.2, 0.1, 0.05, 0.025])
FINAL_TIME = 1.0


def satisfies_order_conditions(a, b, c, order, atol=1e-10):
    """
    Check classical RK order conditions through Analytical verification.
    """

    one = np.ones_like(c)

    conditions = [
        (b @ one, 1.0),
    ]

    if order >= 2:
        conditions += [
            (b @ c, 1 / 2),
        ]

    if order >= 3:
        conditions += [
            (b @ (c**2), 1 / 3),
            (b @ (a @ c), 1 / 6),
        ]

    if order >= 4:
        conditions += [
            (b @ (c**3), 1 / 4),
            ((b * c) @ (a @ c), 1 / 8),
            (b @ (a @ (c**2)), 1 / 12),
            (b @ (a @ (a @ c)), 1 / 24),
        ]
    if order >= 5:
        conditions += [
            (b @ (c**4), 1 / 5),
            ((b * c**2) @ (a @ c), 1 / 10),
            ((b * c) @ (a @ (c**2)), 1 / 15),
            ((b * c) @ (a @ (a @ c)), 1 / 30),
            (b @ (a @ (c**3)), 1 / 20),
            (b @ (a @ (c * (a @ c))), 1 / 40),
            (b @ (a @ (a @ (c**2))), 1 / 60),
            (b @ (a @ (a @ (a @ c))), 1 / 120),
        ]

    for lhs, rhs in conditions:
        if not np.isclose(lhs, rhs, atol=atol, rtol=0):
            return False

    return True


@pytest.mark.parametrize(
    "tableau",
    NON_EMBEDDED_TABLEAUX + EMBEDDED_TABLEAUX,
    ids=[tableau.name for tableau in NON_EMBEDDED_TABLEAUX + EMBEDDED_TABLEAUX],
)
def test_stage_consistency(tableau):
    """
    Assert that the sum of each row of the butcher_matrix equals to the time step size
    """
    a = tableau.butcher_matrix
    c = tableau.butcher_time_stages

    np.testing.assert_allclose(
        a.sum(axis=1),
        c,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "tableau", EMBEDDED_TABLEAUX, ids=[tableau.name for tableau in EMBEDDED_TABLEAUX]
)
def test_embedded_main_method_order(tableau):
    """Assert that the main methods confrom the order conditions"""
    assert satisfies_order_conditions(
        tableau.butcher_matrix,
        tableau.butcher_weight_vector,
        tableau.butcher_time_stages,
        min(tableau.p, 5),
    )


@pytest.mark.parametrize(
    "tableau", EMBEDDED_TABLEAUX, ids=[tableau.name for tableau in EMBEDDED_TABLEAUX]
)
def test_embedded_secondary_method_order(tableau):
    """Assert that the embedded methods confrom the order conditions"""
    assert satisfies_order_conditions(
        tableau.butcher_matrix,
        tableau.butcher_adaptive_weights,
        tableau.butcher_time_stages,
        min(tableau.phat, 5),
    )


def observed_order(errors, step_sizes):
    """Estimate the convergence order"""
    return np.log(errors[:-1] / errors[1:]) / np.log(step_sizes[:-1] / step_sizes[1:])


def compute_errors(problem, tableau, step_sizes):
    """Compute global errors for a sequence of step sizes."""

    exact_solution = solution_test1(
        time=FINAL_TIME,
        initial_value=1,
        initial_time=0,
    )

    errors = []

    for h in step_sizes:
        n_steps = int(round(FINAL_TIME / h))

        integration_config = IntegrationConfig(
            False,
            PredefinedNumberOfSteps(n_steps),
            h,
        )

        rk_problem = om.Problem()
        rk_problem.model.add_subsystem(
            "rk_integrator",
            RungeKuttaIntegrator(
                time_stage_problem=problem,
                butcher_tableau=tableau,
                integration_config=integration_config,
                time_integration_quantities=["x"],
            ),
            promotes=["*"],
        )

        rk_problem.setup()
        rk_problem.run_model()

        numerical_solution = rk_problem["x_final"][0]
        errors.append(abs(exact_solution - numerical_solution))

    return np.asarray(errors)


def assert_order(errors, step_sizes, expected_order):
    """Assert that the observed convergence order matches expectation."""

    observed_orders = observed_order(errors, step_sizes)
    estimated_order = np.mean(observed_orders[-2:])

    assert estimated_order == pytest.approx(expected_order, abs=0.5)

    print(
        f"Observed order: {estimated_order:.3f}, " f"expected order: {expected_order}"
    )


@pytest.mark.parametrize(
    "tableau",
    NON_EMBEDDED_TABLEAUX + EMBEDDED_TABLEAUX,
    ids=[tableau.name for tableau in NON_EMBEDDED_TABLEAUX + EMBEDDED_TABLEAUX],
)
def test_tableau_order(tableau):
    """
    Verify that a Runge-Kutta method achieves its designed order.

    The test solves the TestComp1 initial value problem and compares
    the numerical solution at t = 1 against the analytical solution.

    For a sequence of decreasing step sizes h, the global error

        e_h = |y_h - y_exact|

    is computed and used to estimate the observed convergence rate

        p_obs = log(e_h / e_{h/2}) / log(2).

    The observed order should agree with the order declared by the
    Butcher tableau.
    """

    time_integration_problem = om.Problem()
    time_integration_problem.model.add_subsystem(
        "test_comp",
        TestComp1(),
    )

    errors = compute_errors(
        time_integration_problem,
        tableau,
        STEP_SIZES,
    )

    assert_order(errors, STEP_SIZES, tableau.p)


@pytest.mark.parametrize(
    "tableau",
    EMBEDDED_TABLEAUX,
    ids=[tableau.name for tableau in EMBEDDED_TABLEAUX],
)
def test_embedded_method_order(tableau):
    """Verify the order of the embedded Runge-Kutta scheme."""

    time_integration_problem = om.Problem()
    time_integration_problem.model.add_subsystem(
        "test_comp",
        TestComp1(),
    )

    errors = compute_errors(
        time_integration_problem,
        tableau.embedded_method,
        STEP_SIZES,
    )

    assert_order(errors, STEP_SIZES, tableau.phat)
