# pylint: disable=protected-access
# Lots of inner functions are tested here

import numpy as np
import pytest

from rkopenmdao.butcher_tableau import ButcherTableau, EmbeddedButcherTableau
from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk,
    embedded_second_order_three_stage_esdirk,
    implicit_euler,
    fourth_order_five_stage_sdirk,
    runge_kutta_four,
)


from rkopenmdao.discretized_ode.discretized_ode import (
    DiscretizedODE,
    DiscretizedODEResultState,
)
from rkopenmdao.time_discretization.stage_ordered_runge_kutta_discretization import (
    StageOrderedRungeKuttaDiscretization,
    StageOrderedEmbeddedRungeKuttaDiscretization,
)
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationStartingValues,
)

from .odes import IdentityODE, RootODE, TimeScaledIdentityODE
from .utils.time_discretization_test_utils import convergence_study


@pytest.fixture(
    name="ode_collection",
    params=[
        (
            IdentityODE(),
            lambda starting_values, step_size: starting_values.initial_values
            * np.exp(step_size),
            np.inf,
        ),
        (
            TimeScaledIdentityODE(),
            lambda starting_values, step_size: starting_values.initial_values
            * np.exp(0.5 * step_size * (2 * starting_values.initial_time + step_size)),
            np.inf,
        ),
        (
            RootODE(),
            lambda starting_values, step_size: (
                0.5 * step_size + starting_values.initial_values
            )
            ** 2,
            np.inf,
        ),
    ],
)
def fixture_ode_collection(request):
    ode, analytical_solution, order_barrier = request.param
    starting_values = TimeDiscretizationStartingValues(
        1.0, np.ones(ode.get_state_size()), np.ones(ode.get_independent_input_size())
    )
    return {
        "ode": ode,
        "starting_values": starting_values,
        "analytical_solution": analytical_solution,
        "order_barrier": order_barrier,
    }


@pytest.mark.parametrize(
    "butcher_tableau, base_step_size",
    [
        (implicit_euler, 0.1),
        (embedded_second_order_two_stage_sdirk, 0.1),
        (runge_kutta_four, 0.1),
        (fourth_order_five_stage_sdirk, 0.1),
    ],
)
def test_convergence(
    butcher_tableau: ButcherTableau,
    base_step_size: float,
    ode_collection: dict,
):
    discretization = StageOrderedRungeKuttaDiscretization(butcher_tableau)
    order_barrier = ode_collection.pop("order_barrier")
    norm_comparisons = convergence_study(
        discretization=discretization,
        base_step_size=base_step_size,
        num_calculations=4,
        **ode_collection
    )
    norm_comparisons = np.log2(norm_comparisons)

    expected_order = min(order_barrier, butcher_tableau.p + 1)

    # Hitting the order exactly is not going to happen all the time.
    # Instead assert that the order is at least the expected one times
    # a safety factor.
    assert np.all(norm_comparisons >= 0.95 * expected_order)


@pytest.mark.parametrize(
    "butcher_tableau, base_step_size",
    [
        (embedded_second_order_two_stage_sdirk, 0.01),
        (embedded_second_order_three_stage_esdirk, 0.01),
    ],
)
def test_error_estimate_convergence(
    butcher_tableau: EmbeddedButcherTableau,
    base_step_size: float,
    ode_collection: dict,
):
    discretization = StageOrderedEmbeddedRungeKuttaDiscretization(butcher_tableau)
    order_barrier = ode_collection.pop("order_barrier")
    ode: DiscretizedODE = ode_collection["ode"]
    starting_values = ode_collection["starting_values"]
    initial_state = discretization.time_discretization_starting_scheme(
        ode, starting_values, 0.0
    )

    error_norms = np.zeros(4)
    for i in range(4):
        step_size = base_step_size * 0.5**i
        embedded_state = discretization.compute_step(
            ode_collection["ode"], initial_state, step_size
        )
        error = embedded_state.error_estimate
        error_norms[i] = ode.compute_state_norm(
            DiscretizedODEResultState(np.zeros(0), error, np.zeros(0))
        )
    norm_comparisons = np.zeros(3)
    for i in range(3):
        norm_comparisons[i] = error_norms[i] / error_norms[i + 1]
    norm_comparisons = np.log2(norm_comparisons)
    expected_order = min(order_barrier, butcher_tableau.p)
    assert np.all(norm_comparisons >= 0.95 * expected_order)
