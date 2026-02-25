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
from rkopenmdao.time_discretization.runge_kutta_discretization_state import (
    RungeKuttaDiscretizationState,
)
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationStartingValues,
    TimeDiscretizationFinalizationValues,
)

from .odes import IdentityODE, RootODE, TimeScaledIdentityODE, TimeODE, ParameterODE
from .utils.time_discretization_test_utils import (
    convergence_study,
    step_duality,
    starting_scheme_duality,
    finalization_scheme_duality,
)


def state_dot_func(
    state1: RungeKuttaDiscretizationState, state2: RungeKuttaDiscretizationState
) -> float:
    """
    Compute the inner product between two RungeKuttaDiscretizationState objects.

    The inner product is computed as the sum of element-wise products of all
    numpy arrays in the states.

    Parameters
    ----------
    state1 : RungeKuttaDiscretizationState
        First state
    state2 : RungeKuttaDiscretizationState
        Second state

    Returns
    -------
    float
        The inner product <state1, state2>
    """
    dot_product = 0.0

    # ODE state arrays
    dot_product += np.dot(state1.final_state, state2.final_state)

    # Independent inputs/outputs
    dot_product += np.dot(state1.independent_inputs, state2.independent_inputs)
    dot_product += np.dot(
        state1.final_independent_outputs, state2.final_independent_outputs
    )

    # Time-related arrays
    dot_product += np.dot(state1.final_time, state2.final_time)

    return dot_product


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
    "butcher_tableau",
    [
        implicit_euler,
        runge_kutta_four,
        fourth_order_five_stage_sdirk,
        embedded_second_order_two_stage_sdirk,
        embedded_second_order_three_stage_esdirk,
    ],
)
@pytest.mark.parametrize(
    "ode",
    [IdentityODE(), TimeScaledIdentityODE(), RootODE(), TimeODE(), ParameterODE()],
)
@pytest.mark.parametrize("seed", range(5))
def test_step_duality(butcher_tableau: ButcherTableau, ode: DiscretizedODE, seed: int):
    """Test duality between compute_step_derivative and compute_step_adjoint_derivative."""
    discretization = StageOrderedRungeKuttaDiscretization(butcher_tableau)

    # Create a state at which to evaluate the Jacobian
    starting_values = TimeDiscretizationStartingValues(
        initial_time=0.0,
        initial_values=np.ones(ode.get_state_size()),
        independent_inputs=np.ones(ode.get_independent_input_size()),
    )
    state = discretization.time_discretization_starting_scheme(
        ode, starting_values, 0.0
    )

    # Compute a step to get a state at t=step_size
    step_size = 0.1
    state = discretization.compute_step(ode, state, step_size)

    generator = np.random.default_rng(seed)

    # Create perturbations
    input_state_derivative = discretization.create_empty_discretization_state(ode)
    input_state_derivative.final_state = generator.random(ode.get_state_size())
    input_state_derivative.final_time = generator.random(1)
    input_state_derivative.independent_inputs = generator.random(
        ode.get_independent_input_size()
    )
    input_state_derivative.final_independent_outputs = generator.random(
        ode.get_independent_output_size()
    )

    output_state_adjoint = discretization.create_empty_discretization_state(ode)
    output_state_adjoint.final_state = generator.random(ode.get_state_size())
    output_state_adjoint.final_time = generator.random(1)
    output_state_adjoint.independent_inputs = generator.random(
        ode.get_independent_input_size()
    )
    output_state_adjoint.final_independent_outputs = generator.random(
        ode.get_independent_output_size()
    )

    # Test duality
    dual_fwd, dual_rev = step_duality(
        discretization=discretization,
        ode=ode,
        state=state,
        input_state_derivative=input_state_derivative,
        output_state_adjoint=output_state_adjoint,
        step_size=step_size,
        state_dot_func=state_dot_func,
    )

    # The duality relationship should hold: <dy, J*dx> = <J^T*dy, dx>
    np.testing.assert_allclose(dual_fwd, dual_rev, rtol=1e-10)


@pytest.mark.parametrize(
    "butcher_tableau",
    [
        implicit_euler,
        runge_kutta_four,
        fourth_order_five_stage_sdirk,
    ],
)
def test_starting_scheme_duality(butcher_tableau: ButcherTableau):
    """Test duality between starting scheme derivative and adjoint derivative."""
    discretization = StageOrderedRungeKuttaDiscretization(butcher_tableau)
    ode = IdentityODE()

    starting_values = TimeDiscretizationStartingValues(
        initial_time=0.0,
        initial_values=np.ones(ode.get_state_size()),
        independent_inputs=np.ones(ode.get_independent_input_size()),
    )

    # Create perturbations
    starting_value_derivative = TimeDiscretizationStartingValues(
        initial_time=0.01,
        initial_values=np.array([0.1]),
        independent_inputs=np.ones(ode.get_independent_input_size()),
    )

    state_adjoint = discretization.create_empty_discretization_state(ode)
    state_adjoint.final_state = np.array([0.2])

    step_size = 0.1

    # Test duality
    dual_fwd, dual_rev = starting_scheme_duality(
        discretization=discretization,
        ode=ode,
        starting_values=starting_values,
        starting_value_derivative=starting_value_derivative,
        state_adjoint=state_adjoint,
        step_size=step_size,
        state_dot_func=state_dot_func,
    )

    # The duality relationship should hold
    np.testing.assert_allclose(dual_fwd, dual_rev, rtol=1e-10)


@pytest.mark.parametrize(
    "butcher_tableau",
    [
        implicit_euler,
        runge_kutta_four,
        fourth_order_five_stage_sdirk,
    ],
)
def test_finalization_scheme_duality(
    butcher_tableau: ButcherTableau,
):
    """Test duality between finalization scheme derivative and adjoint derivative."""
    discretization = StageOrderedRungeKuttaDiscretization(butcher_tableau)
    ode = IdentityODE()

    # Create a final state
    starting_values = TimeDiscretizationStartingValues(
        initial_time=0.0,
        initial_values=np.ones(ode.get_state_size()),
        independent_inputs=np.ones(ode.get_independent_input_size()),
    )
    state = discretization.time_discretization_starting_scheme(
        ode, starting_values, 0.0
    )

    step_size = 0.1
    final_state = discretization.compute_step(ode, state, step_size)

    # Create perturbations
    final_state_derivative = discretization.create_empty_discretization_state(ode)
    final_state_derivative.final_state = np.array([0.1])

    finalization_value_adjoint = TimeDiscretizationFinalizationValues(
        final_time=0.0,
        final_values=np.array([0.2]),
        final_independent_outputs=np.ones(ode.get_independent_output_size()),
    )

    # Test duality
    dual_fwd, dual_rev = finalization_scheme_duality(
        discretization=discretization,
        ode=ode,
        final_state=final_state,
        final_state_derivative=final_state_derivative,
        finalization_value_adjoint=finalization_value_adjoint,
        step_size=step_size,
        state_dot_func=state_dot_func,
    )

    # The duality relationship should hold
    np.testing.assert_allclose(dual_fwd, dual_rev, rtol=1e-10)


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
