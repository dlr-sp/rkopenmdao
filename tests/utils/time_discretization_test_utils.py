import numpy as np
from rkopenmdao.discretized_ode.discretized_ode import (
    DiscretizedODE,
    DiscretizedODEResultState,
)
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationSchemeInterface,
    TimeDiscretizationStartingValues,
    TimeDiscretizationFinalizationValues,
    TimeDiscretizationStateInterface,
)


def convergence_study(
    discretization: TimeDiscretizationSchemeInterface,
    ode: DiscretizedODE,
    starting_values: TimeDiscretizationStartingValues,
    # starting_value_perturbations: TimeDiscretizationStartingValues,
    # final_value_perturbations: TimeDiscretizationFinalizationValues,
    analytical_solution,
    # analytical_derivative,
    # analytical_adjoint_derivative,
    base_step_size: float,
    num_calculations: int,
) -> np.ndarray:
    result_norms = np.zeros(num_calculations)
    for i in range(num_calculations):
        step_size = base_step_size * 0.5**i

        # primal convergence study
        initial_state = discretization.time_discretization_starting_scheme(
            ode, starting_values, step_size
        )
        final_state = discretization.compute_step(ode, initial_state, step_size)
        final_values = discretization.time_discretization_finalization_scheme(
            ode, final_state, step_size
        )
        error = final_values.final_values - analytical_solution(
            starting_values, step_size
        )
        result_norms[i] = ode.compute_state_norm(
            DiscretizedODEResultState(np.zeros(0), error, np.zeros(0))
        )

        # FIXME: Use this later once test suite is revamped
        # # forward derivative convergence study
        # initial_state_perturbations = (
        #     discretization.time_discretization_starting_scheme_derivative(
        #         ode, starting_values, starting_value_perturbations, step_size
        #     )
        # )
        # final_state_perturbations = discretization.compute_step_derivative(
        #     ode, final_state, initial_state_perturbations, step_size
        # )
        # _final_value_perturbations = (
        #     discretization.time_discretization_finalization_scheme_derivative(
        #         ode, final_state, final_state_perturbations, step_size
        #     )
        # )
        # error = _final_value_perturbations - analytical_derivative(
        #     starting_values, starting_value_perturbations
        # )
        # result_norms[i, 1] = ode.compute_state_norm(
        #     DiscretizedODEResultState(np.zeros(0), error, np.zeros(0))
        # )

        # # adjoint derivative convergence study
        # final_state_perturbations = (
        #     discretization.time_discretization_finalization_scheme_adjoint_derivative(
        #         ode, final_state, final_value_perturbations, step_size
        #     )
        # )
        # initial_state_perturbations = discretization.compute_step_adjoint_derivative(
        #     ode, final_state, final_state_perturbations, step_size
        # )
        # _initial_value_perturbations = (
        #     discretization.time_discretization_starting_scheme_adjoint_derivative(
        #         ode, starting_values, initial_state_perturbations, step_size
        #     )
        # )
        # error = _initial_value_perturbations - analytical_adjoint_derivative(
        #     starting_values, final_value_perturbations
        # )
        # result_norms[i, 2] = ode.compute_state_norm(
        #     DiscretizedODEResultState(np.zeros(0), error, np.zeros(0))
        # )

    norm_comparisons = np.zeros(num_calculations - 1)
    for i in range(num_calculations - 1):
        norm_comparisons[i] = result_norms[i] / result_norms[i + 1]
    print(result_norms)
    return norm_comparisons


def step_duality(
    discretization: TimeDiscretizationSchemeInterface,
    ode: DiscretizedODE,
    state: TimeDiscretizationStateInterface,
    input_state_derivative: TimeDiscretizationStateInterface,
    output_state_adjoint: TimeDiscretizationStateInterface,
    step_size: float,
    state_dot_func: callable,
) -> tuple[float, float]:
    """
    Test the duality between compute_step_derivative and compute_step_adjoint_derivative.

    The duality relationship states that for any perturbations dx and dy:
        <dy, J * dx> = <J^T * dy, dx>
    where J is the Jacobian of the step operation.

    Parameters
    ----------
    discretization : TimeDiscretizationSchemeInterface
        The discretization scheme to test.
    ode : DiscretizedODE
        The ODE instance.
    state : TimeDiscretizationStateInterface
        The state at which to evaluate the Jacobian.
    state_perturbation : TimeDiscretizationStateInterface
        The perturbation to the state (dx for the step operation).
    final_state_perturbation : TimeDiscretizationStateInterface
        The perturbation to the final state (dy for the step operation).
    step_size : float
        The step size.
    state_dot_func : callable
        A function that computes the inner product between two states.
        Signature: state_dot_func(state1, state2) -> float

    Returns
    -------
    tuple[float, float]
        The forward duality pairing <dy, J*dx> and reverse duality pairing <J^T*dy, dx>.
    """
    # Compute forward derivative: delta_y = J_step * dx_step
    input_state_derivative_copy = discretization.create_empty_discretization_state(ode)
    input_state_derivative_copy.set(input_state_derivative)
    final_state_derivative = discretization.compute_step_derivative(
        ode, state, input_state_derivative_copy, step_size
    )

    # Compute forward duality: <dy, J_step*dx_step>
    # dy is the perturbation to the final state from the step operation
    dual_fwd = state_dot_func(final_state_derivative, output_state_adjoint)

    # Compute adjoint derivative: J^T * dy
    input_state_adjoint = discretization.compute_step_adjoint_derivative(
        ode, state, output_state_adjoint, step_size
    )

    # Compute reverse duality: <J^T*dy, dx>
    dual_rev = state_dot_func(input_state_derivative, input_state_adjoint)

    return dual_fwd, dual_rev


def starting_scheme_duality(
    discretization: TimeDiscretizationSchemeInterface,
    ode: DiscretizedODE,
    starting_values: TimeDiscretizationStartingValues,
    starting_value_derivative: TimeDiscretizationStartingValues,
    state_adjoint: TimeDiscretizationStateInterface,
    step_size: float,
    state_dot_func: callable,
) -> tuple[float, float]:
    """
    Test the duality between time_discretization_starting_scheme_derivative
    and time_discretization_starting_scheme_adjoint_derivative.

    The duality relationship states that for any perturbations dx and dy:
        <dy, J * dx> = <J^T * dy, dx>
    where J is the Jacobian of the starting scheme operation.

    Parameters
    ----------
    discretization : TimeDiscretizationSchemeInterface
        The discretization scheme to test.
    ode : DiscretizedODE
        The ODE instance.
    starting_values : TimeDiscretizationStartingValues
        The starting values at which to evaluate the Jacobian.
    starting_value_derivative : TimeDiscretizationStartingValues
        The perturbation to the starting values (dx).
    state_adjoint : TimeDiscretizationStateInterface
        The seed for the state output (dy).
    step_size : float
        The step size.
    state_dot_func : callable
        A function that computes the inner product between two states.
        Signature: state_dot_func(state1, state2) -> float

    Returns
    -------
    tuple[float, float]
        The forward duality pairing <dy, J*dx> and reverse duality pairing <J^T*dy, dx>.
    """
    # Compute forward derivative: delta_y = J_step * dx_step

    state_derivative = discretization.time_discretization_starting_scheme_derivative(
        ode, starting_values, starting_value_derivative, step_size
    )

    # Compute forward duality: <dy, J_step*dx_step>
    # dy is the perturbation to the final state from the step operation
    dual_fwd = state_dot_func(state_derivative, state_adjoint)

    # Compute adjoint derivative: J^T * dy
    starting_value_adjoint = (
        discretization.time_discretization_starting_scheme_adjoint_derivative(
            ode, starting_values, state_adjoint, step_size
        )
    )

    # Compute reverse duality: <J^T*dy, dx>
    dual_rev = (
        starting_value_derivative.initial_time * starting_value_adjoint.initial_time
    )
    dual_rev += np.dot(
        starting_value_derivative.initial_values, starting_value_adjoint.initial_values
    )
    dual_rev += np.dot(
        starting_value_derivative.independent_inputs,
        starting_value_adjoint.independent_inputs,
    )

    return dual_fwd, dual_rev


def finalization_scheme_duality(
    discretization: TimeDiscretizationSchemeInterface,
    ode: DiscretizedODE,
    final_state: TimeDiscretizationStateInterface,
    final_state_derivative: TimeDiscretizationStateInterface,
    finalization_value_adjoint: TimeDiscretizationFinalizationValues,
    step_size: float,
    state_dot_func: callable,
) -> tuple[float, float]:
    """
    Test the duality between time_discretization_finalization_scheme_derivative
    and time_discretization_finalization_scheme_adjoint_derivative.

    The duality relationship states that for any perturbations dx and dy:
        <dy, J * dx> = <J^T * dy, dx>
    where J is the Jacobian of the finalization scheme operation.

    Parameters
    ----------
    discretization : TimeDiscretizationSchemeInterface
        The discretization scheme to test.
    ode : DiscretizedODE
        The ODE instance.
    final_state : TimeDiscretizationStateInterface
        The state at which to evaluate the Jacobian.
    final_state_derivative : TimeDiscretizationStateInterface
        The perturbation to the state (dx).
    finalization_value_adjoint : TimeDiscretizationFinalizationValues
        The seed for the finalization output (dy).
    step_size : float
        The step size.
    state_dot_func : callable
        A function that computes the inner product between two states.
        Signature: state_dot_func(state1, state2) -> float

    Returns
    -------
    tuple[float, float]
        The forward duality pairing <dy, J*dx> and reverse duality pairing <J^T*dy, dx>.
    """
    # Compute forward derivative: delta_y = J_step * dx_step

    finalization_value_derivative = (
        discretization.time_discretization_finalization_scheme_derivative(
            ode, final_state, final_state_derivative, step_size
        )
    )

    # Compute forward duality: <dy, J_step*dx_step>
    # dy is the perturbation to the final state from the step operation
    dual_fwd = (
        finalization_value_derivative.final_time * finalization_value_adjoint.final_time
    )
    dual_fwd += np.dot(
        finalization_value_derivative.final_values,
        finalization_value_adjoint.final_values,
    )
    dual_fwd += np.dot(
        finalization_value_derivative.final_independent_outputs,
        finalization_value_adjoint.final_independent_outputs,
    )

    # Compute adjoint derivative: J^T * dy
    final_state_adjoint = (
        discretization.time_discretization_finalization_scheme_adjoint_derivative(
            ode, final_state, finalization_value_adjoint, step_size
        )
    )

    # Compute reverse duality: <J^T*dy, dx>
    dual_rev = state_dot_func(final_state_derivative, final_state_adjoint)

    return dual_fwd, dual_rev
