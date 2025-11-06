import numpy as np
from rkopenmdao.discretized_ode.discretized_ode import (
    DiscretizedODE,
    DiscretizedODEResultState,
)
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationSchemeInterface,
    TimeDiscretizationStartingValues,
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
