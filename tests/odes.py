"""Some direct implementations of DiscretizedODE to test time discretizations."""

# All ODE test implementations and their reference solutions should reside in
# one file, artificially splitting this will only hinder readability.
# pylint: disable=too-many-lines

# pylint: disable=unnecessary-lambda

from dataclasses import dataclass, field

import numpy as np

from rkopenmdao.discretized_ode.discretized_ode import (
    DiscretizedODE,
    DiscretizedODEInputState,
    DiscretizedODEResultState,
)
from rkopenmdao.states import StartingValues, FinalizationValues


class IdentityODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = x(t).
    """

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        stage_update = (ode_input.step_input + step_size * ode_input.stage_input) / (
            1.0 - step_size * stage_factor
        )
        stage_state = stage_update.copy()
        return DiscretizedODEResultState(stage_update, stage_state, np.zeros(0))

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        return self.compute_update(
            ode_input_perturbation,
            step_size,
            stage_factor,
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        step_input_pert = (
            ode_result_perturbation.stage_update + ode_result_perturbation.stage_state
        ) / (1 - step_size * stage_factor)
        stage_output_pert = step_size * step_input_pert
        return DiscretizedODEInputState(
            step_input_pert, stage_output_pert, np.zeros(0), 0.0
        )

    def compute_state_norm(self, state: DiscretizedODEResultState):
        return np.abs(state.stage_state)[0]

    def get_state_size(self) -> int:
        return 1

    def get_independent_input_size(self) -> int:
        return 0

    def get_independent_output_size(self) -> int:
        return 0

    def get_linearization_point_size(self):
        return 0


def identity_ode_solution(
    initial_values: StartingValues, passed_time: float
) -> FinalizationValues:
    """Compute the analytic solution of the ODE x'(t) = x(t).

    Evaluates x(t0 + s) = x0 * e^s, where t0 and x0 are the initial time
    and values, and s is the passed time.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Solution at time t0 + s, including final time, values, and
        independent outputs.
    """
    return FinalizationValues(
        passed_time + initial_values.initial_time,
        initial_values.initial_values * np.e**passed_time,
        np.zeros(0),
    )


def identity_ode_solution_derivative(
    initial_values: StartingValues,
    initial_value_perturbations: StartingValues,
    passed_time: float,
) -> FinalizationValues:
    """Compute the derivative of the solution of the ODE x'(t) = x(t) with
    respect to the initial values.

    Evaluates the derivative of x(t0 + s) = x0 * e^s applied to the given
    perturbations of the initial values.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    initial_value_perturbations: StartingValues
        Perturbations of the initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Perturbed solution at time t0 + s, including final time, values,
        and independent outputs.
    """
    return FinalizationValues(
        initial_value_perturbations.initial_time,
        initial_value_perturbations.initial_values * np.e**passed_time,
        np.zeros(0),
    )


def identity_ode_solution_adjoint_derivative(
    initial_values: StartingValues,
    final_value_perturbations: FinalizationValues,
    passed_time: float,
) -> StartingValues:
    """Compute the adjoint derivative of the solution of the ODE
    x'(t) = x(t) with respect to the initial values.

    Propagates the given perturbations of the final values of the solution
    x(t0 + s) = x0 * e^s back to the initial values.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    final_value_perturbations: FinalizationValues
        Perturbations of the final time, values, and independent outputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    StartingValues
        Perturbations of the initial values induced by
        ``final_value_perturbations``.
    """
    return StartingValues(
        final_value_perturbations.final_time,
        final_value_perturbations.final_values * np.e**passed_time,
        np.zeros(0),
    )


class TimeODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = t.
    """

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        stage_update = np.array([ode_input.time])
        stage_state = ode_input.step_input + step_size * (
            ode_input.stage_input + stage_factor * stage_update
        )
        return DiscretizedODEResultState(stage_update, stage_state, 0.0)

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        return self.compute_update(
            ode_input_perturbation,
            step_size,
            stage_factor,
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        step_input_pert = ode_result_perturbation.stage_state
        stage_input_pert = step_size * ode_result_perturbation.stage_state
        time_pert = (
            ode_result_perturbation.stage_update
            + step_size * stage_factor * ode_result_perturbation.stage_state
        )
        return DiscretizedODEInputState(
            step_input_pert, stage_input_pert, np.zeros(0), time_pert[0]
        )

    def compute_state_norm(self, state: DiscretizedODEResultState):
        return np.abs(state.stage_state)[0]

    def get_state_size(self) -> int:
        return 1

    def get_independent_input_size(self) -> int:
        return 0

    def get_independent_output_size(self) -> int:
        return 0

    def get_linearization_point_size(self):
        return 0


def time_ode_solution(
    initial_values: StartingValues, passed_time: float
) -> FinalizationValues:
    """Compute the analytic solution of the ODE x'(t) = t.

    Evaluates x(t0 + s) = x0 + t0 * s + s**2 / 2, where t0 and x0 are the
    initial time and values, and s is the passed time.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Solution at time t0 + s, including final time, values, and
        independent outputs.
    """
    return FinalizationValues(
        passed_time + initial_values.initial_time,
        initial_values.initial_values
        + initial_values.initial_time * passed_time
        + 0.5 * passed_time**2,
        np.zeros(0),
    )


def time_ode_solution_derivative(
    initial_values: StartingValues,
    initial_value_perturbations: StartingValues,
    passed_time: float,
) -> FinalizationValues:
    """Compute the derivative of the solution of the ODE x'(t) = t with
    respect to the initial values.

    Evaluates the derivative of x(t0 + s) = x0 + t0 * s + s**2 / 2 with
    respect to t0 and x0 applied to the given perturbations of the initial
    values.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    initial_value_perturbations: StartingValues
        Perturbations of the initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Perturbed solution at time t0 + s, including final time, values,
        and independent outputs.
    """
    return FinalizationValues(
        initial_value_perturbations.initial_time,
        initial_value_perturbations.initial_values
        + passed_time * initial_value_perturbations.initial_time,
        np.zeros(0),
    )


def time_ode_solution_adjoint_derivative(
    initial_values: StartingValues,
    final_value_perturbations: FinalizationValues,
    passed_time: float,
) -> StartingValues:
    """Compute the adjoint derivative of the solution of the ODE x'(t) = t
    with respect to the initial values.

    Propagates the given perturbations of the final values of the solution
    x(t0 + s) = x0 + t0 * s + s**2 / 2 back to the initial values.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    final_value_perturbations: FinalizationValues
        Perturbations of the final time, values, and independent outputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    StartingValues
        Perturbations of the initial values induced by
        ``final_value_perturbations``.
    """
    return StartingValues(
        final_value_perturbations.final_time
        + passed_time * final_value_perturbations.final_values[0],
        final_value_perturbations.final_values,
        np.zeros(0),
    )


@dataclass
class TimeScaledIdentityODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = t*x(t).
    """

    _cached_linearization: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        stage_state = (ode_input.step_input + step_size * ode_input.stage_input) / (
            1 - ode_input.time * step_size * stage_factor
        )
        stage_update = ode_input.time * stage_state
        return DiscretizedODEResultState(
            stage_update,
            stage_state,
            np.zeros(0),
            np.array(
                [ode_input.time, ode_input.step_input[0], ode_input.stage_input[0]]
            ),
        )

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        time = ode_input_perturbation.linearization_point[0]
        step_input = ode_input_perturbation.linearization_point[1]
        stage_input = ode_input_perturbation.linearization_point[2]

        inv_divisor = 1 / (1 - time * step_size * stage_factor)

        stage_update_pert = (
            time
            * (
                ode_input_perturbation.step_input
                + step_size * ode_input_perturbation.stage_input
            )
            * inv_divisor
        ) + (step_input + step_size * stage_input) * (
            inv_divisor + step_size * stage_factor * inv_divisor**2
        ) * ode_input_perturbation.time

        stage_state_pert = (
            ode_input_perturbation.step_input
            + step_size * ode_input_perturbation.stage_input
        ) * inv_divisor + (
            step_input + step_size * stage_input
        ) * step_size * stage_factor * inv_divisor**2 * ode_input_perturbation.time

        return DiscretizedODEResultState(
            stage_update_pert, stage_state_pert, np.zeros(0)
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        time = ode_result_perturbation.linearization_point[0]
        step_input = ode_result_perturbation.linearization_point[1]
        stage_input = ode_result_perturbation.linearization_point[2]

        inv_divisor = 1 / (1 - time * step_size * stage_factor)

        step_input_pert = (
            time * ode_result_perturbation.stage_update
            + ode_result_perturbation.stage_state
        ) * inv_divisor
        stage_input_pert = step_size * step_input_pert
        time_pert = (step_input + step_size * stage_input) * (
            inv_divisor + step_size * stage_factor * inv_divisor**2
        ) * ode_result_perturbation.stage_update + (
            step_input + step_size * stage_input
        ) * step_size * stage_factor * inv_divisor**2 * (
            ode_result_perturbation.stage_state
        )
        return DiscretizedODEInputState(
            step_input_pert, stage_input_pert, np.zeros(0), time_pert[0]
        )

    def compute_state_norm(self, state: DiscretizedODEResultState):
        return np.abs(state.stage_state)[0]

    def get_state_size(self) -> int:
        return 1

    def get_independent_input_size(self) -> int:
        return 0

    def get_independent_output_size(self) -> int:
        return 0

    def get_linearization_point_size(self):
        return 3


def time_scaled_identity_ode_solution(
    initial_values: StartingValues, passed_time: float
) -> FinalizationValues:
    """Compute the analytic solution of the ODE x'(t) = t * x(t).

    Evaluates x(t0 + s) = x0 * e^(t0 * s + s**2 / 2), where t0 and x0 are
    the initial time and values, and s is the passed time.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Solution at time t0 + s, including final time, values, and
        independent outputs.
    """
    return FinalizationValues(
        passed_time + initial_values.initial_time,
        initial_values.initial_values
        * np.exp(initial_values.initial_time * passed_time + 0.5 * passed_time**2),
        np.zeros(0),
    )


def time_scaled_identity_ode_solution_derivative(
    initial_values: StartingValues,
    initial_value_perturbations: StartingValues,
    passed_time: float,
) -> FinalizationValues:
    """Compute the derivative of the solution of the ODE x'(t) = t * x(t)
    with respect to the initial values.

    Evaluates the derivative of
    x(t0 + s) = x0 * e^(t0 * s + s**2 / 2) with respect to t0 and x0
    applied to the given perturbations of the initial values.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    initial_value_perturbations: StartingValues
        Perturbations of the initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Perturbed solution at time t0 + s, including final time, values,
        and independent outputs.
    """
    exp_factor = np.exp(
        initial_values.initial_time * passed_time + 0.5 * passed_time**2
    )
    return FinalizationValues(
        initial_value_perturbations.initial_time,
        exp_factor
        * (
            initial_values.initial_values
            * passed_time
            * initial_value_perturbations.initial_time
            + initial_value_perturbations.initial_values
        ),
        np.zeros(0),
    )


def time_scaled_identity_ode_solution_adjoint_derivative(
    initial_values: StartingValues,
    final_value_perturbations: FinalizationValues,
    passed_time: float,
) -> StartingValues:
    """Compute the adjoint derivative of the solution of the ODE
    x'(t) = t * x(t) with respect to the initial values.

    Propagates the given perturbations of the final values of the solution
    x(t0 + s) = x0 * e^(t0 * s + s**2 / 2) back to the initial values.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    final_value_perturbations: FinalizationValues
        Perturbations of the final time, values, and independent outputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    StartingValues
        Perturbations of the initial values induced by
        ``final_value_perturbations``.
    """
    exp_factor = np.exp(
        initial_values.initial_time * passed_time + 0.5 * passed_time**2
    )
    return StartingValues(
        final_value_perturbations.final_time
        + exp_factor
        * initial_values.initial_values[0]
        * passed_time
        * final_value_perturbations.final_values[0],
        exp_factor * final_value_perturbations.final_values,
        np.zeros(0),
    )


class ParameterODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = b, with b a time independent
    parameter.
    """

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        stage_update = ode_input.independent_input
        stage_output = ode_input.step_input + step_size * (
            ode_input.stage_input * stage_factor * stage_update
        )

        return DiscretizedODEResultState(stage_update, stage_output, np.zeros(0))

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        stage_update_pert = ode_input_perturbation.independent_input
        stage_output_pert = ode_input_perturbation.step_input + step_size * (
            ode_input_perturbation.stage_input + stage_factor * stage_update_pert
        )
        return DiscretizedODEResultState(
            stage_update_pert, stage_output_pert, np.zeros(0)
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        step_input_pert = ode_result_perturbation.stage_state
        stage_input_pert = step_size * step_input_pert
        independent_input_pert = (
            ode_result_perturbation.stage_update
            + step_size * stage_factor * ode_result_perturbation.stage_state
        )

        return DiscretizedODEInputState(
            step_input_pert, stage_input_pert, independent_input_pert, 0.0
        )

    def compute_state_norm(self, state: DiscretizedODEResultState):
        return np.abs(state.stage_state)[0]

    def get_state_size(self) -> int:
        return 1

    def get_independent_input_size(self) -> int:
        return 1

    def get_independent_output_size(self) -> int:
        return 0

    def get_linearization_point_size(self):
        return 0


def parameter_ode_solution(
    initial_values: StartingValues, passed_time: float
) -> FinalizationValues:
    """Compute the analytic solution of the ODE x'(t) = b.

    Evaluates x(t0 + s) = x0 + s * b, where t0 and x0 are the initial time
    and values, b is the time independent parameter given by the
    independent inputs, and s is the passed time.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Solution at time t0 + s, including final time, values, and
        independent outputs.
    """
    return FinalizationValues(
        passed_time + initial_values.initial_time,
        initial_values.initial_values + passed_time * initial_values.independent_inputs,
        np.zeros(0),
    )


def parameter_ode_solution_derivative(
    initial_values: StartingValues,
    initial_value_perturbations: StartingValues,
    passed_time: float,
) -> FinalizationValues:
    """Compute the derivative of the solution of the ODE x'(t) = b with
    respect to the initial values.

    Evaluates the derivative of x(t0 + s) = x0 + s * b with respect to
    x0 and b applied to the given perturbations of the initial values.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    initial_value_perturbations: StartingValues
        Perturbations of the initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Perturbed solution at time t0 + s, including final time, values,
        and independent outputs.
    """
    return FinalizationValues(
        initial_value_perturbations.initial_time,
        initial_value_perturbations.initial_values
        + passed_time * initial_value_perturbations.independent_inputs,
        np.zeros(0),
    )


def parameter_ode_solution_adjoint_derivative(
    initial_values: StartingValues,
    final_value_perturbations: FinalizationValues,
    passed_time: float,
) -> StartingValues:
    """Compute the adjoint derivative of the solution of the ODE x'(t) = b
    with respect to the initial values.

    Propagates the given perturbations of the final values of the solution
    x(t0 + s) = x0 + s * b back to the initial values and the independent
    inputs.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    final_value_perturbations: FinalizationValues
        Perturbations of the final time, values, and independent outputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    StartingValues
        Perturbations of the initial values induced by
        ``final_value_perturbations``.
    """
    return StartingValues(
        final_value_perturbations.final_time,
        final_value_perturbations.final_values,
        passed_time * final_value_perturbations.final_values,
    )


@dataclass
class RootODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = sqrt(x(t)).
    """

    _cached_linearization: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        stage_update = 0.5 * step_size * stage_factor + np.sqrt(
            0.25 * step_size**2 * stage_factor**2
            + ode_input.step_input
            + step_size * ode_input.stage_input
        )
        stage_output = ode_input.step_input + step_size * (
            ode_input.stage_input + stage_factor * stage_update
        )

        return DiscretizedODEResultState(
            stage_update,
            stage_output,
            np.zeros(0),
            np.array([ode_input.step_input[0], ode_input.stage_input[0]]),
        )

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        step_input = ode_input_perturbation.linearization_point[0]
        stage_input = ode_input_perturbation.linearization_point[1]

        inv_divisor = 0.5 / np.sqrt(
            0.25 * step_size**2 * stage_factor**2 + step_input + step_size * stage_input
        )

        stage_update_pert = inv_divisor * (
            ode_input_perturbation.step_input
            + step_size * ode_input_perturbation.stage_input
        )
        stage_output_pert = (1 + 0.5 * step_size * stage_factor * inv_divisor) * (
            ode_input_perturbation.step_input
            + step_size * ode_input_perturbation.stage_input
        )
        return DiscretizedODEResultState(
            stage_update_pert, stage_output_pert, np.zeros(0)
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        step_input = ode_result_perturbation.linearization_point[0]
        stage_input = ode_result_perturbation.linearization_point[1]

        inv_divisor = 0.5 / np.sqrt(
            0.25 * step_size**2 * stage_factor**2 + step_input + step_size * stage_input
        )

        step_input_pert = (
            ode_result_perturbation.stage_update * inv_divisor
            + ode_result_perturbation.stage_state
            * (1 + step_size * stage_factor * inv_divisor)
        )
        stage_input_pert = step_size * step_input_pert

        return DiscretizedODEInputState(
            step_input_pert, stage_input_pert, np.zeros(0), 0.0
        )

    def compute_state_norm(self, state: DiscretizedODEResultState):
        return np.abs(state.stage_state)[0]

    def get_state_size(self) -> int:
        return 1

    def get_independent_input_size(self) -> int:
        return 0

    def get_independent_output_size(self) -> int:
        return 0

    def get_linearization_point_size(self):
        return 2


def root_ode_solution(
    initial_values: StartingValues, passed_time: float
) -> FinalizationValues:
    """Compute the analytic solution of the ODE x'(t) = sqrt(x(t)).

    Evaluates x(t0 + s) = x0 + s * sqrt(x0) + s**2 / 4, where t0 and x0
    are the initial time and values, and s is the passed time.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Solution at time t0 + s, including final time, values, and
        independent outputs.
    """
    return FinalizationValues(
        passed_time + initial_values.initial_time,
        initial_values.initial_values
        + passed_time * initial_values.initial_values**0.5
        + 0.25 * passed_time**2,
        np.zeros(0),
    )


def root_ode_solution_derivative(
    initial_values: StartingValues,
    initial_value_perturbations: StartingValues,
    passed_time: float,
) -> FinalizationValues:
    """Compute the derivative of the solution of the ODE x'(t) = sqrt(x(t))
    with respect to the initial values.

    Evaluates the derivative of
    x(t0 + s) = x0 + s * sqrt(x0) + s**2 / 4 with respect to x0 applied to
    the given perturbations of the initial values.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    initial_value_perturbations: StartingValues
        Perturbations of the initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Perturbed solution at time t0 + s, including final time, values,
        and independent outputs.
    """
    return FinalizationValues(
        initial_value_perturbations.initial_time,
        (1 + 0.5 * passed_time / (initial_values.initial_values**0.5))
        * initial_value_perturbations.initial_values,
        np.zeros(0),
    )


def root_ode_solution_adjoint_derivative(
    initial_values: StartingValues,
    final_value_perturbations: FinalizationValues,
    passed_time: float,
) -> StartingValues:
    """Compute the adjoint derivative of the solution of the ODE
    x'(t) = sqrt(x(t)) with respect to the initial values.

    Propagates the given perturbations of the final values of the solution
    x(t0 + s) = x0 + s * sqrt(x0) + s**2 / 4 back to the initial values.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    final_value_perturbations: FinalizationValues
        Perturbations of the final time, values, and independent outputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    StartingValues
        Perturbations of the initial values induced by
        ``final_value_perturbations``.
    """
    return StartingValues(
        final_value_perturbations.final_time,
        (1 + 0.5 * passed_time / (initial_values.initial_values**0.5))
        * final_value_perturbations.final_values,
        np.zeros(0),
    )


@dataclass
class TwoDimODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = ((0, 1), (1, 0)) x(t).
    """

    @staticmethod
    def calculate_inv_matrix(step_size, stage_factor):
        """Compute the inverse of M - factor * I for M = ((0, 1), (1, 0)).

        With factor = step_size * stage_factor, the inverse is applied to
        the step input and stage input in ``compute_update`` to obtain the
        stage update of the ODE x'(t) = M x(t) for an implicit stage
        evaluation.

        Parameters
        ----------
        step_size: float
            Size of the time step.
        stage_factor: float
            Butcher tableau stage factor of the current stage.

        Returns
        -------
        np.ndarray
            Inverse of ``M - step_size * stage_factor * I``.
        """
        factor = step_size * stage_factor
        divisor = factor**2 - 1
        inv_matrix = np.zeros((2, 2))
        inv_matrix[0, 0] = -factor / divisor
        inv_matrix[0, 1] = -1 / divisor
        inv_matrix[1, 0] = -1 / divisor
        inv_matrix[1, 1] = -factor / divisor
        return inv_matrix

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        inv_matrix = self.calculate_inv_matrix(step_size, stage_factor)
        stage_update = inv_matrix @ (
            ode_input.step_input + step_size * ode_input.stage_input
        )

        stage_output = ode_input.step_input + step_size * (
            ode_input.stage_input + stage_factor * stage_update
        )

        return DiscretizedODEResultState(
            stage_update,
            stage_output,
            np.zeros(0),
        )

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        return self.compute_update(
            ode_input_perturbation,
            step_size,
            stage_factor,
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        inv_matrix_transpose = self.calculate_inv_matrix(
            step_size, stage_factor
        ).transpose()
        step_input_perturbation = (
            inv_matrix_transpose @ ode_result_perturbation.stage_update
            + (np.identity(2) + step_size * stage_factor * inv_matrix_transpose)
            @ ode_result_perturbation.stage_state
        )

        stage_input_perturbation = step_size * step_input_perturbation
        return DiscretizedODEInputState(
            step_input_perturbation, stage_input_perturbation, np.zeros(0), 0.0
        )

    def compute_state_norm(self, state: DiscretizedODEResultState):
        return np.linalg.norm(state.stage_state)

    def get_state_size(self) -> int:
        return 2

    def get_independent_input_size(self) -> int:
        return 0

    def get_independent_output_size(self) -> int:
        return 0

    def get_linearization_point_size(self):
        return 0


def two_dim_ode_solution(
    initial_values: StartingValues, passed_time: float
) -> FinalizationValues:
    """Compute the analytic solution of the ODE x'(t) = M x(t).

    Evaluates x(t0 + s) = cosh(s) * x0 + sinh(s) * M * x0 with
    M = ((0, 1), (1, 0)), where t0 and x0 are the initial time and values,
    and s is the passed time.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Solution at time t0 + s, including final time, values, and
        independent outputs.
    """
    return FinalizationValues(
        passed_time + initial_values.initial_time,
        np.array(
            [
                initial_values.initial_values[0] * np.cosh(passed_time)
                + initial_values.initial_values[1] * np.sinh(passed_time),
                initial_values.initial_values[0] * np.sinh(passed_time)
                + initial_values.initial_values[1] * np.cosh(passed_time),
            ]
        ),
        np.zeros(0),
    )


def two_dim_ode_solution_derivative(
    initial_values: StartingValues,
    initial_value_perturbations: StartingValues,
    passed_time: float,
) -> FinalizationValues:
    """Compute the derivative of the solution of the ODE x'(t) = M x(t)
    with respect to the initial values.

    Evaluates the derivative of
    x(t0 + s) = cosh(s) * x0 + sinh(s) * M * x0 with
    M = ((0, 1), (1, 0)) applied to the given perturbations of the initial
    values.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    initial_value_perturbations: StartingValues
        Perturbations of the initial time, values, and independent inputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    FinalizationValues
        Perturbed solution at time t0 + s, including final time, values,
        and independent outputs.
    """
    return FinalizationValues(
        initial_value_perturbations.initial_time,
        np.array(
            [
                initial_value_perturbations.initial_values[0] * np.cosh(passed_time)
                + initial_value_perturbations.initial_values[1] * np.sinh(passed_time),
                initial_value_perturbations.initial_values[0] * np.sinh(passed_time)
                + initial_value_perturbations.initial_values[1] * np.cosh(passed_time),
            ]
        ),
        np.zeros(0),
    )


def two_dim_ode_solution_adjoint_derivative(
    initial_values: StartingValues,
    final_value_perturbations: FinalizationValues,
    passed_time: float,
) -> StartingValues:
    """Compute the adjoint derivative of the solution of the ODE
    x'(t) = M x(t) with respect to the initial values.

    Propagates the given perturbations of the final values of the solution
    x(t0 + s) = cosh(s) * x0 + sinh(s) * M * x0 with
    M = ((0, 1), (1, 0)) back to the initial values.

    Parameters
    ----------
    initial_values: StartingValues
        Initial time, values, and independent inputs.
    final_value_perturbations: FinalizationValues
        Perturbations of the final time, values, and independent outputs.
    passed_time: float
        Time elapsed since the start of the integration.

    Returns
    -------
    StartingValues
        Perturbations of the initial values induced by
        ``final_value_perturbations``.
    """
    return StartingValues(
        final_value_perturbations.final_time,
        np.array(
            [
                final_value_perturbations.final_values[0] * np.cosh(passed_time)
                + final_value_perturbations.final_values[1] * np.sinh(passed_time),
                final_value_perturbations.final_values[0] * np.sinh(passed_time)
                + final_value_perturbations.final_values[1] * np.cosh(passed_time),
            ]
        ),
        np.zeros(0),
    )
