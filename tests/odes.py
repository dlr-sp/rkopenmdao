"""Some direct implementations of DiscretizedODE to test time discretizations."""

from dataclasses import dataclass

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
    return StartingValues(
        final_value_perturbations.final_time
        + passed_time * final_value_perturbations.final_values,
        final_value_perturbations.final_values,
        np.zeros(0),
    )


@dataclass
class TimeScaledIdentityODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = t*x(t).
    """

    _cached_linearization: np.ndarray = np.zeros(3)

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
    exp_factor = np.exp(
        initial_values.initial_time * passed_time + 0.5 * passed_time**2
    )
    return StartingValues(
        final_value_perturbations.final_time
        + exp_factor
        * initial_values.initial_values
        * passed_time
        * final_value_perturbations.final_values,
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

    _cached_linearization: np.ndarray = np.zeros(2)

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
    return StartingValues(
        final_value_perturbations.final_time,
        (1 + 0.5 * passed_time / (initial_values.initial_values**0.5))
        * final_value_perturbations.final_values,
        np.zeros(0),
    )
