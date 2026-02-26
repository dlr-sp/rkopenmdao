"""Time discretization scheme implementation for ERK and DIRK methods."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


from rkopenmdao.butcher_tableau import ButcherTableau, EmbeddedButcherTableau
from rkopenmdao.discretized_ode.discretized_ode import (
    DiscretizedODE,
    DiscretizedODEInputState,
    DiscretizedODEResultState,
)

from rkopenmdao.time_discretization.runge_kutta_discretization_state import (
    RungeKuttaDiscretizationState,
    EmbeddedRungeKuttaDiscretizationState,
)
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationSchemeInterface,
    TimeDiscretizationStartingValues,
    TimeDiscretizationFinalizationValues,
)


@dataclass
class StageOrderedRungeKuttaDiscretization(TimeDiscretizationSchemeInterface):
    """
    Implementation of `TimeDiscretizationSchemeInterface` for "stage ordered"
    Runge-Kutta methods, meaning that their stages can be performed one after another
    in an sequential manner. This includes ERK and DIRK schemes.

    Parameters
    ----------
    butcher_tableau: ButcherTableau
        Representation of RK scheme as butcher tableau.
    """

    butcher_tableau: ButcherTableau

    def create_empty_discretization_state(
        self, ode: DiscretizedODE
    ) -> RungeKuttaDiscretizationState:
        return RungeKuttaDiscretizationState(
            ode.get_state_size(),
            ode.get_independent_input_size(),
            ode.get_independent_output_size(),
            self.butcher_tableau.number_of_stages(),
            ode.get_linearization_point_size(),
        )

    def compute_step(
        self,
        ode: DiscretizedODE,
        time_discretization_state: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> RungeKuttaDiscretizationState:
        self._shift_state(time_discretization_state)
        time_discretization_state.step_size[0] = step_size
        for i in range(self.butcher_tableau.number_of_stages()):
            time_discretization_state = self._compute_stage(
                ode, time_discretization_state, step_size, i
            )
        time_discretization_state.final_time[0] = (
            time_discretization_state.start_time[0] + step_size
        )
        time_discretization_state.final_state[:] = self._accumulate_step(
            time_discretization_state.start_state,
            time_discretization_state.stage_updates,
            step_size,
        )
        time_discretization_state.final_independent_outputs[:] = (
            self._compute_final_independent_outputs(
                time_discretization_state.stage_times,
                time_discretization_state.stage_independent_outputs,
                time_discretization_state.final_time[0],
            )
        )
        return time_discretization_state

    def compute_step_derivative(
        self,
        ode: DiscretizedODE,
        time_discretization_state: RungeKuttaDiscretizationState,
        time_discretization_state_perturbation: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> RungeKuttaDiscretizationState:
        self._shift_state(time_discretization_state_perturbation)
        for i in range(self.butcher_tableau.number_of_stages()):
            time_discretization_state_perturbation = self._compute_stage_derivative(
                ode,
                time_discretization_state_perturbation,
                step_size,
                i,
                time_discretization_state.linearization_points[i],
            )
        time_discretization_state_perturbation.final_time[0] = (
            time_discretization_state_perturbation.start_time[0]
        )
        time_discretization_state_perturbation.final_state[:] = self._accumulate_step(
            time_discretization_state_perturbation.start_state,
            time_discretization_state_perturbation.stage_updates,
            step_size,
        )
        time_discretization_state_perturbation.final_independent_outputs[:] = (
            self._compute_final_independent_output_derivative(
                time_discretization_state.stage_times,
                time_discretization_state.stage_independent_outputs,
                time_discretization_state.final_time[0],
                time_discretization_state_perturbation.stage_times,
                time_discretization_state_perturbation.stage_independent_outputs,
                time_discretization_state_perturbation.final_time[0],
            )
        )
        return time_discretization_state_perturbation

    def compute_step_adjoint_derivative(
        self,
        ode: DiscretizedODE,
        time_discretization_state: RungeKuttaDiscretizationState,
        time_discretization_state_perturbation: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> RungeKuttaDiscretizationState:
        (
            stage_time_perturbations,
            stage_independent_output_perturbations,
            final_time_perturbations,
        ) = self._compute_final_independent_output_adjoint_derivative(
            time_discretization_state.stage_times,
            time_discretization_state.stage_independent_outputs,
            time_discretization_state.final_time,
            time_discretization_state_perturbation.final_independent_outputs,
        )
        time_discretization_state_perturbation.stage_times[:] = stage_time_perturbations
        time_discretization_state_perturbation.stage_independent_outputs[:] = (
            stage_independent_output_perturbations
        )
        time_discretization_state_perturbation.final_time += final_time_perturbations

        time_discretization_state_perturbation.start_state[:] = (
            time_discretization_state_perturbation.final_state
        )
        for i in range(self.butcher_tableau.number_of_stages()):
            time_discretization_state_perturbation.stage_updates[i] = (
                step_size
                * self.butcher_tableau.butcher_weight_vector[i]
                * time_discretization_state_perturbation.final_state
            )
        time_discretization_state_perturbation.start_time[0] = (
            time_discretization_state_perturbation.final_time[0]
        )
        for i in reversed(range(self.butcher_tableau.number_of_stages())):
            time_discretization_state_perturbation = (
                self._compute_stage_adjoint_derivative(
                    ode,
                    time_discretization_state_perturbation,
                    step_size,
                    i,
                    time_discretization_state.linearization_points[i],
                )
            )
        self._shift_state_reverse(time_discretization_state_perturbation)
        return time_discretization_state_perturbation

    @staticmethod
    def _shift_state(discretization_state: RungeKuttaDiscretizationState):
        discretization_state.start_time[0] = discretization_state.final_time[0]
        discretization_state.start_state[:] = discretization_state.final_state

    @staticmethod
    def _shift_state_reverse(discretization_state: RungeKuttaDiscretizationState):
        discretization_state.final_time[0] = discretization_state.start_time[0]
        discretization_state.final_state[:] = discretization_state.start_state

    def _accumulate_stages(self, stage: int, stage_field: np.ndarray):
        return (
            np.zeros(stage_field.shape[1])
            if stage == 0
            else np.einsum(
                "ij,i",
                stage_field[:stage, :],
                self.butcher_tableau.butcher_matrix[stage, :stage],
            )
        )

    def _accumulate_step(
        self, start_state: np.ndarray, stage_field: np.ndarray, step_size: float
    ) -> np.ndarray:
        return start_state + np.einsum(
            "ij,i",
            stage_field,
            step_size * self.butcher_tableau.butcher_weight_vector,
        )

    def _compute_stage(
        self,
        ode: DiscretizedODE,
        time_discretization_state: RungeKuttaDiscretizationState,
        step_size: float,
        stage: int,
    ) -> RungeKuttaDiscretizationState:
        time_discretization_state.stage_times[stage] = (
            time_discretization_state.start_time[0]
            + step_size * self.butcher_tableau.butcher_time_stages[stage]
        )
        stage_input = self._accumulate_stages(
            stage, time_discretization_state.stage_updates
        )
        ode_input = DiscretizedODEInputState(
            step_input=time_discretization_state.start_state,
            stage_input=stage_input,
            independent_input=time_discretization_state.independent_inputs,
            time=time_discretization_state.stage_times[stage],
        )
        ode_result = ode.compute_update(
            ode_input,
            step_size,
            self.butcher_tableau.butcher_matrix[stage, stage],
        )
        time_discretization_state.stage_updates[stage, :] = ode_result.stage_update
        time_discretization_state.stage_states[stage, :] = ode_result.stage_state
        time_discretization_state.stage_independent_outputs[stage, :] = (
            ode_result.independent_output
        )
        time_discretization_state.linearization_points[stage, :] = (
            ode_result.linearization_point
        )
        return time_discretization_state

    def _compute_stage_derivative(
        self,
        ode: DiscretizedODE,
        time_discretization_state_perturbation: RungeKuttaDiscretizationState,
        step_size: float,
        stage: int,
        linearization_point: np.ndarray,
    ) -> RungeKuttaDiscretizationState:
        time_discretization_state_perturbation.stage_times[stage] = (
            time_discretization_state_perturbation.start_time[0]
        )
        stage_input_perturbations = self._accumulate_stages(
            stage, time_discretization_state_perturbation.stage_updates
        )
        ode_input_perturbations = DiscretizedODEInputState(
            step_input=time_discretization_state_perturbation.start_state,
            stage_input=stage_input_perturbations,
            independent_input=time_discretization_state_perturbation.independent_inputs,
            time=time_discretization_state_perturbation.stage_times[stage],
            linearization_point=linearization_point,
        )
        ode_result_perturbations = ode.compute_update_derivative(
            ode_input_perturbations,
            step_size,
            self.butcher_tableau.butcher_matrix[stage, stage],
        )
        time_discretization_state_perturbation.stage_updates[stage, :] = (
            ode_result_perturbations.stage_update
        )
        time_discretization_state_perturbation.stage_states[stage, :] = (
            ode_result_perturbations.stage_state
        )
        time_discretization_state_perturbation.stage_independent_outputs[stage, :] = (
            ode_result_perturbations.independent_output
        )
        return time_discretization_state_perturbation

    def _compute_stage_adjoint_derivative(
        self,
        ode: DiscretizedODE,
        time_discretization_state_perturbation: RungeKuttaDiscretizationState,
        step_size: float,
        stage: int,
        linearization_point: np.ndarray,
    ) -> RungeKuttaDiscretizationState:
        independent_output = (
            time_discretization_state_perturbation.stage_independent_outputs[stage]
        )
        ode_result_perturbation = DiscretizedODEResultState(
            stage_update=time_discretization_state_perturbation.stage_updates[stage, :],
            stage_state=time_discretization_state_perturbation.stage_states[stage, :],
            independent_output=independent_output,
            linearization_point=linearization_point,
        )
        ode_input_perturbation = ode.compute_update_adjoint_derivative(
            ode_result_perturbation,
            step_size,
            self.butcher_tableau.butcher_matrix[stage, stage],
        )
        time_discretization_state_perturbation.independent_inputs += (
            ode_input_perturbation.independent_input
        )
        time_discretization_state_perturbation.stage_times[
            stage
        ] += ode_input_perturbation.time
        time_discretization_state_perturbation.start_state += (
            ode_input_perturbation.step_input
        )
        for i in range(stage):
            time_discretization_state_perturbation.stage_updates[i, :] += (
                self.butcher_tableau.butcher_matrix[stage, i]
                * ode_input_perturbation.stage_input
            )
        time_discretization_state_perturbation.start_time += (
            time_discretization_state_perturbation.stage_times[stage]
        )
        return time_discretization_state_perturbation

    # TODO: Implement the following three methods.
    # What can be done here greatly depends on how the used RK method looks like
    # For SDIRK methods, this can in general be done by computing pseudo time
    # derivatives. For methods involving explicit stages, this in general needs
    # interpolation.
    # In the case of stiffly-accurate methods (i.e. last stage = end of time step), we
    # can just copy the value from the last stage
    #
    # This will be done at a later date.
    # pylint: disable=unused-argument
    @staticmethod
    def _compute_final_independent_outputs(
        stage_times: np.ndarray,
        stage_independent_outputs: np.ndarray,
        final_time: float,
    ) -> np.ndarray:
        return np.zeros_like(stage_independent_outputs[0])

    @staticmethod
    def _compute_final_independent_output_derivative(
        stage_times: np.ndarray,
        stage_independent_outputs: np.ndarray,
        final_time: float,
        stage_time_perturbations: np.ndarray,
        stage_independent_output_perturbations: np.ndarray,
        final_time_perturbation: float,
    ) -> np.ndarray:
        return np.zeros_like(stage_independent_outputs[0])

    @staticmethod
    def _compute_final_independent_output_adjoint_derivative(
        stage_times: np.ndarray,
        stage_independent_outputs: np.ndarray,
        final_time: float,
        final_independent_output_perturbation: np.ndarray,
    ) -> (np.ndarray, np.ndarray, float):
        return np.zeros_like(stage_times), np.zeros_like(stage_independent_outputs), 0.0

    def time_discretization_starting_scheme(
        self,
        ode: DiscretizedODE,
        starting_values: TimeDiscretizationStartingValues,
        step_size: float,
    ) -> RungeKuttaDiscretizationState:
        initial_discretization_state = self.create_empty_discretization_state(ode)
        initial_discretization_state.final_time[0] = starting_values.initial_time
        initial_discretization_state.final_state[:] = starting_values.initial_values
        initial_discretization_state.independent_inputs[:] = (
            starting_values.independent_inputs
        )
        initial_discretization_state.step_size[0] = step_size
        return initial_discretization_state

    def time_discretization_starting_scheme_derivative(
        self,
        ode: DiscretizedODE,
        starting_values: TimeDiscretizationStartingValues,
        starting_value_perturbations: TimeDiscretizationStartingValues,
        step_size: float,
    ) -> RungeKuttaDiscretizationState:
        initial_discretization_state_perturbations = (
            self.create_empty_discretization_state(ode)
        )
        initial_discretization_state_perturbations.final_time[0] = (
            starting_value_perturbations.initial_time
        )
        initial_discretization_state_perturbations.final_state[:] = (
            starting_value_perturbations.initial_values
        )
        initial_discretization_state_perturbations.independent_inputs[:] = (
            starting_value_perturbations.independent_inputs
        )
        return initial_discretization_state_perturbations

    def time_discretization_starting_scheme_adjoint_derivative(
        self,
        ode: DiscretizedODE,
        starting_values: TimeDiscretizationStartingValues,
        started_discretization_state_perturbations: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> TimeDiscretizationStartingValues:
        return TimeDiscretizationStartingValues(
            started_discretization_state_perturbations.final_time,
            started_discretization_state_perturbations.final_state,
            started_discretization_state_perturbations.independent_inputs,
        )

    def time_discretization_finalization_scheme(
        self,
        ode: DiscretizedODE,
        discretization_state: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> TimeDiscretizationFinalizationValues:
        return TimeDiscretizationFinalizationValues(
            discretization_state.final_time,
            discretization_state.final_state,
            discretization_state.final_independent_outputs,
        )

    def time_discretization_finalization_scheme_derivative(
        self,
        ode: DiscretizedODE,
        discretization_state: RungeKuttaDiscretizationState,
        discretization_state_perturbations: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> TimeDiscretizationFinalizationValues:
        return TimeDiscretizationFinalizationValues(
            discretization_state_perturbations.final_time,
            discretization_state_perturbations.final_state,
            discretization_state_perturbations.final_independent_outputs,
        )

    def time_discretization_finalization_scheme_adjoint_derivative(
        self,
        ode: DiscretizedODE,
        discretization_state: RungeKuttaDiscretizationState,
        finalization_value_perturbations: TimeDiscretizationFinalizationValues,
        step_size: float,
    ) -> RungeKuttaDiscretizationState:
        final_time_discretization_state_perturbation = (
            self.create_empty_discretization_state(ode)
        )
        final_time_discretization_state_perturbation.final_time[0] = (
            finalization_value_perturbations.final_time
        )
        final_time_discretization_state_perturbation.final_state[:] = (
            finalization_value_perturbations.final_values
        )
        final_time_discretization_state_perturbation.final_independent_outputs[:] = (
            finalization_value_perturbations.final_independent_outputs
        )
        return final_time_discretization_state_perturbation


@dataclass
class EmbeddedRungeKuttaMixin:
    """
    Mixin containing all necessary methods and modifications for enabling an embedded
    Runge-Kutta scheme.

    Parameters
    ----------
    butcher_tableau: EmbeddedButcherTableau
        Representation of RK scheme as butcher tableau. Contains an additional set
        of weights for a lower order embedded time integration scheme.
    """

    butcher_tableau: EmbeddedButcherTableau

    def compute_error_estimate(
        self,
        time_discretization_state: EmbeddedRungeKuttaDiscretizationState,
        step_size: float,
    ) -> EmbeddedRungeKuttaDiscretizationState:
        """
        Computes the error estimate for the embedded Runge-Kutta scheme based on the
        schemes true and embedded states.

        Note that this happens in place, meaning the argument
        `time_discretization_state` contains the new data after the call and is
        returned additionally for convenience.

        Parameters
        ----------
        time_discretization_state: EmbeddedRungeKuttaDiscretizationState
            Discretization state on which computations take place.
        step_size: float
            Step size for the current step of time integration.

        Returns
        -------
        time_discretization_state: EmbeddedRungeKuttaDiscretizationState
        """
        time_discretization_state.embedded_state[:] = self._accumulate_embedded_step(
            time_discretization_state.start_state,
            time_discretization_state.stage_updates,
            step_size,
        )
        time_discretization_state.error_estimate[:] = (
            time_discretization_state.final_state
            - time_discretization_state.embedded_state
        )
        return time_discretization_state

    def _accumulate_embedded_step(
        self, start_state: np.ndarray, stage_field: np.ndarray, step_size: float
    ) -> np.ndarray:
        return start_state + np.einsum(
            "ij,i",
            stage_field,
            step_size * self.butcher_tableau.butcher_adaptive_weights,
        )


@dataclass
class StageOrderedEmbeddedRungeKuttaDiscretization(
    EmbeddedRungeKuttaMixin, StageOrderedRungeKuttaDiscretization
):
    """
    Implementation of `TimeDiscretizationSchemeInterface` for "stage ordered"
    Runge-Kutta methods with an embedded scheme.

    Parameters
    ----------
    butcher_tableau: EmbeddedButcherTableau
        Representation of RK scheme as butcher tableau. Contains an additional set
        of weights for a lower order embedded time integration scheme.
    """

    def create_empty_discretization_state(
        self, ode: DiscretizedODE
    ) -> EmbeddedRungeKuttaDiscretizationState:
        return EmbeddedRungeKuttaDiscretizationState(
            ode.get_state_size(),
            ode.get_independent_input_size(),
            ode.get_independent_output_size(),
            self.butcher_tableau.number_of_stages(),
            ode.get_linearization_point_size(),
        )

    def compute_step(
        self,
        ode: DiscretizedODE,
        time_discretization_state: EmbeddedRungeKuttaDiscretizationState,
        step_size: float,
    ) -> EmbeddedRungeKuttaDiscretizationState:
        time_discretization_state = super().compute_step(
            ode, time_discretization_state, step_size
        )
        return super().compute_error_estimate(time_discretization_state, step_size)
