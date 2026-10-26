"""Time discretization scheme implementation for ERK and DIRK methods."""

# All stage-ordered Runge-Kutta discretizations should reside in one file,
# artificially splitting this will only hinder readability.
# pylint: disable=too-many-lines

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
)
from rkopenmdao.states import StartingValues, FinalizationValues


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
        """
        Creates an empty discretization state with sizes given by the `ode` and the
        number of stages of the butcher tableau.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE for which the discretization state is valid.

        Returns
        -------
        time_discretization_state: RungeKuttaDiscretizationState
            Empty initialized discretization state.
        """
        return RungeKuttaDiscretizationState(
            ode_state_size=ode.get_state_size(),
            independent_input_size=ode.get_independent_input_size(),
            independent_output_size=ode.get_independent_output_size(),
            number_of_stages=self.butcher_tableau.number_of_stages(),
            linearization_point_size=ode.get_linearization_point_size(),
        )

    def compute_step(
        self,
        ode: DiscretizedODE,
        time_discretization_state: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> RungeKuttaDiscretizationState:
        """
        Computes one (primal) step of time integration on the `ode` with `step_size`
        based on `time_discretization_state`, evaluating the stages one after another.

        Note that this happens in place, meaning the argument
        `time_discretization_state` contains the new data after the call and is
        returned additionally for convenience.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE implementation of which the step is computed.
        time_discretization_state: RungeKuttaDiscretizationState
            Discretization state on which computations take place.
        step_size: float
            Step size used for the computation of the time step.

        Returns
        -------
        time_discretization_state: RungeKuttaDiscretizationState
        """
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
        """
        Forward-mode differentiated version of `compute_step`. Uses
        `time_discretization_state` as linearization point, calculating the
        jacvec product of `time_discretization_state_perturbation` valid to `ode` with
        `step_size`.

        Note that this happens in place, meaning the
        argument `time_discretization_state_perturbation` contains the new data after
        the call and is returned additionally for convenience.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE implementation of which the jacvec product is computed.
        time_discretization_state: RungeKuttaDiscretizationState
            Primal discretization state acting as linearization point.
        time_discretization_state_perturbation: RungeKuttaDiscretizationState
            (Linear) discretization state on which computations take place.
        step_size: float
            Step size used for the computation of the time step.

        Returns
        -------
        time_discretization_state_perturbation: RungeKuttaDiscretizationState
        """
        self._shift_state(time_discretization_state_perturbation)
        for i in range(self.butcher_tableau.number_of_stages()):
            lin_pt = time_discretization_state.linearization_points[i]
            time_discretization_state_perturbation = self._compute_stage_derivative(
                ode=ode,
                time_discretization_state_perturbation=(
                    time_discretization_state_perturbation
                ),
                step_size=step_size,
                stage=i,
                linearization_point=lin_pt,
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
                stage_times=time_discretization_state.stage_times,
                stage_independent_outputs=(
                    time_discretization_state.stage_independent_outputs,
                )[0],
                final_time=time_discretization_state.final_time[0],
                stage_time_perturbations=(
                    time_discretization_state_perturbation.stage_times
                ),
                stage_independent_output_perturbations=(
                    time_discretization_state_perturbation.stage_independent_outputs
                )[0],
                final_time_perturbation=(
                    time_discretization_state_perturbation.final_time[0]
                ),
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
        """
        Reverse-mode differentiated version of `compute_step`. Uses
        `time_discretization_state` as linearization point, calculating the
        adjoint jacvec product of `time_discretization_state_perturbation` valid to
        `ode` with `step_size`.

        Note that this happens in place, meaning the
        argument `time_discretization_state_perturbation` contains the new data after
        the call and is returned additionally for convenience.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE implementation of which the adjoint jacvec product is computed.
        time_discretization_state: RungeKuttaDiscretizationState
            Primal discretization state acting as linearization point.
        time_discretization_state_perturbation: RungeKuttaDiscretizationState
            (Linear) discretization state on which computations take place.
        step_size: float
            Step size used for the computation of the time step.

        Returns
        -------
        time_discretization_state_perturbation: RungeKuttaDiscretizationState
        """
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
            lin_pt = time_discretization_state.linearization_points[i]
            time_discretization_state_perturbation = (
                self._compute_stage_adjoint_derivative(
                    ode=ode,
                    time_discretization_state_perturbation=(
                        time_discretization_state_perturbation
                    ),
                    step_size=step_size,
                    stage=i,
                    linearization_point=lin_pt,
                )
            )
        self._shift_state_reverse(time_discretization_state_perturbation)
        return time_discretization_state_perturbation

    @staticmethod
    def _shift_state(discretization_state: RungeKuttaDiscretizationState):
        """Copies the final time and state of the discretization state to its start."""
        discretization_state.start_time[0] = discretization_state.final_time[0]
        discretization_state.start_state[:] = discretization_state.final_state

    @staticmethod
    def _shift_state_reverse(discretization_state: RungeKuttaDiscretizationState):
        """Copies the start time and state of the discretization state to its end."""
        discretization_state.final_time[0] = discretization_state.start_time[0]
        discretization_state.final_state[:] = discretization_state.start_state

    def _accumulate_stages(self, stage: int, stage_field: np.ndarray):
        """
        Computes the input of the stage with index `stage` by accumulating the
        stage updates of the previous stages with the corresponding entries of the
        butcher matrix.

        Parameters
        ----------
        stage: int
            Index of the stage whose input is computed.
        stage_field: np.ndarray
            Stage updates of all stages for the current time step.

        Returns
        -------
        stage_input: np.ndarray
            Accumulated stage input for the given stage.
        """
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
        """
        Computes the state at the end of a time step by adding the stage updates
        weighted with the weights of the butcher tableau to the start state.

        Parameters
        ----------
        start_state: np.ndarray
            State at the start of the time step.
        stage_field: np.ndarray
            Stage updates of all stages for the current time step.
        step_size: float
            Step size of the current time step.

        Returns
        -------
        final_state: np.ndarray
            State at the end of the time step.
        """
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
        """
        Computes the stage with the given index of the current time step by evaluating
        the `ode` at the stage time and storing the resulting stage update, stage
        state, independent output, and linearization point.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE implementation of which the stage is computed.
        time_discretization_state: RungeKuttaDiscretizationState
            Discretization state on which computations take place.
        step_size: float
            Step size of the current time step.
        stage: int
            Index of the stage that is computed.

        Returns
        -------
        time_discretization_state: RungeKuttaDiscretizationState
            Discretization state with the data of the computed stage.
        """
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
        *,
        ode: DiscretizedODE,
        time_discretization_state_perturbation: RungeKuttaDiscretizationState,
        step_size: float,
        stage: int,
        linearization_point: np.ndarray,
    ) -> RungeKuttaDiscretizationState:
        """
        Forward-mode differentiated version of `_compute_stage`, evaluating the
        `ode` at the given linearization point and storing the resulting stage
        update, stage state, and independent output of the perturbed state.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE implementation of which the stage is computed.
        time_discretization_state_perturbation: RungeKuttaDiscretizationState
            (Linear) discretization state on which computations take place.
        step_size: float
            Step size of the current time step.
        stage: int
            Index of the stage that is computed.
        linearization_point: np.ndarray
            Linearization point of the primal stage at which the ODE is
            linearized.

        Returns
        -------
        time_discretization_state_perturbation: RungeKuttaDiscretizationState
            (Linear) discretization state with the data of the computed stage.
        """
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
        *,
        ode: DiscretizedODE,
        time_discretization_state_perturbation: RungeKuttaDiscretizationState,
        step_size: float,
        stage: int,
        linearization_point: np.ndarray,
    ) -> RungeKuttaDiscretizationState:
        """
        Reverse-mode differentiated version of `_compute_stage`, accumulating the
        adjoint contributions of the given stage onto the perturbed state, namely on
        the independent inputs, stage times, start state, the stage updates of the
        previous stages, and the start time.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE implementation of which the stage is computed.
        time_discretization_state_perturbation: RungeKuttaDiscretizationState
            (Linear) discretization state on which computations take place.
        step_size: float
            Step size of the current time step.
        stage: int
            Index of the stage that is computed.
        linearization_point: np.ndarray
            Linearization point of the primal stage at which the ODE is
            linearized.

        Returns
        -------
        time_discretization_state_perturbation: RungeKuttaDiscretizationState
            (Linear) discretization state with the accumulated adjoint
            contributions of the stage.
        """
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
        """
        Computes the independent outputs at the end of a time step from the stage
        times and stage independent outputs. Currently returns zero values.

        Parameters
        ----------
        stage_times: np.ndarray
            Times at the stages of the current time step.
        stage_independent_outputs: np.ndarray
            Independent outputs at the stages of the current time step.
        final_time: float
            Time at the end of the current time step.

        Returns
        -------
        final_independent_outputs: np.ndarray
            Independent outputs at the end of the time step.
        """
        return np.zeros_like(stage_independent_outputs[0])

    @staticmethod
    def _compute_final_independent_output_derivative(
        *,
        stage_times: np.ndarray,
        stage_independent_outputs: np.ndarray,
        final_time: float,
        stage_time_perturbations: np.ndarray,
        stage_independent_output_perturbations: np.ndarray,
        final_time_perturbation: float,
    ) -> np.ndarray:
        """
        Computes the perturbation of the independent outputs at the end of a time
        step. Currently returns zero values.

        Parameters
        ----------
        stage_times: np.ndarray
            Times at the stages of the current time step.
        stage_independent_outputs: np.ndarray
            Independent outputs at the stages of the current time step.
        final_time: float
            Time at the end of the current time step.
        stage_time_perturbations: np.ndarray
            Perturbations of the stage times.
        stage_independent_output_perturbations: np.ndarray
            Perturbations of the stage independent outputs.
        final_time_perturbation: float
            Perturbation of the final time.

        Returns
        -------
        final_independent_output_perturbations: np.ndarray
            Perturbation of the independent outputs at the end of the time step.
        """
        return np.zeros_like(stage_independent_outputs[0])

    @staticmethod
    def _compute_final_independent_output_adjoint_derivative(
        stage_times: np.ndarray,
        stage_independent_outputs: np.ndarray,
        final_time: float,
        final_independent_output_perturbation: np.ndarray,
    ) -> (np.ndarray, np.ndarray, float):
        """
        Computes the adjoint contributions of the independent outputs at the end of a
        time step onto the stage times, stage independent outputs, and final time.
        Currently returns zero values.

        Parameters
        ----------
        stage_times: np.ndarray
            Times at the stages of the current time step.
        stage_independent_outputs: np.ndarray
            Independent outputs at the stages of the current time step.
        final_time: float
            Time at the end of the current time step.
        final_independent_output_perturbation: np.ndarray
            Perturbation of the independent outputs at the end of the time step.

        Returns
        -------
        stage_time_perturbations: np.ndarray
            Adjoint contributions to the stage times.
        stage_independent_output_perturbations: np.ndarray
            Adjoint contributions to the stage independent outputs.
        final_time_perturbation: float
            Adjoint contribution to the final time.
        """
        return np.zeros_like(stage_times), np.zeros_like(stage_independent_outputs), 0.0

    def time_discretization_starting_scheme(
        self,
        ode: DiscretizedODE,
        starting_values: StartingValues,
        step_size: float,
    ) -> RungeKuttaDiscretizationState:
        """
        Starting scheme of the time discretization for converting usual data
        representation of state of ODEs to one compatible to the used time
        discretization.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        starting_values: StartingValues
            Values on which the starting scheme is performed.
        step_size: float
            Step size for the starting scheme.

        Returns
        -------
        started_discretization_state: RungeKuttaDiscretizationState
            Converted version of `starting_values` compatible with the used time
            discretization.
        """
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
        starting_values: StartingValues,
        starting_value_perturbations: StartingValues,
        step_size: float,
    ) -> RungeKuttaDiscretizationState:
        """
        Forward-mode differentiated version of the starting scheme.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        starting_values: StartingValues
            Linearization point for the jacobian of the starting scheme.
        starting_value_perturbations: StartingValues
            Perturbations to be multiplied with the jacobian of the starting
            scheme.
        step_size: float
            Step size for the starting scheme.

        Returns
        -------
        started_discretization_state_perturbations: RungeKuttaDiscretizationState
            Result of the jacvec-product of the starting scheme.
        """
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
        starting_values: StartingValues,
        started_discretization_state_perturbations: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> StartingValues:
        """
        Reverse-mode differentiated version of the starting scheme.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        starting_values: StartingValues
            Linearization point for the adjoint jacobian of the starting scheme.
        started_discretization_state_perturbations: RungeKuttaDiscretizationState
            Perturbations to be multiplied with the adjoint jacobian of the
            starting scheme.
        step_size: float
            Step size for the starting scheme.

        Returns
        -------
        starting_value_perturbations: StartingValues
            Result of the adjoint jacvec-product of the starting scheme.
        """
        return StartingValues(
            started_discretization_state_perturbations.final_time[0],
            started_discretization_state_perturbations.final_state,
            started_discretization_state_perturbations.independent_inputs,
        )

    def time_discretization_finalization_scheme(
        self,
        ode: DiscretizedODE,
        discretization_state: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> FinalizationValues:
        """
        Finalization scheme of the time discretization for converting a state
        specific to the discretization back to one compatible with an ODE.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        discretization_state: RungeKuttaDiscretizationState
            Values on which the finalization scheme is performed.
        step_size: float
            Step size for the finalization scheme.

        Returns
        -------
        finalization_values: FinalizationValues
            Converted version of `discretization_state` compatible with the
            used ODE.
        """
        return FinalizationValues(
            discretization_state.final_time[0],
            discretization_state.final_state,
            discretization_state.final_independent_outputs,
        )

    def time_discretization_finalization_scheme_derivative(
        self,
        ode: DiscretizedODE,
        discretization_state: RungeKuttaDiscretizationState,
        discretization_state_perturbations: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> FinalizationValues:
        """
        Forward-mode differentiated version of the finalization scheme.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        discretization_state: RungeKuttaDiscretizationState
            Linearization point for the jacobian of the finalization scheme.
        discretization_state_perturbations: RungeKuttaDiscretizationState
            Perturbations to be multiplied with the jacobian of the finalization
            scheme.
        step_size: float
            Step size for the finalization scheme.

        Returns
        -------
        finalization_value_perturbations: FinalizationValues
            Result of the jacvec-product of the finalization scheme.
        """
        return FinalizationValues(
            discretization_state_perturbations.final_time[0],
            discretization_state_perturbations.final_state,
            discretization_state_perturbations.final_independent_outputs,
        )

    def time_discretization_finalization_scheme_adjoint_derivative(
        self,
        ode: DiscretizedODE,
        discretization_state: RungeKuttaDiscretizationState,
        finalization_value_perturbations: FinalizationValues,
        step_size: float,
    ) -> RungeKuttaDiscretizationState:
        """
        Reverse-mode differentiated version of the finalization scheme.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        discretization_state: RungeKuttaDiscretizationState
            Linearization point for the adjoint jacobian of the finalization
            scheme.
        finalization_value_perturbations: FinalizationValues
            Perturbations to be multiplied with the adjoint jacobian of the
            finalization scheme.
        step_size: float
            Step size for the finalization scheme.

        Returns
        -------
        discretization_state_perturbations: RungeKuttaDiscretizationState
            Result of the adjoint jacvec-product of the finalization scheme.
        """
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

    def get_ode_state(
        self,
        ode: DiscretizedODE,
        discretization_state: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> DiscretizedODEResultState:
        """
        Uses the contents of `discretization_state` to create a valid result state for
        the passed ode containing state data.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        discretization_state: RungeKuttaDiscretizationState
            Discretization state from which data is used.
        step_size: float
            Step size that is possibly necessary for some calculations.

        Returns
        -------
        ode_state: DiscretizedODEResultState
            ODE-compatible result state containing discretization state data.
        """
        return DiscretizedODEResultState(
            stage_update=(
                discretization_state.final_state - discretization_state.start_state
            ),
            stage_state=discretization_state.final_state,
            independent_output=discretization_state.final_independent_outputs,
            linearization_point=discretization_state.linearization_points[-1],
        )

    def get_ode_error_estimate(
        self,
        ode: DiscretizedODE,
        discretization_state: RungeKuttaDiscretizationState,
        step_size: float,
    ) -> DiscretizedODEResultState | None:
        """
        Uses the contents of `discretization_state` to create a valid result state for
        the passed ode containing error estimate data.

        For non-embedded Runge-Kutta methods, no error estimate is available.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        discretization_state: RungeKuttaDiscretizationState
            Discretization state from which data is used.
        step_size: float
            Step size that is possibly necessary for some calculations.

        Returns
        -------
        ode_error_estimate: DiscretizedODEResultState | None
            ODE-compatible result state containing error estimate data, or None if
            no error estimate is available for this method.
        """
        return None


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
        """
        Computes the state at the end of a time step of the embedded scheme by
        adding the stage updates weighted with the adaptive weights of the butcher
        tableau to the start state.

        Parameters
        ----------
        start_state: np.ndarray
            State at the start of the time step.
        stage_field: np.ndarray
            Stage updates of all stages for the current time step.
        step_size: float
            Step size of the current time step.

        Returns
        -------
        embedded_state: np.ndarray
            State at the end of the time step of the embedded scheme.
        """
        return start_state + np.einsum(
            "ij,i",
            stage_field,
            step_size * self.butcher_tableau.butcher_adaptive_weights,
        )

    # pylint: disable=unused-argument
    def get_ode_error_estimate(
        self,
        ode: DiscretizedODE,
        discretization_state: EmbeddedRungeKuttaDiscretizationState,
        step_size: float,
    ) -> DiscretizedODEResultState | None:
        """
        Uses the contents of `discretization_state` to create a valid result state for
        the passed ode containing error estimate data.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        discretization_state: EmbeddedRungeKuttaDiscretizationState
            Discretization state from which data is used.
        step_size: float
            Step size that is possibly necessary for some calculations.

        Returns
        -------
        ode_error_estimate: DiscretizedODEResultState
            ODE-compatible result state containing error estimate data.
        """
        return DiscretizedODEResultState(
            stage_update=discretization_state.error_estimate,
            stage_state=discretization_state.error_estimate,
            independent_output=np.zeros_like(
                discretization_state.final_independent_outputs
            ),
            linearization_point=None,
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
        """
        Creates an empty embedded discretization state with sizes given by the `ode`
        and the number of stages of the butcher tableau.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE for which the discretization state is valid.

        Returns
        -------
        time_discretization_state: EmbeddedRungeKuttaDiscretizationState
            Empty initialized discretization state.
        """
        return EmbeddedRungeKuttaDiscretizationState(
            ode_state_size=ode.get_state_size(),
            independent_input_size=ode.get_independent_input_size(),
            independent_output_size=ode.get_independent_output_size(),
            number_of_stages=self.butcher_tableau.number_of_stages(),
            linearization_point_size=ode.get_linearization_point_size(),
        )

    def compute_step(
        self,
        ode: DiscretizedODE,
        time_discretization_state: EmbeddedRungeKuttaDiscretizationState,
        step_size: float,
    ) -> EmbeddedRungeKuttaDiscretizationState:
        """
        Computes one (primal) step of time integration on the `ode` with `step_size`
        based on `time_discretization_state`, including the error estimate of the
        embedded scheme.

        Note that this happens in place, meaning the argument
        `time_discretization_state` contains the new data after the call and is
        returned additionally for convenience.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE implementation of which the step is computed.
        time_discretization_state: EmbeddedRungeKuttaDiscretizationState
            Discretization state on which computations take place.
        step_size: float
            Step size used for the computation of the time step.

        Returns
        -------
        time_discretization_state: EmbeddedRungeKuttaDiscretizationState
        """
        time_discretization_state = super().compute_step(
            ode, time_discretization_state, step_size
        )
        return super().compute_error_estimate(time_discretization_state, step_size)
