# pylint: disable=missing-module-docstring

from __future__ import annotations

import numpy as np

from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.discretized_ode.discretized_ode import (
    DiscretizedODE,
    DiscretizedODEInputState,
    DiscretizedODEResultState,
)
from rkopenmdao.error_controller import ErrorController
from rkopenmdao.error_measurer import ErrorMeasurer
from rkopenmdao.errors import RungeKuttaError


class RungeKuttaScheme:
    """Implements functions used to apply a (embedded) Runge-Kutta method a function
    represented by a functor. The current main use is with an OpenMDAO problem wrapped
    to such a functor.

    Parameters
    ----------
    butcher_tableau: ButcherTableau, EmbeddedButcherTableau
        The butcher tableau to apply the Runge-Kutta method to.
    ode: DiscretizedODE
        The ODE that the Runge-Kutta scheme is applied to.
    use_adaptive_time_stepping: bool
        Whether to use adaptive time stepping.
    error_controller: ErrorController
        Error controller that estimates the next time-difference jumps of a Runge-Kutta
        scheme.
    """

    def __init__(
        self,
        butcher_tableau: ButcherTableau,
        ode: DiscretizedODE,
        use_adaptive_time_stepping: bool = False,
        error_controller: ErrorController = None,
        error_measurer: ErrorMeasurer = None,
    ):
        self.butcher_tableau = butcher_tableau
        self.ode = ode
        self.use_adaptive_time_stepping = use_adaptive_time_stepping
        if use_adaptive_time_stepping and not error_controller:
            raise RungeKuttaError(
                "An error controller must be passed if Butcher Tableau is embedded"
            )
        else:
            self.error_controller: ErrorController = error_controller
            self.error_measurer: ErrorMeasurer = error_measurer

    def compute_stage(
        self,
        stage: int,
        delta_t: float,
        old_time: float,
        old_state: np.ndarray,
        accumulated_stages: np.ndarray,
        parameters: np.ndarray,
    ) -> np.ndarray:
        """Computes the new stage variable based on the current information."""
        stage_time = (
            old_time + delta_t * self.butcher_tableau.butcher_time_stages[stage]
        )
        butcher_diagonal_element = self.butcher_tableau.butcher_matrix[stage, stage]
        return self.ode.compute_update(
            DiscretizedODEInputState(
                old_state, accumulated_stages, parameters, stage_time
            ),
            delta_t,
            butcher_diagonal_element,
        ).stage_update

    def compute_accumulated_stages(
        self, stage: int, stage_field: np.ndarray
    ) -> np.ndarray:
        """Calculates the weighted sum (according to the Butcher matrix) of the
        previous stage updates."""
        return np.tensordot(
            stage_field[:stage, :],
            self.butcher_tableau.butcher_matrix[stage, :stage],
            ((0,), (0,)),
        )

    def compute_step(
        self,
        delta_t: float,
        old_state: np.ndarray,
        stage_field: np.ndarray,
        remaining_time: float,
        error_history: np.ndarray = None,
        step_size_history: np.ndarray = None,
    ) -> (np.ndarray, float, bool, float):
        """Computes the next state and"""
        new_state = old_state.copy()
        new_state += np.tensordot(
            stage_field,
            delta_t * self.butcher_tableau.butcher_weight_vector,
            ((0,), (0,)),
        )
        if self.butcher_tableau.is_embedded and self.use_adaptive_time_stepping:
            new_state_embedded = old_state.copy()
            new_state_embedded += np.tensordot(
                stage_field,
                delta_t * self.butcher_tableau.butcher_adaptive_weights,
                ((0,), (0,)),
            )
            error_measure = self.error_measurer.get_measure(
                new_state - new_state_embedded,
                new_state,
                self.ode,
            )
            error_controller_status = self.error_controller(
                error_measure=error_measure,
                delta_t=delta_t,
                remaining_time=remaining_time,
                error_history=error_history,
                step_size_history=step_size_history,
            )
            return (
                new_state,
                error_controller_status.step_size_suggestion,
                error_controller_status.acceptance,
                error_measure,
            )
        elif not self.butcher_tableau.is_embedded and self.use_adaptive_time_stepping:
            raise RungeKuttaError(
                "Impossible to run adaptive scheme on non-embedded butcher tableau."
            )
        return new_state, delta_t, True, np.nan

    def compute_stage_jacvec(
        self,
        stage: int,
        delta_t: float,
        old_state_perturbation: np.ndarray,
        accumulated_stages_perturbation: np.ndarray,
        parameter_perturbations: np.ndarray,
        linearization_cache=None,
    ) -> np.ndarray:
        """Computes the matrix-vector-product of the jacobian of the stage wrt. to the
        old state and the accumulated stages."""
        if linearization_cache:
            self.ode.set_linearization_point(linearization_cache)

        butcher_diagonal_element = self.butcher_tableau.butcher_matrix[stage, stage]
        return self.ode.compute_update_derivative(
            DiscretizedODEInputState(
                old_state_perturbation,
                accumulated_stages_perturbation,
                parameter_perturbations,
                0.0,  # currently not used
            ),
            delta_t,
            butcher_diagonal_element,
        ).stage_update

    def compute_accumulated_stage_perturbations(
        self, stage: int, stage_perturbation_field: np.ndarray
    ) -> np.ndarray:
        """Calculates the weighted sum (according to the Butcher matrix) of the
        perturbations of the previous stage variables."""
        return self.compute_accumulated_stages(
            stage=stage, stage_field=stage_perturbation_field
        )

    def compute_step_jacvec(
        self,
        delta_t: float,
        old_state_perturbation: np.ndarray,
        stage_perturbation_field,
    ) -> np.ndarray:
        """Joins the perturbations of the old state and of the stage variables to the
        perturbations of new state."""
        new_state = old_state_perturbation.copy()
        new_state += np.tensordot(
            stage_perturbation_field,
            delta_t * self.butcher_tableau.butcher_weight_vector,
            ((0,), (0,)),
        )
        return new_state

    def compute_stage_transposed_jacvec(
        self,
        stage: int,
        delta_t: float,
        joined_perturbation: np.ndarray,
        linearization_cache=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes the matrix-vector-product of the transposed of the jacobian of the
        stage wrt. to the old state and the accumulated stages."""
        if linearization_cache:
            self.ode.set_linearization_point(linearization_cache)

        butcher_diagonal_element = self.butcher_tableau.butcher_matrix[stage, stage]
        result = self.ode.compute_update_adjoint_derivative(
            DiscretizedODEResultState(
                joined_perturbation,
                0.0,  # currently not used
                0.0,  # currently not used
            ),
            delta_t,
            butcher_diagonal_element,
        )
        return result.step_input, result.stage_input, result.independent_input

    def join_perturbations(
        self,
        stage: int,
        new_state_perturbation: np.ndarray,
        accumulated_stages_perturbation_field: np.ndarray,
    ) -> np.ndarray:
        """Joins perturbations of the new state of the accumulated stages."""
        joined_perturbation = (
            self.butcher_tableau.butcher_weight_vector[stage] * new_state_perturbation
        )
        joined_perturbation += np.tensordot(
            self.butcher_tableau.butcher_matrix[stage + 1 :, stage],
            accumulated_stages_perturbation_field[stage + 1 :, :],
            axes=((0,), (0,)),
        )
        return joined_perturbation

    @staticmethod
    def compute_step_transposed_jacvec(
        delta_t: float,
        new_state_perturbation: np.ndarray,
        stage_perturbation_field: np.ndarray,
    ) -> np.ndarray:
        """Currently not needed, keeping for later if necessary."""
        return new_state_perturbation + delta_t * stage_perturbation_field.sum(axis=0)
