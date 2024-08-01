# pylint: disable=missing-module-docstring

from typing import Callable, Tuple

import numpy as np


from rkopenmdao.butcher_tableau import ButcherTableau


class RungeKuttaScheme:
    """Implements functions used to apply a Runge-Kutta method a function represented by
    a functor. The current main use is with an OpenMDAO problem wrapped to such a
    functor."""

    def __init__(
        self,
        butcher_tableau: ButcherTableau,
        stage_computation_functor: Callable[
            [np.ndarray, np.ndarray, float, float, float], np.ndarray
        ],
        # old_state, accumulated_stages, stage_time, delta_t, butcher_diagonal_element
        # -> stage_state
        stage_computation_functor_jacvec: Callable[
            [np.ndarray, np.ndarray, float, float, float], np.ndarray
        ],
        # old_state_perturbation, accumulated_stages_perturbation, stage_time, delta_t,
        # butcher_diagonal_element
        # -> stage_perturbation
        stage_computation_functor_transposed_jacvec: Callable[
            [np.ndarray, float, float, float], Tuple[np.ndarray, np.ndarray]
        ],
        # stage_perturbation, stage_time, delta_t, butcher_diagonal_element
        # -> old_state_perturbation, accumulated_stages_perturbation
    ):
        self.butcher_tableau = butcher_tableau
        self.stage_computation_functor = stage_computation_functor
        self.stage_computation_functor_jacvec = stage_computation_functor_jacvec
        self.stage_computation_functor_transposed_jacvec = (
            stage_computation_functor_transposed_jacvec
        )

    def compute_stage(
        self,
        stage: int,
        delta_t: float,
        old_time: float,
        old_state: np.ndarray,
        accumulated_stages: np.ndarray,
    ) -> np.ndarray:
        """Computes the new stage variable based on the current information."""
        stage_time = (
            old_time + delta_t * self.butcher_tableau.butcher_time_stages[stage]
        )
        butcher_diagonal_element = self.butcher_tableau.butcher_matrix[stage, stage]
        return self.stage_computation_functor(
            old_state, accumulated_stages, stage_time, delta_t, butcher_diagonal_element
        )

    def compute_accumulated_stages(
        self, stage: int, stage_field: np.ndarray
    ) -> np.ndarray:
        """Calculates the weighted sum (according to the Butcher matrix) of the previous
        stage updates."""
        return np.tensordot(
            stage_field[:stage, :],
            self.butcher_tableau.butcher_matrix[stage, :stage],
            ((0,), (0,)),
        )

    def compute_step(
        self, delta_t: float, old_state: np.ndarray, stage_field: np.ndarray
    ) -> np.ndarray:
        """Joins the old state and the stage variables to the new state."""
        new_state = old_state.copy()
        new_state += np.tensordot(
            stage_field,
            delta_t * self.butcher_tableau.butcher_weight_vector,
            ((0,), (0,)),
        )

        return new_state

    def compute_stage_jacvec(
        self,
        stage: int,
        delta_t: float,
        old_time: float,
        old_state_perturbation: np.ndarray,
        accumulated_stages_perturbation: np.ndarray,
        **linearization_args
    ) -> np.ndarray:
        """Computes the matrix-vector-product of the jacobian of the stage wrt. to the
        old state and the accumulated stages."""
        if hasattr(self.stage_computation_functor_jacvec, "linearize"):
            self.stage_computation_functor_jacvec.linearize(**linearization_args)

        stage_time = (
            old_time + delta_t * self.butcher_tableau.butcher_time_stages[stage]
        )
        butcher_diagonal_element = self.butcher_tableau.butcher_matrix[stage, stage]
        return self.stage_computation_functor_jacvec(
            old_state_perturbation,
            accumulated_stages_perturbation,
            stage_time,
            delta_t,
            butcher_diagonal_element,
        )

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
        return self.compute_step(
            delta_t=delta_t,
            old_state=old_state_perturbation,
            stage_field=stage_perturbation_field,
        )

    def compute_stage_transposed_jacvec(
        self,
        stage: int,
        delta_t: float,
        old_time: float,
        joined_perturbation: np.ndarray,
        **linearization_args
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the matrix-vector-product of the transposed of the jacobian of the
        stage wrt. to the old state and the accumulated stages."""
        if hasattr(self.stage_computation_functor_transposed_jacvec, "linearize"):
            self.stage_computation_functor_transposed_jacvec.linearize(
                **linearization_args
            )

        stage_time = (
            old_time + delta_t * self.butcher_tableau.butcher_time_stages[stage]
        )
        butcher_diagonal_element = self.butcher_tableau.butcher_matrix[stage, stage]
        result = self.stage_computation_functor_transposed_jacvec(
            joined_perturbation, stage_time, delta_t, butcher_diagonal_element
        )
        return result

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

    def compute_step_transposed_jacvec(
        self,
        delta_t: float,
        new_state_perturbation: np.ndarray,
        stage_perturbation_field: np.ndarray,
    ) -> np.ndarray:
        """Currently not needed, keeping for later if necessary."""
        return new_state_perturbation + delta_t * stage_perturbation_field.sum(axis=0)
