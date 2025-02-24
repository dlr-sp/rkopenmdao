from collections.abc import Callable
from dataclasses import dataclass, field


import numpy as np

from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.time_discretization.time_discretization_interface import (
    TimeDiscretizationInterface,
)


class DIRKStateInfo:

    def __init__(
        self,
        time_planes: int,
        stages: int,
        independent_input_size: int,
        state_size: int,
        independent_output_size: int,
    ):
        self.current_time_index = 0
        self.current_time_update_index = 1
        self.update_indices = {
            time_plane: 1 + time_plane for time_plane in range(time_planes)
        }
        self.norm_indices = {
            time_plane: 1 + time_planes + time_plane
            for time_plane in range(time_planes)
        }
        self.independent_input_slice = slice(
            max(2, 1 + 2 * time_planes),
            max(2, 1 + 2 * time_planes) + independent_input_size,
        )
        self.state_slice = slice(
            self.independent_input_slice.stop,
            self.independent_input_slice.stop + state_size,
        )
        self.stage_time_indices = {
            stage: self.state_slice.stop + stage for stage in range(stages)
        }

        self.stage_time_update_indices = {
            stage: self.state_slice.stop + stages + stage for stage in range(stages)
        }

        self.accumulated_stage_slices = {
            stage: slice(
                self.state_slice.stop + 2 * stages + stage * state_size,
                self.state_slice.stop + 2 * stages + (stage + 1) * state_size,
            )
            for stage in range(stages)
        }

        self.stage_state_slices = {
            stage: slice(
                self.accumulated_stage_slices[stages - 1].stop + stage * state_size,
                self.accumulated_stage_slices[stages - 1].stop
                + (stage + 1) * state_size,
            )
            for stage in range(stages)
        }

        self.stage_update_slices = {
            stage: slice(
                self.stage_state_slices[stages - 1].stop + stage * state_size,
                self.stage_state_slices[stages - 1].stop + (stage + 1) * state_size,
            )
            for stage in range(stages)
        }

        self.stage_independent_output_slices = {
            stage: slice(
                self.stage_update_slices[stages - 1].stop
                + stage * independent_output_size,
                self.stage_update_slices[stages - 1].stop
                + (stage + 1) * independent_output_size,
            )
            for stage in range(stages)
        }
        self.total_size = self.stage_independent_output_slices[stages - 1].stop + 1


@dataclass
class DiagonallyImplicitRungeKuttaDiscretization(TimeDiscretizationInterface):
    """"""

    butcher_tableau: ButcherTableau
    stage_computation_functor: Callable[
        [np.ndarray, np.ndarray, np.ndarray, float, float, float], np.ndarray
    ]
    stage_computation_functor_jacvec: Callable[
        [np.ndarray, np.ndarray, np.ndarray, float, float, float], np.ndarray
    ]
    stage_computation_functor_transposed_jacvec: Callable[
        [np.ndarray, float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]
    ]
    _state_info: DIRKStateInfo = field(init=False)

    def __post_init__(self):
        self._state_info = DIRKStateInfo(
            0,
            self.butcher_tableau.number_of_stages(),
            self.independent_input_size,
            self.single_state_size,
            self.independent_output_size,
        )

    def create_zero_state(self):
        return np.zeros(self._state_info.total_size)

    def create_initial_state(
        self,
        initial_time: float,
        initial_time_update: float,
        initial_value: np.ndarray,
        independent_inputs: np.ndarray,
    ) -> np.ndarray:
        initial_state = self.create_zero_state()
        initial_state[self._state_info.current_time_index] = initial_time
        initial_state[self._state_info.current_time_update_index] = initial_time_update
        initial_state[self._state_info.state_slice] = initial_value
        initial_state[self._state_info.independent_input_slice] = independent_inputs
        return initial_state

    def export_final_state_state(
        self, step_output_state: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        return (
            step_output_state[self._state_info.current_time_index],
            step_output_state[self._state_info.current_time_update_index],
            step_output_state[self._state_info.state_slice],
        )

    def compute_step(self, step_input_state: np.ndarray, step_output_state: np.ndarray):
        for i in range(self.butcher_tableau.number_of_stages()):
            self._update_stage_time(i, step_input_state, step_output_state)
            self._accumulate_stages(i, step_output_state)
            self.compute_stage(i, step_input_state, step_output_state)
        self._step_closure(step_input_state, step_output_state)
        self._update_step_time(step_input_state, step_output_state)
        step_output_state[self._state_info.independent_input_slice] = step_input_state[
            self._state_info.independent_input_slice
        ]

    def compute_step_forward_derivative(
        self,
        step_input_state: np.ndarray,
        step_output_state: np.ndarray,
        step_input_perturbation: np.ndarray,
        step_output_perturbation: np.ndarray,
    ):
        for i in range(self.butcher_tableau.number_of_stages()):
            self._update_stage_time(i, step_input_state, step_output_state)
            self._accumulate_stages(i, step_output_state)
            self._accumulate_stages_forward_derivative(i, step_output_perturbation)
            self.compute_stage(i, step_input_state, step_output_state)
            self.compute_stage_forward_derivative(
                i,
                step_input_state,
                step_output_state,
                step_input_perturbation,
                step_output_perturbation,
            )
        self._step_closure(step_input_state, step_output_state)
        self._step_closure_forward_derivative(
            step_input_perturbation, step_output_perturbation
        )
        self._update_step_time(step_input_state, step_output_state)

    def compute_step_reverse_derivative(
        self,
        step_input_state: np.ndarray,
        step_output_state: np.ndarray,
        step_input_perturbation: np.ndarray,
        step_output_perturbation: np.ndarray,
    ):
        # TODO add forward iteration
        self._step_closure_reverse_derivative(
            step_input_perturbation, step_output_perturbation
        )
        for i in reversed(range(self.butcher_tableau.number_of_stages())):
            self._update_stage_time(i, step_input_state, step_output_state)
            self.compute_stage_reverse_derivative(
                i,
                step_input_state,
                step_output_state,
                step_input_perturbation,
                step_output_perturbation,
            )
            self._accumulate_stages_reverse_derivative(i, step_output_perturbation)

    def compute_stage(
        self, stage, step_input_state: np.ndarray, step_output_state: np.ndarray
    ):
        step_output_state[self._state_info.stage_update_slices[stage]] = (
            self.stage_computation_functor(
                step_input_state[self._state_info.state_slice],
                step_output_state[self._state_info.accumulated_stage_slices[stage]],
                step_input_state[self._state_info.independent_input_slice],
                step_output_state[self._state_info.stage_time_indices[stage]],
                step_input_state[self._state_info.current_time_update_index],
                self.butcher_tableau.butcher_matrix[stage, stage],
            )
        )

    def compute_stage_forward_derivative(
        self,
        stage,
        step_input_state: np.ndarray,
        step_output_state: np.ndarray,
        step_input_perturbation: np.ndarray,
        step_output_perturbation: np.ndarray,
    ):
        step_output_perturbation[self._state_info.stage_update_slices[stage]] = (
            self.stage_computation_functor_jacvec(
                step_input_perturbation[self._state_info.state_slice],
                step_output_perturbation[
                    self._state_info.accumulated_stage_slices[stage]
                ],
                step_input_perturbation[self._state_info.independent_input_slice],
                step_output_state[self._state_info.stage_time_indices[stage]],
                step_input_state[self._state_info.current_time_update_index],
                self.butcher_tableau.butcher_matrix[stage, stage],
            )
        )

    def compute_stage_reverse_derivative(
        self,
        stage,
        step_input_state: np.ndarray,
        step_output_state: np.ndarray,
        step_input_perturbation: np.ndarray,
        step_output_perturbation: np.ndarray,
    ):
        results = self.stage_computation_functor_transposed_jacvec(
            step_output_perturbation[self._state_info.stage_state_slices[stage]],
            step_output_state[self._state_info.stage_time_indices[stage]],
            step_input_state[self._state_info.current_time_update_index],
            self.butcher_tableau.butcher_matrix[stage, stage],
        )
        step_input_perturbation[self._state_info.state_slice] += results[0]
        step_output_perturbation[
            self._state_info.accumulated_stage_slices[stage]
        ] += results[1]
        step_input_perturbation[self._state_info.independent_input_slice] += results[2]

    def step_residual(
        self,
        step_input_state: np.ndarray,
        step_output_state: np.ndarray,
        step_residual_state: np.ndarray,
    ):
        raise NotImplementedError("Currently not implemented.")

    def stage_residual(
        self,
        stage_input_state: np.ndarray,
        stage_output_state: np.ndarray,
        stage_residual_state: np.ndarray,
    ):
        raise NotImplementedError("Currently not implemented.")

    def _step_closure(
        self, step_input_state: np.ndarray, step_output_state: np.ndarray
    ):
        step_output_state[self._state_info.state_slice] = step_input_state[
            self._state_info.state_slice
        ]
        for stage in range(self.butcher_tableau.number_of_stages()):
            step_output_state[self._state_info.state_slice] += (
                step_input_state[self._state_info.current_time_update_index]
                * self.butcher_tableau.butcher_weight_vector[stage]
                * step_output_state[self._state_info.stage_update_slices[stage]]
            )

    def _step_closure_forward_derivative(
        self, step_input_perturbation: np.ndarray, step_output_perturbation: np.ndarray
    ):
        self._step_closure(step_input_perturbation, step_output_perturbation)

    def _step_closure_reverse_derivative(
        self, step_input_perturbation: np.ndarray, step_output_perturbation: np.ndarray
    ):
        step_input_perturbation[
            self._state_info.state_slice
        ] += step_output_perturbation[self._state_info.state_slice]
        for stage in range(self.butcher_tableau.number_of_stages()):
            step_output_perturbation[self._state_info.stage_update_slices[stage]] += (
                step_input_perturbation[self._state_info.current_time_update_index]
                * self.butcher_tableau.butcher_weight_vector[stage]
                * step_output_perturbation[self._state_info.state_slice]
            )

    def _update_step_time(
        self, step_input_state: np.ndarray, step_output_state: np.ndarray
    ):
        step_output_state[self._state_info.current_time_index] = (
            step_input_state[self._state_info.current_time_index]
            + step_input_state[self._state_info.current_time_update_index]
        )
        step_output_state[self._state_info.current_time_update_index] = (
            step_input_state[self._state_info.current_time_update_index]
        )

    def _update_stage_time(
        self, stage, step_input_state: np.ndarray, step_output_state: np.ndarray
    ):
        step_output_state[self._state_info.stage_time_indices[stage]] = (
            step_input_state[self._state_info.current_time_index]
            + self.butcher_tableau.butcher_time_stages[stage]
            * step_input_state[self._state_info.current_time_update_index]
        )
        step_output_state[self._state_info.stage_time_update_indices[stage]] = (
            self.butcher_tableau.butcher_time_stages[stage]
            * step_input_state[self._state_info.current_time_update_index]
        )

    def _accumulate_stages(
        self,
        stage: int,
        step_output_state: np.ndarray,
    ):
        for prev_stage in range(stage):
            step_output_state[self._state_info.accumulated_stage_slices[stage]] += (
                self.butcher_tableau.butcher_weight_vector[prev_stage]
                * step_output_state[self._state_info.stage_update_slices[prev_stage]]
            )

    def _accumulate_stages_forward_derivative(
        self, stage, step_output_perturbation: np.ndarray
    ):
        self._accumulate_stages(stage, step_output_perturbation)

    def _accumulate_stages_reverse_derivative(
        self, stage, step_output_perturbation: np.ndarray
    ):
        for prev_stage in range(stage):
            step_output_perturbation[
                self._state_info.stage_update_slices[prev_stage]
            ] += (
                self.butcher_tableau.butcher_weight_vector[prev_stage]
                * step_output_perturbation[
                    self._state_info.accumulated_stage_slices[stage]
                ]
            )
