from collections.abc import Callable
from dataclasses import dataclass


import numpy as np

from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.time_discretization.time_discretization_interface import (
    TimeDiscretizationInterface,
)


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

    time_planes: int
    step_time_indices: list
    step_time_update_indices: list
    independent_input_start: int
    independent_input_end: int
    state_start: int
    state_end: int
    stage_time_indices: list
    stage_time_update_indices: list
    accumulated_stages_start: list
    accumulated_stages_end: list
    stage_state_start: list
    stage_state_end: list
    stage_state_update_start: list
    stage_state_update_end: list
    stage_independent_output_start: list
    stage_independent_output_end: list

    def _create_state_ranges(self):
        for i in range(self.time_planes):
            self.step_time_indices[i] = 2 * i  # t_n-i
            self.step_time_update_indices[i] = 2 * i + 1  # dt_n-1
        # p_n: independent parameters
        self.independent_input_start = 2 * self.time_planes
        self.independent_input_end = (
            self.independent_input_start + self.independent_input_size
        )
        # x_n: state
        self.state_start = self.independent_input_end
        self.state_end = self.state_start + self.single_state_size
        for i in range(self.butcher_tableau.number_of_stages()):
            # t_n^i
            if i == 0:
                self.stage_time_indices[i] = self.state_end
            else:
                self.stage_time_indices[i] = self.stage_independent_output_end[i - 1]
            self.stage_time_update_indices[i] = self.stage_time_indices[i] + 1  # dt_n^i
            # s_i
            self.accumulated_stages_start[i] = self.stage_time_update_indices + 1
            self.accumulated_stages_end[i] = (
                self.accumulated_stages_start[i] + self.single_state_size
            )
            # x_n^i
            self.stage_state_start[i] = self.accumulated_stages_end[i]
            self.stage_state_end[i] = self.stage_state_start[i] + self.single_state_size
            # k_i
            self.stage_state_update_start[i] = self.stage_state_end[i]
            self.stage_state_update_end[i] = (
                self.stage_state_update_start[i] + self.single_state_size
            )
            # q_n^i
            self.stage_independent_output_start[i] = self.stage_state_update_end[i]
            self.stage_independent_output_end[i] = (
                self.stage_independent_output_start[i] + self.independent_output_size
            )

    def compute_step(self, step_input_state: np.ndarray, step_output_state: np.ndarray):
        for i in range(self.butcher_tableau.number_of_stages()):
            self._update_stage_time(i, step_input_state, step_output_state)
            self._accumulate_stages(i, step_output_state)
            self.compute_stage(i, step_input_state, step_output_state)
        self._step_closure(step_input_state, step_output_state)
        self._update_step_time(step_input_state, step_output_state)

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
        step_output_state[
            self.stage_state_update_start[stage] : self.stage_state_update_end[stage]
        ] = self.stage_computation_functor(
            step_input_state[self.state_start : self.state_end],
            step_output_state[
                self.accumulated_stages_start[stage] : self.accumulated_stages_end[
                    stage
                ]
            ],
            step_input_state[self.independent_input_start : self.independent_input_end],
            step_output_state[self.stage_time_indices[stage]],
            step_input_state[self.step_time_update_indices],
            self.butcher_tableau.butcher_matrix[stage, stage],
        )

    def compute_stage_forward_derivative(
        self,
        stage,
        step_input_state: np.ndarray,
        step_output_state: np.ndarray,
        step_input_perturbation: np.ndarray,
        step_output_perturbation: np.ndarray,
    ):
        step_output_perturbation[
            self.stage_state_update_start[stage] : self.stage_state_update_end[stage]
        ] = self.stage_computation_functor_jacvec(
            step_input_perturbation[self.state_start : self.state_end],
            step_output_perturbation[
                self.accumulated_stages_start[stage] : self.accumulated_stages_end[
                    stage
                ]
            ],
            step_input_perturbation[
                self.independent_input_start : self.independent_input_end
            ],
            step_output_state[self.stage_time_indices[stage]],
            step_input_state[self.step_time_update_indices],
            self.butcher_tableau.butcher_matrix[stage, stage],
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
            step_output_perturbation[
                self.stage_state_update_start : self.stage_state_update_end
            ],
            step_output_state[self.stage_time_indices[stage]],
            step_input_state[self.step_time_update_indices],
            self.butcher_tableau.butcher_matrix[stage, stage],
        )
        step_input_perturbation[self.state_start : self.state_end] += results[0]
        step_output_perturbation[
            self.accumulated_stages_start[stage] : self.accumulated_stages_end[stage]
        ] += results[1]
        step_input_perturbation[
            self.independent_input_start : self.independent_input_end
        ] += results[2]

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
        step_output_state[self.state_start : self.state_end] = step_input_state[
            self.state_start : self.state_end
        ]
        for i in range(self.butcher_tableau.number_of_stages()):
            step_output_state[self.state_start : self.state_end] += (
                step_input_state[self.step_time_update_indices[0]]
                * self.butcher_tableau.butcher_weight_vector[i]
                * step_output_state[
                    self.stage_state_update_start[i] : self.stage_state_update_end[i]
                ]
            )

    def _step_closure_forward_derivative(
        self, step_input_perturbation: np.ndarray, step_output_perturbation: np.ndarray
    ):
        self._step_closure(step_input_perturbation, step_output_perturbation)

    def _step_closure_reverse_derivative(
        self, step_input_perturbation: np.ndarray, step_output_perturbation: np.ndarray
    ):
        step_input_perturbation[
            self.state_start : self.state_end
        ] += step_output_perturbation[self.state_start : self.state_end]
        for i in range(self.butcher_tableau.number_of_stages()):
            step_output_perturbation[
                self.stage_state_update_start[i] : self.stage_state_update_end[i]
            ] += (
                step_input_perturbation[self.step_time_update_indices[0]]
                * self.butcher_tableau.butcher_weight_vector[i]
                * step_output_perturbation[self.state_start : self.state_end]
            )

    def _update_step_time(
        self, step_input_state: np.ndarray, step_output_state: np.ndarray
    ):
        step_output_state[self.step_time_indices[0]] = (
            step_input_state[self.step_time_indices[0]]
            + step_input_state[self.step_time_update_indices[0]]
        )
        step_output_state[self.step_time_update_indices[0]] = step_input_state[
            self.step_time_update_indices[0]
        ]

    def _update_stage_time(
        self, stage, step_input_state: np.ndarray, step_output_state: np.ndarray
    ):
        step_output_state[self.stage_time_indices[stage]] = (
            step_input_state[self.step_time_indices[0]]
            + self.butcher_tableau.butcher_time_stages[stage]
            * step_input_state[self.step_time_update_indices[0]]
        )
        step_output_state[self.stage_time_update_indices[stage]] = (
            self.butcher_tableau.butcher_time_stages[stage]
            * step_input_state[self.step_time_update_indices[0]]
        )

    def _accumulate_stages(
        self,
        stage: int,
        step_output_state: np.ndarray,
    ):
        for i in range(stage):
            step_output_state[
                self.accumulated_stages_start[stage] : self.accumulated_stages_end[
                    stage
                ]
            ] += (
                self.butcher_tableau.butcher_weight_vector[i]
                * step_output_state[
                    self.stage_state_update_start[i] : self.stage_state_update_end[i]
                ]
            )

    def _accumulate_stages_forward_derivative(
        self, stage, step_output_perturbation: np.ndarray
    ):
        self._accumulate_stages(stage, step_output_perturbation)

    def _accumulate_stages_reverse_derivative(
        self, stage, step_output_perturbation: np.ndarray
    ):
        for i in range(stage):
            step_output_perturbation[
                self.stage_state_update_start[i] : self.stage_state_update_end[i]
            ] += (
                self.butcher_tableau.butcher_weight_vector[i]
                * step_output_perturbation[
                    self.accumulated_stages_start[stage] : self.accumulated_stages_end[
                        stage
                    ]
                ]
            )
