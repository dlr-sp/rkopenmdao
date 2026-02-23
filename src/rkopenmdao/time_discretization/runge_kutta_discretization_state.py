""""""

from __future__ import annotations

import numpy as np


from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationStateInterface,
)


class RungeKuttaDiscretizationState(TimeDiscretizationStateInterface):

    start_state: np.ndarray
    stage_states: np.ndarray
    stage_updates: np.ndarray
    final_state: np.ndarray

    independent_inputs: np.ndarray

    stage_independent_outputs: np.ndarray
    final_independent_outputs: np.ndarray

    start_time: np.ndarray
    stage_times: np.ndarray
    final_time: np.ndarray

    step_size: np.ndarray

    linearization_points: np.ndarray

    def __init__(
        self,
        ode_state_size: int,
        independent_input_size: int,
        independent_output_size: int,
        number_of_stages: int,
        linearization_point_size: int,
    ):

        self.start_state = np.zeros(ode_state_size)
        self.stage_states = np.zeros((number_of_stages, ode_state_size))
        self.stage_updates = np.zeros((number_of_stages, ode_state_size))
        self.final_state = np.zeros(ode_state_size)

        self.independent_inputs = np.zeros(independent_input_size)

        self.stage_independent_outputs = np.zeros(
            (number_of_stages, independent_output_size)
        )
        self.final_independent_outputs = np.zeros(independent_output_size)

        self.start_time = np.zeros(1)
        self.stage_times = np.zeros(number_of_stages)
        self.final_time = np.zeros(1)

        self.step_size = np.zeros(1)

        self.linearization_points = np.zeros(
            (number_of_stages, linearization_point_size)
        )

    def set(self, other: RungeKuttaDiscretizationState):
        """"""
        self.start_state[:] = other.start_state[:]
        self.stage_states[:] = other.stage_states[:]
        self.stage_updates[:] = other.stage_updates[:]
        self.final_state[:] = other.final_state[:]
        self.independent_inputs[:] = other.independent_inputs[:]
        self.stage_independent_outputs[:] = other.stage_independent_outputs[:]
        self.final_independent_outputs[:] = other.final_independent_outputs[:]
        self.start_time[:] = other.start_time[:]
        self.stage_times[:] = other.stage_times[:]
        self.final_time[:] = other.final_time[:]
        self.step_size[:] = other.step_size[:]
        self.linearization_points[:] = other.linearization_points[:]

    def to_dict(self) -> dict:
        time_step_dict = {"start_state": self.start_state}
        time_step_dict["start_time"] = self.start_time
        number_of_stages = self.stage_states.shape[0]
        time_step_dict["stage_times"] = self.stage_times
        for i in range(number_of_stages):
            time_step_dict[f"stage_{i}"] = {
                "state": self.stage_states[i, :],
                "update": self.stage_updates[i, :],
                "independent_output": self.stage_independent_outputs[i, :],
            }
        time_step_dict["final_state"] = self.final_state
        time_step_dict["independent_input"] = self.independent_inputs
        time_step_dict["final_independent_output"] = self.final_independent_outputs
        time_step_dict["final_time"] = self.final_time
        time_step_dict["linearization_points"] = self.linearization_points

        return time_step_dict

    @classmethod
    def from_dict(cls, state_dict: dict):
        number_of_stages = len(state_dict["linearization_points"])
        state = cls(
            state_dict["final_state"].size,
            state_dict["independent_input"],
            state_dict["stage_0"]["independent_output"].size,
            number_of_stages,
            state_dict["linearization_points"].shape[1],
        )
        state.start_state = state_dict["start_state"]
        state.start_time = state_dict["start_time"]
        state.stage_times = state_dict["stage_times"]
        for i in range(number_of_stages):
            state.stage_states[i, :] = state_dict[f"stage_{i}"]["state"]
            state.stage_updates[i, :] = state_dict[f"stage_{i}"]["update"]
            state.stage_independent_outputs[i, :] = state_dict[f"stage_{i}"][
                "independent_output"
            ]
        state.final_state = state_dict["final_state"]
        state.independent_inputs = state_dict["independent_input"]
        state.final_independent_outputs = state_dict["final_independent_output"]
        state.final_time = state_dict["final_time"]
        state.linearization_points = state_dict["linearization_points"]
        return state


class EmbeddedRungeKuttaDiscretizationState(RungeKuttaDiscretizationState):
    embedded_state: np.ndarray
    error_estimate: np.ndarray

    def __init__(
        self,
        ode_state_size,
        independent_input_size,
        independent_output_size,
        number_of_stages,
        linearization_point_size,
    ):
        super().__init__(
            ode_state_size,
            independent_input_size,
            independent_output_size,
            number_of_stages,
            linearization_point_size,
        )
        self.embedded_state = np.zeros_like(self.start_state)
        self.error_estimate = np.zeros_like(self.start_state)

    def to_dict(self):
        time_step_dict = super().to_dict()
        time_step_dict["embedded_state"] = self.embedded_state
        time_step_dict["error_estimate"] = self.error_estimate
        return time_step_dict

    @classmethod
    def from_dict(cls, state_dict):
        state = super().from_dict(state_dict)
        state.embedded_state = state_dict["embedded_dict"]
        state.error_estimate = state_dict["error_estimate"]
