from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rkopenmdao.time_discretization.runge_kutta_discretization_state import (
    RungeKuttaDiscretizationState,
)


@dataclass
class TimeIntegrationState:
    discretization_state: RungeKuttaDiscretizationState
    step_size_suggestion: np.ndarray
    step_size_history: np.ndarray
    error_history: np.ndarray

    def set(self, other: TimeIntegrationState):
        self.discretization_state.set(other.discretization_state)
        self.step_size_suggestion[:] = other.step_size_suggestion[:]
        self.step_size_history[:] = other.step_size_history[:]
        self.error_history[:] = other.error_history[:]

    def to_dict(self) -> dict:
        time_state_dict = {
            "discretization_state": self.discretization_state.to_dict(),
            "step_size_suggestion": self.step_size_suggestion,
            "step_size_history": self.step_size_history,
            "error_history": self.error_history,
        }
        return time_state_dict

    @classmethod
    def from_dict(cls, time_state_dict: dict):
        return cls(
            RungeKuttaDiscretizationState.from_dict(
                time_state_dict["discretization_state"]
            ),
            time_state_dict["step_size_suggestion"][0],
            time_state_dict["step_size_history"],
            time_state_dict["error_history"],
        )
