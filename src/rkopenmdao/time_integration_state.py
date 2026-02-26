"""Definition for one state of time integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rkopenmdao.time_discretization.runge_kutta_discretization_state import (
    RungeKuttaDiscretizationState,
)


@dataclass
class TimeIntegrationState:
    """
    Describes the state of one step of time integration:

    TODO: currently hard-coded on Runge-Kutta. Change that once general time
    integration interface is implemented.

    Parameters
    ----------
    discretization_state: RungeKuttaDiscretizationState
        State of the used time discretization.
    step_size_suggestion: np.ndarray
        suggestion for the step size of the next time step.
    step_size_history: np.ndarray
        Small window of history on the step sizes of previous time steps.
    error_history: np.ndarray
        Small window of history on the (estimated) error measures of previous
        time steps.
    """

    discretization_state: RungeKuttaDiscretizationState
    step_size_suggestion: np.ndarray
    step_size_history: np.ndarray
    error_history: np.ndarray

    def set(self, other: TimeIntegrationState):
        """
        Sets the contents of this instance of the time integration state to the
        contents of `other`. Note that it is *required* that this performs a copy into
        the existing internal data structures, as else checkpointing would break.

        Parameters
        ----------
        other: TimeIntegrationState
            State that contains the data copied over into the internal structures.
        """
        self.discretization_state.set(other.discretization_state)
        self.step_size_suggestion[:] = other.step_size_suggestion[:]
        self.step_size_history[:] = other.step_size_history[:]
        self.error_history[:] = other.error_history[:]

    def to_dict(self) -> dict:
        """
        Exports the internal data into a dict of numpy arrays.

        Returns
        -------
        time_step_dict: dict
            Internal data represented as dict of numpy arrays.
        """
        time_state_dict = {
            "discretization_state": self.discretization_state.to_dict(),
            "step_size_suggestion": self.step_size_suggestion,
            "step_size_history": self.step_size_history,
            "error_history": self.error_history,
        }
        return time_state_dict

    @classmethod
    def from_dict(cls, time_state_dict: dict):
        """
        Creates a new time integration state from a dict. Dicts created by `to_dict`
        must be supported by this method.

        Parameters
        ----------
        state_dict: dict
            Dictionary from which a time integration state is created.
        """
        return cls(
            RungeKuttaDiscretizationState.from_dict(
                time_state_dict["discretization_state"]
            ),
            time_state_dict["step_size_suggestion"][0],
            time_state_dict["step_size_history"],
            time_state_dict["error_history"],
        )
