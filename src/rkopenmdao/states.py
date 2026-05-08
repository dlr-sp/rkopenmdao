"""State types for time integration and discretized ODEs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class TimeDiscretizationStateInterface(ABC):
    """Interface required for general states of time discretizations."""

    @abstractmethod
    def set(self, other: TimeDiscretizationStateInterface):
        """
        Sets the contents of this instance of the time discretization state to the
        contents of `other`. Note that it is *required* that this performs a copy into
        the existing internal data structures, as else checkpointing would break.

        Parameters
        ----------
        other: TimeDiscetizationStateInterface
            State that contains the data copied over into the internal structures.
        """

    @abstractmethod
    def to_dict(self) -> dict[np.ndarray]:
        """
        Exports the internal data into a dict of numpy arrays.

        Returns
        -------
        time_step_dict: dict
            Internal data represented as dict of numpy arrays.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, state_dict: dict):
        """
        Creates a new time discretization state from a dict. Dicts created by `to_dict`
        must be supported by this method.

        Parameters
        ----------
        state_dict: dict
            Dictionary from which a time discretization state is created.
        """


@dataclass
class StartingValues:
    """
    Dataclass containing data for starting a time discretization.

    Parameters
    ----------
    initial_time: float
        Time at start of time integration.
    initial_values: np.ndarray
        Initial values for initial value problem.
    independent_inputs: np.ndarray
        Independent inputs for time integration.
    """

    initial_time: float
    initial_values: np.ndarray
    independent_inputs: np.ndarray


@dataclass
class FinalizationValues:
    """
    Dataclass containing data for finalizing a time discretization.

    Parameters
    ----------
    final_time: float
        Time at end of time integration.
    final_values: np.ndarray
        Values for end of time integration.
    final_independent_outputs: np.ndarray
        Independent outputs at end of time integration.
    """

    final_time: float
    final_values: np.ndarray
    final_independent_outputs: np.ndarray


@dataclass
class DiscretizedODEInputState:
    """
    Dataclass containing all the information about the input state of a discretized ODE.
    An instance of this class is used as an input for a non-differentiated run and for a
    differentiated run in forward mode. A run for reverse-mode derivatives has an
    instance of this class as result.

    Parameters
    ----------
    step_input: np.ndarray
        A vector corresponding to the input or a perturbation of the input data coming
        from the start of a time step.
    stage_input: np.ndarray
        A vector corresponding to the input or a perturbation of the input data coming
        from the start of a time stage.
    independent_input: np.ndarray
        A vector corresponding to the input or a perturbation of the input data that is
        independent of the time.
    time: float
        The time the ODE is evaluated at.
    linearization_point: np.ndarray | None
        Optional linearization point for use in differentiated operations.
    """

    step_input: np.ndarray
    stage_input: np.ndarray
    independent_input: np.ndarray
    time: float
    linearization_point: np.ndarray | None = None


@dataclass
class DiscretizedODEResultState:
    """
    Dataclass containing all the information about the resulting state of a discretized
    ODE. An instance of this class is the result of both a non-differentiated run and
    for a differentiated run in forward mode. A run for reverse-mode derivatives needs
    an instance of this class as input.

    Parameters
    ----------
    stage_update: np.ndarray
        A vector corresponding to the output or a perturbation of the output data coming
        from the update of a time stage.
    stage_state: np.ndarray
        A vector corresponding to the output or a perturbation of the output data coming
        from the state of a time stage.
    independent_output: np.ndarray
        A vector corresponding to the output or a perturbation of the output data that
        is not directly dependent of the time integration (i.e. there is no time
        derivative for the contained data in the ODE system).
    linearization_point: np.ndarray | None
        Optional linearization point for use in differentiated operations.
    """

    stage_update: np.ndarray
    stage_state: np.ndarray
    independent_output: np.ndarray
    linearization_point: np.ndarray | None = None


@dataclass
class TimeIntegrationState:
    """
    Describes the state of one step of time integration

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

    discretization_state: TimeDiscretizationStateInterface
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
    def from_dict(
        cls,
        time_state_dict: dict,
        discretization_state_type: type[TimeDiscretizationStateInterface],
    ):
        """
        Creates a new time integration state from a dict. Dicts created by `to_dict`
        must be supported by this method, provided they used the same type for the
        state of the discretization.

        Parameters
        ----------
        state_dict: dict
            Dictionary from which a time integration state is created.
        discretization_state_type: type[TimeDiscretizationStateInterface]
            Type of the discretization state used by the dict.
        """
        return cls(
            discretization_state_type.from_dict(
                time_state_dict["discretization_state"]
            ),
            time_state_dict["step_size_suggestion"][0],
            time_state_dict["step_size_history"],
            time_state_dict["error_history"],
        )
