"""Interface and implementations of criteria for terminating time integration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

# from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE
# from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
#     TimeDiscretizationSchemeInterface,
# )
from rkopenmdao.time_integration_state import TimeIntegrationState


class TerminationCriterion(ABC):
    """
    Abstract interface for termination criteria of time integrations.
    """

    # pylint: disable=too-few-public-methods

    @abstractmethod
    def is_iteration_finished(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        # The following two will be introduced once the TimeIntegrationInterface is
        # implemented.
        # ode: DiscretizedODE,
        # discretization_scheme: TimeDiscretizationSchemeInterface,
    ) -> bool:
        """
        Check whether to stop the time integration or not.

        Parameters
        ----------
        iteration: int
            The current iteration of the time integration.
        time_integration_state: TimeIntegrationState
            Current state of the time integration.
        """


@dataclass
class PredefinedNumberOfSteps(TerminationCriterion):
    """
    Termination criterion for stopping after a predefined number of time steps has
    passed.

    Parameters
    ----------
    number_of_step: int
        The number of steps the time integration has to perform.
    """

    number_of_steps: int

    def is_iteration_finished(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        # ode: DiscretizedODE,
        # discretization_scheme: TimeDiscretizationSchemeInterface,
    ) -> bool:
        return iteration >= self.number_of_steps


@dataclass
class PredefinedFinalTime(TerminationCriterion):
    """
    Termination criterion for stopping after a predefined simulation time has passed.

    Parameters
    ----------
    termination_time: float
        SImulation time after which to terminate the time integration.
    """

    termination_time: float

    def is_iteration_finished(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        # ode: DiscretizedODE,
        # discretization_scheme: TimeDiscretizationSchemeInterface,
    ) -> bool:
        # Currently hard-coded on Runge-Kutta. Once abstracted away, it should look
        # like this.
        # time = discretization_scheme.time_discretization_finalization_scheme(
        #     ode,
        #     time_integration_state.discretization_state,
        #     time_integration_state.step_size_history[0],
        # )
        time = time_integration_state.discretization_state.final_time
        return self.remaining_time(time) <= 0

    def remaining_time(self, current_time: float) -> float:
        """
        Remaining simulation time of the time integration.

        Parameters
        ----------
        current_time: float
            Current simulation time.
        """
        return self.termination_time - current_time
