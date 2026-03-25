from abc import ABC, abstractmethod
from dataclasses import dataclass

# from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE
# from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
#     TimeDiscretizationSchemeInterface,
# )
from rkopenmdao.time_integration_state import TimeIntegrationState


class TerminationCriterion(ABC):

    @abstractmethod
    def is_iteration_finished(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        # The following two will be introduced once the TimeIntegrationInterface is implemented.
        # ode: DiscretizedODE,
        # discretization_scheme: TimeDiscretizationSchemeInterface,
    ) -> bool:
        """ """


@dataclass
class PredefinedNumberOfSteps(TerminationCriterion):
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
        return self.termination_time - current_time
