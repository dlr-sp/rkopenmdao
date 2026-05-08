"""Checkpointing implementation that saves all intermediate data."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

from rkopenmdao.checkpointed_time_integration.checkpointed_time_integration import (
    CheckpointedTimeIntegration,
)
from rkopenmdao.states import TimeIntegrationState


@dataclass
class AllCheckpointTimeIntegration(CheckpointedTimeIntegration):
    """
    TODO

    Parameters
    ----------
    TODO
    """

    _storage: deque = field(init=False, default_factory=deque)

    def create_checkpointer(self):
        """Resets internal storage such that checkpointing can begin anew."""
        self._storage.clear()

    def integrate(self, initial_state: TimeIntegrationState) -> TimeIntegrationState:
        """
        TODO
        """
        self._storage.clear()
        iteration = 0
        while not self.time_integration_config.termination_criterion.is_iteration_finished(
            iteration, initial_state, self.ode, self.time_discretization_scheme
        ):
            iteration += 1
            self._run_step(iteration, initial_state)
            self._storage.append(deepcopy(initial_state))
        return [initial_state]

    def integrate_adjoint_derivative(
        self,
        initial_state: TimeIntegrationState,
        final_state_perturbations: list[TimeIntegrationState],
    ) -> TimeIntegrationState:
        """
        TODO
        """
        if not self._storage:
            self.integrate(initial_state)
        final_state_perturbations = final_state_perturbations[0]
        iteration = len(self._storage)
        while self._storage:
            state = self._storage.pop()
            self._run_step_adjoint_derivative(
                iteration, state, final_state_perturbations
            )
            iteration -= 1
        return final_state_perturbations
