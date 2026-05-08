"""
Pseudo checkpointing implementation that does not actually checkpoint data, and thus is
unusable for reverse-mode computations.
"""

from dataclasses import dataclass

from rkopenmdao.checkpointed_time_integration.checkpointed_time_integration import (
    CheckpointedTimeIntegration,
)
from rkopenmdao.states import TimeIntegrationState


@dataclass
class NoCheckpointTimeIntegration(CheckpointedTimeIntegration):
    """
    TODO

    Parameters
    ----------
    TODO
    """

    def integrate(
        self, initial_state: TimeIntegrationState
    ) -> list[TimeIntegrationState]:
        """
        TODO
        """
        iteration = 0
        while not self.time_integration_config.termination_criterion.is_iteration_finished(
            iteration, initial_state, self.ode, self.time_discretization_scheme
        ):
            iteration += 1
            self._run_step(iteration, initial_state)
        return [initial_state]

    def integrate_adjoint_derivative(
        self,
        initial_state: TimeIntegrationState,
        final_state_perturbations: list[TimeIntegrationState],
    ) -> TimeIntegrationState:
        """
        TODO
        """
        raise NotImplementedError(
            "NoCheckpointTimeIntegration is explicitly for cases where no reverse mode"
            " is used. If you need reverse mode, use another time integration "
            "implementation."
        )
