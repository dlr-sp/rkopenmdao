"""Forward-only pseudo checkpointing strategy without reverse mode support."""

from dataclasses import dataclass

from rkopenmdao.checkpointed_time_integration.checkpointed_time_integration import (
    CheckpointedTimeIntegration,
)
from rkopenmdao.states import TimeIntegrationState


@dataclass
class NoCheckpointTimeIntegration(CheckpointedTimeIntegration):
    """
    Forward-only integration without checkpointing support.

    This implementation performs standard forward time integration without storing
    any intermediate states for reverse-mode computation. It is the simplest and
    most memory-efficient strategy but does not support adjoint (reverse-mode)
    derivative computations.

    This class is intended for use cases whereo nly primal integration is needed

    See Also
    --------
    PyrevolveTimeIntegration : Memory-efficient checkpointing with reverse support
    AllCheckpointTimeIntegration : Full storage with reverse support
    CheckpointedTimeIntegration : Base class with common functionality
    """

    def integrate(
        self, initial_state: TimeIntegrationState
    ) -> list[TimeIntegrationState]:
        """
        Performs forward integration without checkpointing.

        Parameters
        ----------
        initial_state : TimeIntegrationState
            The initial state for integration.

        Returns
        -------
        list[TimeIntegrationState]
            List containing the final state after completing all time steps.
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
        Raises NotImplementedError - reverse mode not supported.

        This class does not support reverse-mode (adjoint) integration because
        it does not store intermediate states for checkpointing.

        Parameters
        ----------
        initial_state : TimeIntegrationState
            The initial state (unused - raise happens before use).
        final_state_perturbations : list[TimeIntegrationState]
            The final state perturbations (unused - raise happens before use).

        Raises
        ------
        NotImplementedError
            Always raised to indicate that reverse-mode integration is not supported.
        """
        raise NotImplementedError(
            "NoCheckpointTimeIntegration is explicitly for cases where no reverse mode"
            " is used. If you need reverse mode, use another time integration "
            "implementation."
        )
