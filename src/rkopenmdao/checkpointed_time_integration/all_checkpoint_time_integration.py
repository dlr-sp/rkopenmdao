"""Full storage checkpointing strategy for reverse-mode integration."""

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
    Full storage checkpointing strategy for reverse-mode integration.

    This implementation stores ALL intermediate states in memory during forward
    integration. During reverse integration, it retrieves states from storage
    without any recomputation, making reverse integration as fast as forward.

    Memory and computational tradeoffs:
    - Memory: O(n_steps) - stores full state at each time step
    - Computation: O(1) forward + O(1) reverse - no recomputation
    - Best for: Small problems, prototyping, or when memory is abundant

    For larger problems, consider:
    - PyrevolveTimeIntegration: O(sqrt(n_steps)) memory, O(log(n_steps)) recomputation
    - NoCheckpointTimeIntegration: O(1) memory, no reverse support
    """

    _storage: deque = field(init=False, default_factory=deque)

    def integrate(
        self, initial_state: TimeIntegrationState
    ) -> list[TimeIntegrationState]:
        """
        Performs forward integration with full state storage.

        Stores a complete copy of the state after each time step for use
        during reverse integration.

        Parameters
        ----------
        initial_state : TimeIntegrationState
            The initial state for integration.

        Returns
        -------
        list[TimeIntegrationState]
            List containing the final state after completing all time steps.

        Notes
        -----
        Memory usage grows linearly with the number of time steps:
        storage_size = n_steps × state_size

        For problems with many steps, this may become prohibitively large.
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
        Performs reverse integration using stored states.

        Retrieves previously stored states from memory and performs reverse
        integration without any recomputation.

        Parameters
        ----------
        initial_state : TimeIntegrationState
            The initial state from forward integration (linearization point).
            Used to verify that forward integration was performed, but not
            needed for the computation since states are stored.

        final_state_perturbations : list[TimeIntegrationState]
            List containing the final state perturbation (adjoint variable).
            Must contain exactly one element.

        Returns
        -------
        TimeIntegrationState
            The perturbation of the initial state after reverse integration.
            Contains the gradient of the cost function with respect to initial conditions.

        Notes
        -----
        This method retrieves stored states in reverse order and performs
        reverse integration through each time step. No recomputation is needed,
        making this very fast but memory-intensive.

        If the current storage is empty, calls integrate in oreder to populate it.

        See Also
        --------
        integrate : Forward integration that populates storage
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
