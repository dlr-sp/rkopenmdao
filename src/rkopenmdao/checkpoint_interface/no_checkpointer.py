"""
Pseudo checkpointing implementation that does not actually checkpoint data, and thus is
unusable for reverse-mode computations.
"""

from dataclasses import dataclass

from rkopenmdao.checkpoint_interface.checkpoint_interface import CheckpointInterface
from rkopenmdao.time_integration_state import TimeIntegrationState


@dataclass
class NoCheckpointer(CheckpointInterface):
    """
    Checkpointer that sets no checkpoints, for cases where reverse mode computations
    are not needed.

    Parameters
    ----------
    integration_control: IntegrationControl
        IntegrationControl object for sharing data between ODE time discretization and
        time integration.
    run_step_func: Callable[[TimeIntegrationState], TimeIntegrationState]
        Function for the computation of one step of the forward (primal) time
        integration. Input is the state of the time integration at the start of the
        step, return value the state at the end of the same step.
    run_step_jacvec_rev_func: Callable[
        [TimeIntegrationState, TimeIntegrationState], TimeIntegrationState
    ]
        Function for the computation of one step of the reverse (linear) time
        integration. Inputs are the state of the time integration during the step
        acting as linearization point, as well as the perturbations for the end of the
        time step. Return value is the perturbation for the start of the time step.
    state: TimeIntegrationState
        Time integration state on which all computations for the forward (primal) time
        integration are performed.
    state_perturbation: TimeIntegrationState
        Time integration state on which all computations for the reverse (linear) time
        integration are performed.
    """

    def create_checkpointer(self):
        """Doesn't checkpoint, so does nothing"""

    def iterate_forward(
        self, initial_state: TimeIntegrationState
    ) -> TimeIntegrationState:
        """
        Runs time integration from start to finish, without saving checkpoints.

        Parameters
        ----------
        initial_state: TimeIntegrationState
            The state at the beginning of the time integration process.

        Returns
        -------
        final_state: TimeIntegrationState
            The resulting state after completing all time steps.
        """
        self.state.set(initial_state)
        while self.integration_control.termination_condition_status():
            self.state.set(self.run_step_func(self.state))
        return self.state

    def iterate_reverse(self, final_state_perturbation: TimeIntegrationState):
        """
        Does nothing.

        Parameters
        ----------
        final_state_perturbation: TimeIntegrationState
            The perturbation of the final state used as starting point for reverse
            iteration.

        Raises
        ------
        NotImplementedError
            If called at all.
        """
        raise NotImplementedError(
            "NoCheckpointer is explicitly for cases where no reverse mode is used."
            "If you need reverse mode, use another checkpointing implementation."
        )
