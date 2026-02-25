# pylint: disable=missing-module-docstring
from collections import deque
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .checkpoint_interface import CheckpointInterface


@dataclass
class AllCheckpointer(CheckpointInterface):
    """Checkpointer that sets checkpoints for all time steps. Memory inefficient, but
    fast and could in the future work with a variable number of time steps."""

    def __post_init__(self):
        """Reserves memory for time integration state and perturbation."""
        self._storage = deque()  # queue of TimeIntegrationStates

    def create_checkpointer(self):
        """Resets internal storage such that checkpointing can begin anew."""
        self._storage.clear()

    def iterate_forward(self, initial_state):
        """Runs time integration from start to finish."""
        self.state.set(initial_state)
        while self.integration_control.termination_condition_status():
            self.state.set(self.run_step_func(self.state))
            self._storage.append(deepcopy(self.state))
        return self.state

    def iterate_reverse(self, final_state_perturbation):
        """Goes backwards through time using the internal storage to calculate the
        reverse derivative."""
        self._serialized_state_perturbation = final_state_perturbation
        while self.integration_control.step > 0:
            state = self._storage.pop()
            self.state_perturbation.set(
                self.run_step_jacvec_rev_func(state, self.state_perturbation)
            )
            self.integration_control.decrement_step()
        return self.state_perturbation
