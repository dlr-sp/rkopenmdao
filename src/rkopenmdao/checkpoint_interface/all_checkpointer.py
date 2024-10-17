# pylint: disable=missing-module-docstring
from collections import deque
from dataclasses import dataclass
import numpy as np

from .checkpoint_interface import CheckpointInterface


@dataclass
class AllCheckpointer(CheckpointInterface):
    """Checkpointer that sets checkpoints for all time steps. Memory inefficient, but
    fast and could in the future work with a variable number of time steps."""

    def __post_init__(self):
        """Reserves memory for time integration state and perturbation."""
        self._state = np.zeros(self.array_size)
        self._serialized_state_perturbation = np.zeros(self.array_size)
        self._storage = deque()  # queue of (state_i, delta_t_i) where _i is the step number

    def create_checkpointer(self):
        """Resets internal storage such that checkpointing can begin anew."""
        self._storage.clear()

    def iterate_forward(self, initial_state: np.ndarray):
        """Runs time intgration from start to finish."""
        self._state = initial_state
        if self.integration_control.termination_criterion.criterion == 'num_steps':
            while self.integration_control.step != self.integration_control.termination_criterion.value:
                self._storage.append((self._state.copy(), self.integration_control.delta_t))
                self._state = self.run_step_func(self._state.copy())
                self.integration_control.increment_step()
        elif self.integration_control.termination_criterion.criterion == 'end_time':
            while (np.abs(self.integration_control.remaining_time())
                   >= min(1e-13, self.integration_control.smallest_delta_t)):
                self._storage.append((self._state.copy(), self.integration_control.delta_t))
                self._state = self.run_step_func(self._state.copy())
                self.integration_control.increment_step()

    def iterate_reverse(self, final_state_perturbation: np.ndarray):
        """Goes backwards through time using the internal storage to calculate the
        reverse derivate."""
        self._serialized_state_perturbation = final_state_perturbation
        while self.integration_control.step != 0:
            _state, self.integration_control.delta_t = self._storage.pop()
            self._serialized_state_perturbation = self.run_step_jacvec_rev_func(
                _state, self._serialized_state_perturbation.copy()
            )
            self.integration_control.decrement_step()
