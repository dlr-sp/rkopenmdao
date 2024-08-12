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
        self._storage = deque()

    def create_checkpointer(self):
        """Resets internal storage such that checkpointing can begin anew."""
        self._storage.clear()

    def iterate_forward(self, initial_state: np.ndarray):
        """Runs time intgration from start to finish."""
        self._state = initial_state
        for i in range(self.num_steps):
            self._storage.append(self._state.copy())
            self._state = self.run_step_func(i + 1, self._state.copy())

    def iterate_reverse(self, final_state_perturbation: np.ndarray):
        """Goes backwards through time using the internal storage to calculate the
        reverse derivate."""
        self._serialized_state_perturbation = final_state_perturbation
        for i in reversed(range(self.num_steps)):
            self._serialized_state_perturbation = self.run_step_jacvec_rev_func(
                i + 1, self._storage.pop(), self._serialized_state_perturbation.copy()
            )
