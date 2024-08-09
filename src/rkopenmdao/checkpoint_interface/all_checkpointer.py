from collections import deque
from dataclasses import dataclass
import numpy as np

from .checkpoint_interface import CheckpointInterface


@dataclass
class AllCheckpointer(CheckpointInterface):
    """Checkpointer that sets checkpoints for all time steps. Memory inefficient, but can work
    with a variable number of time steps"""

    def __post_init__(self):
        self._state = np.zeros(self.array_size)
        self._serialized_state_perturbation = np.zeros(self.array_size)
        self._storage = deque()

    def create_checkpointer(self):
        self._storage.clear()

    def iterate_forward(self, initial_state: np.ndarray):
        self._state = initial_state
        for i in range(self.num_steps):
            self._storage.append(self._state.copy())
            self._state = self.run_step_func(i + 1, self._state.copy())

    def iterate_reverse(self, final_state_perturbation: np.ndarray):
        self._serialized_state_perturbation = final_state_perturbation
        for i in reversed(range(self.num_steps)):
            self._serialized_state_perturbation = self.run_step_jacvec_rev_func(
                i + 1, self._storage.pop(), self._serialized_state_perturbation.copy()
            )
