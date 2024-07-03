from collections import deque

import numpy as np

from .checkpoint_interface import CheckpointInterface


class AllCheckpointer(CheckpointInterface):
    """Checkpointer that sets checkpoints for all time steps. Memory inefficient, but can work
    with a variable number of time steps"""

    def __init__(self):
        self._array_size = 0
        self._num_steps = 0
        self._run_step = None
        self._run_step_jacvec_rev = None
        self._state = None
        self._serialized_state_perturbation = None
        self._storage = deque()

    def setup(self, **kwargs):
        self._array_size = kwargs["array_size"]
        self._num_steps = kwargs["num_steps"]
        self._run_step = kwargs["run_step_func"]
        self._run_step_jacvec_rev = kwargs["run_step_jacvec_rev_func"]
        self._state = np.zeros(self._array_size)

    def create_checkpointer(self):
        self._storage.clear()

    def iterate_forward(self, initial_state):
        self._state = initial_state.copy()
        for i in range(self._num_steps):
            self._storage.append(self._state.copy())
            self._state = self._run_step(i + 1, self._state.copy())

    def iterate_reverse(self, final_state_perturbation):
        self._serialized_state_perturbation = final_state_perturbation.copy()
        for i in reversed(range(self._num_steps)):
            self._serialized_state_perturbation = self._run_step_jacvec_rev(
                i + 1, self._storage.pop(), self._serialized_state_perturbation.copy()
            )
