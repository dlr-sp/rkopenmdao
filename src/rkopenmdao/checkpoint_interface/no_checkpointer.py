from dataclasses import dataclass

import numpy as np

from .checkpoint_interface import CheckpointInterface


@dataclass
class NoCheckpointer(CheckpointInterface):
    """Checkpointer that sets no checkpoints, for cases where we don't need the reverse mode."""

    def __post_init__(self):
        self._state = np.zeros(self.array_size)

    def create_checkpointer(self):
        pass

    def iterate_forward(self, initial_state):
        self._state = initial_state
        for i in range(self.num_steps):
            self._state = self.run_step_func(i + 1, self._state.copy())

    def iterate_reverse(self, final_state_perturbation):
        raise NotImplementedError(
            "NoCheckpointer is explicitly for cases where no reverse mode is used. If you need"
            " reverse mode, use another checkpointing implementation."
        )
