import numpy as np

from .checkpoint_interface import CheckpointInterface


class NoCheckpointer(CheckpointInterface):
    """Checkpointer that sets no checkpoints, for cases where we don't need the reverse mode."""

    def __init__(self):
        self._array_size = 0
        self._num_steps = 0
        self._run_step = None
        self._state = None

    def setup(self, **kwargs):
        self._array_size = kwargs["array_size"]
        self._num_steps = kwargs["num_steps"]
        self._run_step = kwargs["run_step_func"]
        self._state = np.zeros(self._array_size)

    def create_checkpointer(self):
        pass

    def iterate_forward(self, initial_state):
        self._state = initial_state.copy()
        for i in range(self._num_steps):
            self._state = self._run_step(i + 1, self._state.copy())

    def iterate_reverse(self, final_state_perturbation):
        raise NotImplementedError(
            "NoCheckpointer is explicitly for cases where no reverse mode is used. If you need"
            " reverse mode, use another checkpointing implementation."
        )
