# pylint: disable=missing-module-docstring
from dataclasses import dataclass

import numpy as np

from .checkpoint_interface import CheckpointInterface


@dataclass
class NoCheckpointer(CheckpointInterface):
    """Checkpointer that sets no checkpoints, for cases where we don't need the
    reverse mode."""

    def __post_init__(self):
        """Reserves memory for time integration state."""
        self._state = np.zeros(self.array_size)

    def create_checkpointer(self):
        """Doesn't checkpoint, so does nothing"""

    def iterate_forward(self, initial_state):
        """Runs time intgration from start to finish."""
        self._state = initial_state
        if self.integration_control.termination_criterion.criterion == 'num_steps':
            while self.integration_control.step != self.integration_control.termination_criterion.value:
                self._state = self.run_step_func(self._state.copy())
                self.integration_control.increment_step()
        elif self.integration_control.termination_criterion.criterion == 'end_time':
            while (np.abs(self.integration_control.remaining_time())
                   >= min(1e-13, self.integration_control.smallest_delta_t)):
                self._state = self.run_step_func(self._state.copy())
                self.integration_control.increment_step()


    def iterate_reverse(self, final_state_perturbation):
        """Does nothing"""
        raise NotImplementedError(
            "NoCheckpointer is explicitly for cases where no reverse mode is used."
            "If you need reverse mode, use another checkpointing implementation."
        )
