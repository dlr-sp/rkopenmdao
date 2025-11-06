# pylint: disable=missing-module-docstring
from dataclasses import dataclass

import numpy as np

from .checkpoint_interface import CheckpointInterface


@dataclass
class NoCheckpointer(CheckpointInterface):
    """Checkpointer that sets no checkpoints, for cases where we don't need the
    reverse mode."""

    def create_checkpointer(self):
        """Doesn't checkpoint, so does nothing"""

    def iterate_forward(self, initial_state):
        """Runs time integration from start to finish."""
        self.state.set(initial_state)
        while self.integration_control.termination_condition_status():
            print(self.integration_control.step_time)
            self.state.set(self.run_step_func(self.state))
        return self.state

    def iterate_reverse(self, final_state_perturbation):
        """Does nothing"""
        raise NotImplementedError(
            "NoCheckpointer is explicitly for cases where no reverse mode is used."
            "If you need reverse mode, use another checkpointing implementation."
        )
