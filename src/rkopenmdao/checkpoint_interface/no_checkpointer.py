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
        """Runs time integration from start to finish."""
        self._state = initial_state
        while self.integration_control.termination_condition_status():
            self._state = self.run_step_func(self._state.copy())[0]
            if (
                self.integration_control.delta_t
                / self.integration_control.delta_t_suggestion
                < 1e-10
            ):
                raise ValueError(
                    "Oscilliations Error: Oscilliation on forward iteration "
                    f"at step <{self.integration_control.step}>: "
                    f"dt = {self.integration_control.delta_t} and "
                    f"dt_suggestion = {self.integration_control.delta_t_suggestion}"
                )

    def iterate_reverse(self, final_state_perturbation):
        """Does nothing"""
        raise NotImplementedError(
            "NoCheckpointer is explicitly for cases where no reverse mode is used."
            "If you need reverse mode, use another checkpointing implementation."
        )
