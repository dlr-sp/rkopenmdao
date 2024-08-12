# pylint: disable=missing-module-docstring
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CheckpointInterface(ABC):
    """Abstract interface for checkpointing implementations."""

    array_size: int
    num_steps: int
    run_step_func: Callable[[int, np.ndarray], np.ndarray]
    run_step_jacvec_rev_func: Callable[[int, np.ndarray, np.ndarray], np.ndarray]

    @abstractmethod
    def create_checkpointer(self):
        """Routine for creating an object that actually creates checkpoints, if
        necessary."""

    @abstractmethod
    def iterate_forward(self, initial_state):
        """Routine for the forward iteration from start to finish."""

    @abstractmethod
    def iterate_reverse(self, final_state_perturbation):
        """Routine for the reverse iteration, from finish to start."""
