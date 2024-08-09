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
        pass

    @abstractmethod
    def iterate_forward(self, initial_state):
        pass

    @abstractmethod
    def iterate_reverse(self, final_state_perturbation):
        pass
