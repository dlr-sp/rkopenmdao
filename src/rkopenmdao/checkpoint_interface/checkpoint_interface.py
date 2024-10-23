# pylint: disable=missing-module-docstring

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, List
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from rkopenmdao.integration_control import IntegrationControl


@dataclass
class CheckpointInterface(ABC):
    """Abstract interface for checkpointing implementations."""

    array_size: int
    integration_control: IntegrationControl
    run_step_func: Callable[[np.ndarray], Tuple[np.ndarray, List, List]]
    run_step_jacvec_rev_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    _state: np.ndarray = field(init=False)
    _serialized_state_perturbation: np.ndarray = field(init=False)

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
