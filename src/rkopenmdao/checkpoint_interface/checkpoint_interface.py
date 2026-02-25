# pylint: disable=missing-module-docstring

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple
from typing import Callable

import numpy as np
from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.time_integration_state import TimeIntegrationState


@dataclass
class CheckpointInterface(ABC):
    """Abstract interface for checkpointing implementations."""

    integration_control: IntegrationControl
    run_step_func: Callable[[TimeIntegrationState], TimeIntegrationState]
    run_step_jacvec_rev_func: Callable[
        [TimeIntegrationState, TimeIntegrationState], TimeIntegrationState
    ]
    state: TimeIntegrationState
    state_perturbation: TimeIntegrationState

    @abstractmethod
    def create_checkpointer(self):
        """Routine for creating an object that actually creates checkpoints, if
        necessary."""

    @abstractmethod
    def iterate_forward(
        self, initial_state: TimeIntegrationState
    ) -> TimeIntegrationState:
        """Routine for the forward iteration from start to finish."""

    @abstractmethod
    def iterate_reverse(
        self, final_state_perturbation: TimeIntegrationState
    ) -> TimeIntegrationState:
        """Routine for the reverse iteration, from finish to start."""
