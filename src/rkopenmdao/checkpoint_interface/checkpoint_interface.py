"""Interface for checkpointing implementations in RKOpenMDAO."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.time_integration_state import TimeIntegrationState


@dataclass
class CheckpointInterface(ABC):
    """
    Abstract interface for checkpointing implementations.

    This class defines the interface required for different checkpointing strategies to
    be implemented in a consistent manner. It provides an interface for creating
    checkpointers and iterating forward and reverse through time.

    Parameters
    ----------
    integration_control: IntegrationControl
        IntegrationControl object for sharing data between ODE time discretization and
        time integration.
    run_step_func: Callable[[TimeIntegrationState], TimeIntegrationState]
        Function for the computation of one step of the forward (primal) time
        integration. Input is the state of the time integration at the start of the
        step, return value the state at the end of the same step.
    run_step_jacvec_rev_func: Callable[
        [TimeIntegrationState, TimeIntegrationState], TimeIntegrationState
    ]
        Function for the computation of one step of the reverse (linear) time
        integration. Inputs are the state of the time integration during the step
        acting as linearization point, as well as the perturbations for the end of the
        time step. Return value is the perturbation for the start of the time step.
    state: TimeIntegrationState
        Time integration state on which all computations for the forward (primal) time
        integration are performed.
    state_perturbation: TimeIntegrationState
        Time integration state on which all computations for the reverse (linear) time
        integration are performed.
    """

    integration_control: IntegrationControl
    run_step_func: Callable[[TimeIntegrationState], TimeIntegrationState]
    run_step_jacvec_rev_func: Callable[
        [TimeIntegrationState, TimeIntegrationState], TimeIntegrationState
    ]
    state: TimeIntegrationState
    state_perturbation: TimeIntegrationState

    @abstractmethod
    def create_checkpointer(self):
        """
        Routine for creating an internal object that actually creates checkpoints, if
        necessary.

        This method should be implemented by subclasses to return a specific
        checkpointing implementation. The created checkpointer will handle the actual
        storage and retrieval of checkpoints during time integration. Must be
        implemented by a subclass.
        """

    @abstractmethod
    def iterate_forward(
        self, initial_state: TimeIntegrationState
    ) -> TimeIntegrationState:
        """
        Routine for the forward iteration from start to finish.

        This method defines the interface for the forward (primal) time integration.

        Parameters
        ----------
        initial_state: TimeIntegrationState
            The state at the beginning of the time integration process.

        Returns
        -------
        final_state: TimeIntegrationState
            The resulting state after completing all time steps.
        """

    @abstractmethod
    def iterate_reverse(
        self, final_state_perturbation: TimeIntegrationState
    ) -> TimeIntegrationState:
        """
        Routine for the reverse iteration, from finish to start.

        This method defines the interface for the reverse (linear) time integration.

        Parameters
        ----------
        final_state_perturbation: TimeIntegrationState
            The perturbation of the final state used as starting point for reverse
            iteration.

        Returns
        -------
        initial_state_perturbation: TimeIntegrationState
            The resulting perturbation of the initial state after completing all time
            steps.
        """
