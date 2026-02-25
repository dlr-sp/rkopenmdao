"""Classes for the usage of the modernized interface of pyrevolve in the
RungeKuttaIntegrator."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass

import pyrevolve as pr
import numpy as np

from rkopenmdao.time_integration_state import TimeIntegrationState
from rkopenmdao.integration_control import IntegrationControl


class RungeKuttaCheckpoint(pr.Checkpoint):
    """
    Blueprint for one checkpoint.

    Parameters
    ----------
    _symbol: dict
        Storage for making checkpoints accessible to pyRevolve.
    _time_integration_state: TimeIntegrationState
        Time integration state consistent with `_symbol`.
    """

    _symbol: dict
    _time_integration_state: TimeIntegrationState

    def __init__(self, time_integration_state: TimeIntegrationState):
        self._time_integration_state = time_integration_state
        self._symbols = time_integration_state.to_dict()

    def get_data_location(self, timestep):
        """
        Gets location for data at given time step.

        Parameters
        ----------
        timestep: Any
            Unused argument required by the interface.

        Returns
        -------
        locations: list
            Memory locations where checkpoint data is saved.
        """
        # pylint: disable=unused-argument
        # Here the method of getting the data is independent of the time step, but the
        # interface requires the argument.
        locations = self._get_data_impl(self._symbols, location=True)
        return locations

    def get_data(self, timestep):
        """
        Gets data at given time step.

        Parameters
        ----------
        timestep: Any
            Unused argument required by the interface.

        Returns
        -------
        locations: list
            Data if currently loaded checkpoint.
        """
        data = self._get_data_impl(self._symbols, location=False)
        return data

    def _get_data_impl(self, state_part: dict | np.ndarray, location: bool) -> list:
        """
        Recursive implementation of `get_data` and `get_data_location`.

        Parameters
        ----------
        state_part: dict | np.ndarray
            Current sub-part which is traversed to get data (location).
        location: bool
            Flag indicating whether location of data (True) or copy of data (False)
            is requested.

        Returns
        -------
        result: list
            List containing data (locations) of already traversed part of dict.
        """
        result = []
        if isinstance(state_part, dict):
            for part in state_part.values():
                result += self._get_data_impl(part, location)
        elif isinstance(state_part, np.ndarray):
            if state_part.size > 0:
                result.append(state_part if location else state_part.copy())
        else:
            raise TypeError(
                f"Unexpected type {type(state_part)} in RungeKuttaCheckpoint"
            )
        return result

    @property
    def size(self):
        """
        The memory consumption of the data contained in this checkpoint.

        Returns
        -------
        size: int
            Memory size of checkpoint in Bytes.
        """
        return self._size_impl(self._symbols)

    def _size_impl(self, state_part: dict | np.ndarray) -> int:
        """
        Recursive implementation of the size property.

        Parameters
        ----------
        state_part: dict | np.ndarray
            Current sub-part which is traversed to get checkpoint size.

        Returns
        -------
        size: int
            Size of already traversed part of checkpoint.
        """
        size = 0
        if isinstance(state_part, dict):
            for part in state_part.values():
                size += self._size_impl(part)
        elif isinstance(state_part, np.ndarray):
            size += state_part.size
        else:
            raise TypeError(
                f"Unexpected type {type(state_part)} in RungeKuttaCheckpoint"
            )
        return size

    @property
    def dtype(self):
        """
        Data type used for single values of the checkpoint.

        Returns
        -------
        dtype: type
            Data type used by single values of the checkpoint.
        """
        return np.float64


@dataclass
class RungeKuttaForwardOperator(pr.Operator):
    """
    Forward operator of the RungeKuttaIntegrator (i.e. the normal time
    integration).

    Parameters
    ----------
    time_integration_state: TimeIntegrationState
        State on which internal calculations are performed.
    fwd_operation: Callable[[TimeIntegrationState], TimeIntegrationState]
        Function applying one single step of time integration.
    integration_control: IntegrationControl
        Object for sharing data between ODE, time discretization and time integration.
    """

    time_integration_state: TimeIntegrationState
    fwd_operation: Callable[[TimeIntegrationState], TimeIntegrationState]
    integration_control: IntegrationControl

    def apply(self, **kwargs):
        # PyRevolve only ever uses t_start and t_end as arguments, but the interface is
        # how it is.
        """
        Does forward (primal) integration from step `t_start` + 1 to `t_end` + 1.

        t_start: int
            First integrated time step shifted by one.
        t_end: int
            Last integration time step shifted by one.
        """
        self.integration_control.step = kwargs["t_start"] + 1
        self.integration_control.step_time = (
            self.time_integration_state.discretization_state.final_time[0]
        )
        t_end = kwargs["t_end"]

        while self.integration_control.termination_condition_status(t_end + 1):
            self.time_integration_state.set(
                self.fwd_operation(self.time_integration_state)
            )


@dataclass
class RungeKuttaReverseOperator(pr.Operator):
    """
    Backward operator of the Runge-Kutta-integrator (i.e. one reverse step).

    Parameters
    ----------
    time_integration_state: TimeIntegrationState
        State which contains the current linearization point.
    time_integration_state_perturbations: TimeIntegrationState
        Perturbation state on which internal calculations are performed.
    rev_operation: Callable[
        [TimeIntegrationState, TimeIntegrationState], TimeIntegrationState
    ]
        Function applying one single reverse step of time integration.
    integration_control: IntegrationControl
        Object for sharing data between ODE, time discretization and time integration.
    """

    time_integration_state: TimeIntegrationState
    time_integration_state_perturbations: TimeIntegrationState
    rev_operation: Callable[
        [TimeIntegrationState, TimeIntegrationState], TimeIntegrationState
    ]
    integration_control: IntegrationControl

    def apply(self, **kwargs):
        """
        Does reverse (linear) integration from step `t_end` + 2 to `t_start` + 2.

        t_start: int
            First integrated time step shifted by one.
        t_end: int
            Last integration time step shifted by one.
        """
        # PyRevolve only ever uses t_start and t_end as arguments, but the interface is
        # how it is.
        t_start = kwargs["t_start"]
        t_end = kwargs["t_end"]
        for step in reversed(range(t_start + 2, t_end + 2)):
            self.integration_control.step = step
            self.time_integration_state_perturbations.set(
                self.rev_operation(
                    self.time_integration_state,
                    self.time_integration_state_perturbations,
                )
            )
            self.integration_control.decrement_step()
