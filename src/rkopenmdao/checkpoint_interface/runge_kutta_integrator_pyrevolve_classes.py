"""Classes for the usage of the modernized interface of pyrevolve in the
RungeKuttaIntegrator."""

from __future__ import annotations
from collections.abc import Mapping, Callable

import pyrevolve as pr
import numpy as np


# More or less the same as the one from use_modernised.py in the pyrevolve examples
class RungeKuttaIntegratorSymbol:
    """One atomic part of the checkpointed data."""

    def __init__(self, data_dim):
        self._storage = np.zeros(data_dim + 1)
        self._data_dim = data_dim

    @property
    def data(self):
        """Property for the internal storage vector."""
        return self._storage

    @data.setter
    def data(self, data):
        """Setter for the internal storage vector."""
        if isinstance(data, np.ndarray):
            self._storage = data
        else:
            raise TypeError("Symbol data must be a numpy array.")

    @property
    def size(self):
        """Gets the size of the internal storage vector."""
        return self._storage.size


class RungeKuttaCheckpoint(pr.Checkpoint):
    """Blueprint for one checkpoint."""

    def __init__(self, symbols: dict):
        if isinstance(symbols, Mapping):
            self.symbols: dict = symbols
        else:
            raise TypeError(
                "Symbols must be a Mapping, for example a \
                              dictionary."
            )

    def get_data_location(self, timestep):
        """Gets location for data at given time step."""
        # pylint: disable=unused-argument
        # Here the method of getting the data is independent of the time step, but the
        # interface requires the argument.
        return [x.data for x in list(self.symbols.values())]

    def get_data(self, timestep):
        return [x.data for x in self.symbols.values()]

    @property
    def size(self):
        """The memory consumption of the data contained in this checkpoint."""
        size = 0
        for i in self.symbols:
            size = size + self.symbols[i].size
        return size

    @property
    def dtype(self):
        return np.float64


class RungeKuttaForwardOperator(pr.Operator):
    """Forward operator of the RungeKuttaIntegrator (i.e. the normal time
    integration)."""

    # pylint: disable=too-few-public-methods
    # The interface is controlled by PyRevolve, and more methods are not needed.

    serialized_old_state_symbol: RungeKuttaIntegratorSymbol
    serialized_new_state_symbol: RungeKuttaIntegratorSymbol
    fwd_operation: Callable[[int, np.ndarray], np.ndarray]

    def __init__(
        self,
        serialized_old_state_symbol: RungeKuttaIntegratorSymbol,
        serialized_new_state_symbol: RungeKuttaIntegratorSymbol,
        fwd_operation: Callable[
            [int, np.ndarray],
            np.ndarray,
        ],
        integration_control,
    ):
        self.serialized_old_state_symbol = serialized_old_state_symbol
        self.serialized_new_state_symbol = serialized_new_state_symbol
        self.fwd_operation = fwd_operation
        self.integration_control = integration_control

    def apply(self, **kwargs):
        # PyRevolve only ever uses t_start and t_end as arguments, but the interface is
        # how it is.
        self.integration_control.step = kwargs["t_start"]
        t_end = kwargs["t_end"]

        self.integration_control.step_time = self.serialized_new_state_symbol.data[-1]

        while self.integration_control.termination_condition_status(t_end):
            self.serialized_old_state_symbol.data = (
                self.serialized_new_state_symbol.data.copy()
            )
            self.serialized_new_state_symbol.data[:-1] = self.fwd_operation(
                self.serialized_old_state_symbol.data[:-1],
            )[0]
            self.serialized_new_state_symbol.data[-1] = (
                self.integration_control.step_time
            )


class RungeKuttaReverseOperator(pr.Operator):
    """Backward operator of the Runge-Kutta-integrator (i.e. one reverse step)."""

    # pylint: disable=too-few-public-methods
    # The interface is controlled by PyRevolve, and more methods are not needed.

    serialized_old_state_symbol: RungeKuttaIntegratorSymbol
    serialized_state_perturbations: np.ndarray
    rev_operation: Callable[[int, np.ndarray, np.ndarray], np.ndarray]

    def __init__(
        self,
        serialized_old_state_symbol: RungeKuttaIntegratorSymbol,
        state_size: int,
        rev_operation: Callable[
            [int, np.ndarray, np.ndarray],
            np.ndarray,
        ],
        integration_control,
    ):
        self.serialized_old_state_symbol = serialized_old_state_symbol
        self.serialized_state_perturbations = np.zeros(state_size)
        self.rev_operation = rev_operation
        self.integration_control = integration_control

    def apply(self, **kwargs):
        # PyRevolve only ever uses t_start and t_end as arguments, but the interface is
        # how it is.
        t_start = kwargs["t_start"]
        t_end = kwargs["t_end"]
        for step in reversed(range(t_start + 1, t_end + 1)):
            self.integration_control.step = step
            self.integration_control.step_time = self.serialized_old_state_symbol.data[
                -1
            ]
            self.serialized_state_perturbations = self.rev_operation(
                self.serialized_old_state_symbol.data[:-1],
                self.serialized_state_perturbations,
            )
