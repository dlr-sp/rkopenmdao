from collections.abc import Mapping
from typing import Callable, Tuple

import pyrevolve as pr
import numpy as np


class RungeKuttaIntegratorSymbol:
    # basically the same as the one from use_modernised.py in the pyrevolve examples
    def __init__(self, data_dim):
        self._storage = np.zeros(data_dim)
        self._data_dim = data_dim

    @property
    def data(self):
        return self._storage

    @data.setter
    def data(self, value):
        if isinstance(value, np.ndarray):
            self._storage = value
        else:
            raise Exception("Symbol data must be a numpy array.")

    @property
    def size(self):
        return self._storage.size


class RungeKuttaCheckpoint(pr.Checkpoint):
    def __init__(self, symbols: dict):
        if isinstance(symbols, Mapping):
            self.symbols: dict = symbols
        else:
            raise Exception(
                "Symbols must be a Mapping, for example a \
                              dictionary."
            )

    def get_data_location(self, timestep):
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
    def __init__(
        self,
        state_size: int,
        functional_size: int,
        stage_number: int,
        fwd_operation: Callable[
            [int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
        ],
    ):
        self.serialized_state_symbol: RungeKuttaIntegratorSymbol = RungeKuttaIntegratorSymbol(
            state_size
        )
        self.functional_part: np.ndarray = np.zeros(functional_size)
        self.stage_cache: np.ndarray = np.zeros((stage_number, state_size))
        self.accumulated_stages: np.ndarray = np.zeros(state_size)
        self.fwd_operation: Callable[
            [int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
        ] = fwd_operation

    def apply(self, t_start: int, t_end: int):
        for step in range(t_start + 1, t_end + 1):
            (
                self.serialized_state_symbol.data,
                self.functional_part,
            ) = self.fwd_operation(
                step,
                self.serialized_state_symbol.data,
                self.functional_part,
                self.stage_cache,
                self.accumulated_stages,
            )


# TODO: the type hints are probably not necessary in both the signature and the init_body
class RungeKuttaReverseOperator(pr.Operator):
    def __init__(
        self,
        state_size: int,
        stage_number: int,
        rev_operation: Callable[
            [
                int,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
            Tuple[np.ndarray, np.ndarray],
        ],
    ):
        self.serialized_state_symbol: RungeKuttaIntegratorSymbol = RungeKuttaIntegratorSymbol(
            state_size
        )
        self.stage_cache: np.ndarray = np.zeros((stage_number, state_size))
        self.accumulated_stages: np.ndarray = np.zeros(state_size)
        self.serialized_state_perturbations: np.ndarray = np.zeros(state_size)
        self.stage_perturbations_cache: np.ndarray = np.zeros((stage_number, state_size))
        self.functional_perturbations: np.ndarray = np.zeros(state_size)
        self.functional_stage_perturbations_cache: np.ndarray = np.zeros((stage_number, state_size))
        self.original_functional_perturbations: np.zeros(state_size)
        self.rev_operation: Callable[
            [
                int,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
            Tuple[np.ndarray, np.ndarray],
        ] = rev_operation

    def apply(self, t_start: int, t_end: int):
        for step in range(t_end, t_start, -1):
            self.serialized_state_perturbations, self.functional_perturbations = self.rev_operation(
                step,
                self.serialized_state_symbol.data,
                self.stage_cache,
                self.accumulated_stages,
                self.serialized_state_perturbations,
                self.stage_perturbations_cache,
                self.functional_perturbations,
                self.functional_stage_perturbations_cache,
                self.original_functional_perturbations,
            )
