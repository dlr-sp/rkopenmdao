import numpy as np
from abc import ABC, abstractmethod
from typing import TypeVar


class DiscretizedODE(ABC):
    """"""

    CacheType = TypeVar("CacheType")

    @abstractmethod
    def compute_update(
        self,
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
        time: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """"""

    @abstractmethod
    def export_linearization(self) -> CacheType:
        """"""

    @abstractmethod
    def import_linearization(self, cache: CacheType) -> None:
        """"""

    @abstractmethod
    def compute_update_derivative(
        self,
        step_input_pert: np.ndarray,
        stage_input_pert: np.ndarray,
        independent_input_pert: np.ndarray,
        time_pert: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """"""

    @abstractmethod
    def compute_update_adjoint_derivative(
        self,
        stage_output_pert: np.ndarray,
        stage_update_pert: np.ndarray,
        independent_output_pert: np.ndarray,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """"""
