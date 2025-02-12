from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TimeDiscretizationInterface(ABC):
    single_state_size: int
    independent_input_size: int
    independent_output_size: int

    @abstractmethod
    def compute_step(self, step_input_state: np.ndarray, step_output_state: np.ndarray):
        """"""

    @abstractmethod
    def step_residual(
        self,
        step_input_state: np.ndarray,
        step_output_state: np.ndarray,
        step_residual_state: np.ndarray,
    ):
        """"""

    @abstractmethod
    def compute_stage(
        self, step_input_state: np.ndarray, step_output_state: np.ndarray
    ):
        """"""

    @abstractmethod
    def stage_residual(
        self,
        step_input_state: np.ndarray,
        step_output_state: np.ndarray,
        step_residual_state: np.ndarray,
    ):
        """"""
