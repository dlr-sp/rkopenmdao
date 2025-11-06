from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE


class TimeDiscretizationStateInterface(ABC):
    """"""

    @abstractmethod
    def set(self, other: TimeDiscretizationStateInterface):
        """"""

    @abstractmethod
    def to_dict(self) -> dict:
        """"""

    @classmethod
    @abstractmethod
    def from_dict(cls, state_dict: dict):
        """"""


@dataclass
class TimeDiscretizationStartingValues:
    initial_time: float
    initial_values: np.ndarray
    independent_inputs: np.ndarray


@dataclass
class TimeDiscretizationFinalizationValues:
    final_time: float
    final_values: np.ndarray
    final_independent_outputs: np.ndarray


class TimeDiscretizationSchemeInterface(ABC):
    @abstractmethod
    def create_empty_discretization_state(
        self, ode: DiscretizedODE
    ) -> TimeDiscretizationStateInterface:
        """"""

    @abstractmethod
    def compute_step(
        self,
        ode: DiscretizedODE,
        time_discretization_state: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """"""

    @abstractmethod
    def compute_step_derivative(
        self,
        ode: DiscretizedODE,
        time_discretization_state: TimeDiscretizationStateInterface,
        time_discretization_state_perturbation: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """"""

    @abstractmethod
    def compute_step_adjoint_derivative(
        self,
        ode: DiscretizedODE,
        time_discretization_state: TimeDiscretizationStateInterface,
        time_discretization_state_perturbation: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """"""

    @abstractmethod
    def time_discretization_starting_scheme(
        self,
        ode: DiscretizedODE,
        time_discretization_starting_values: TimeDiscretizationStartingValues,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """"""

    @abstractmethod
    def time_discretization_starting_scheme_derivative(
        self,
        ode: DiscretizedODE,
        time_discretization_starting_values: TimeDiscretizationStartingValues,
        time_discretization_starting_value_perturbations: TimeDiscretizationStartingValues,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """"""

    @abstractmethod
    def time_discretization_starting_scheme_adjoint_derivative(
        self,
        ode: DiscretizedODE,
        time_discretization_starting_values: TimeDiscretizationStartingValues,
        initial_discretization_state_perturbations: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> TimeDiscretizationStartingValues:
        """"""

    @abstractmethod
    def time_discretization_finalization_scheme(
        self,
        ode: DiscretizedODE,
        final_time_discretization_state: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> TimeDiscretizationFinalizationValues:
        """"""

    @abstractmethod
    def time_discretization_finalization_scheme_derivative(
        self,
        ode: DiscretizedODE,
        final_time_discretization_state: TimeDiscretizationStateInterface,
        final_time_discretization_state_perturbation: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> TimeDiscretizationFinalizationValues:
        """"""

    @abstractmethod
    def time_discretization_finalization_scheme_adjoint_derivative(
        self,
        ode: DiscretizedODE,
        final_time_discretization_state: TimeDiscretizationStateInterface,
        time_discretization_finalization_values: TimeDiscretizationFinalizationValues,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """"""
