"""Definition for one state of time integration."""

from __future__ import annotations
from abc import ABC, abstractmethod

from rkopenmdao.states import (
    FinalizationValues,
    StartingValues,
    TimeIntegrationState,
)


class TimeIntegrationInterface(ABC):
    """
    TODO
    """

    @abstractmethod
    def create_empty_primal_integration_state(self) -> TimeIntegrationState:
        """
        TODO
        """

    @abstractmethod
    def create_empty_derivative_integration_state(self) -> TimeIntegrationState:
        """
        TODO
        """

    @abstractmethod
    def integrate(
        self, initial_state: TimeIntegrationState
    ) -> list[TimeIntegrationState]:
        """
        TODO
        """

    @abstractmethod
    def integrate_derivative(
        self,
        initial_state: TimeIntegrationState,
        initial_state_perturbation: TimeIntegrationState,
    ) -> tuple[list[TimeIntegrationState], list[TimeIntegrationState]]:
        """
        TODO
        """

    @abstractmethod
    def integrate_adjoint_derivative(
        self,
        initial_state: TimeIntegrationState,
        final_state_perturbations: list[TimeIntegrationState],
    ) -> TimeIntegrationState:
        """
        TODO
        """

    @abstractmethod
    def starting_scheme(self, starting_values: StartingValues) -> TimeIntegrationState:
        """"""

    @abstractmethod
    def starting_scheme_derivative(
        self,
        starting_values: StartingValues,
        starting_value_perturbations: StartingValues,
    ) -> TimeIntegrationState:
        """"""

    @abstractmethod
    def starting_scheme_adjoint_derivative(
        self,
        starting_values: StartingValues,
        integration_state_perturbations: TimeIntegrationState,
    ) -> StartingValues:
        """"""

    @abstractmethod
    def finalization_scheme(
        self, integration_state: TimeIntegrationState
    ) -> FinalizationValues:
        """"""

    @abstractmethod
    def finalization_scheme_derivative(
        self,
        integration_state: TimeIntegrationState,
        integration_state_perturbations: TimeIntegrationState,
    ) -> FinalizationValues:
        """"""

    @abstractmethod
    def finalization_scheme_adjoint_derivative(
        self,
        integration_state: TimeIntegrationState,
        finalization_value_perturbations: FinalizationValues,
    ) -> TimeIntegrationState:
        """"""
