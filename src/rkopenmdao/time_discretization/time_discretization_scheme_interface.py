"""Interface definition for time discretization schemes and adjacent classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rkopenmdao.states import (
    TimeDiscretizationStateInterface,
    StartingValues,
    FinalizationValues,
    DiscretizedODEResultState,
)
from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE


class TimeDiscretizationSchemeInterface(ABC):
    """
    General interface for time discretization in RKOpenMDAO. Models one basic step of
    time integration and its differentiated version, as well as creating a valid
    discretization state from usual data given by an initial value problem, and vice
    versa.
    """

    @abstractmethod
    def create_empty_discretization_state(
        self, ode: DiscretizedODE
    ) -> TimeDiscretizationStateInterface:
        """
        Creates a valid discretization state using sizes given by the `ode`.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE for which the discretization state is valid.

        Returns
        -------
        time_discretization_state: TimeDiscretizationStateInterface
            Empty initialized discretization state.
        """

    @abstractmethod
    def compute_step(
        self,
        ode: DiscretizedODE,
        time_discretization_state: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """
        Computes one (primal) step of time integration on the `ode` with `step_size`
        based on `time_discretization_state`.

        Note that this happens in place, meaning the argument
        `time_discretization_state` contains the new data after the call and is
        returned additionally for convenience.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE implementation of which the step is computed.
        time_discretization_state: TimeDiscretizationStateInterface
            Discretization state on which computations take place.
        step_size: float
            Step size used for the computation of the time step.

        Returns
        -------
        time_discretization_state: TimeDiscretizationStateInterface
        """

    @abstractmethod
    def compute_step_derivative(
        self,
        ode: DiscretizedODE,
        time_discretization_state: TimeDiscretizationStateInterface,
        time_discretization_state_perturbation: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """
        Forward-mode differentiated version of `compute_step`. Uses
        `time_discretization_state` as linearization point, calculating the
        jacvec product of `time_discretization_state_perturbation` valid to `ode` with
        `step_size`.

        Note that this happens in place, meaning the
        argument `time_discretization_state_perturbation` contains the new data after
        the call and is returned additionally for convenience.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE implementation of which the jacvec product is computed.
        time_discretization_state: TimeDiscretizationStateInterface
            Primal discretization state acting as linearization point.
        time_discretization_state_perturbation: TimeDiscretizationStateInterface
            (Linear) discretization state on which computations take place.
        step_size: float
            Step size used for the computation of the time step.

        Returns
        -------
        time_discretization_state_perturbation: TimeDiscretizationStateInterface
        """

    @abstractmethod
    def compute_step_adjoint_derivative(
        self,
        ode: DiscretizedODE,
        time_discretization_state: TimeDiscretizationStateInterface,
        time_discretization_state_perturbation: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """
        Reverse-mode differentiated version of `compute_step`. Uses
        `time_discretization_state` as linearization point, calculating the
        adjoint jacvec product of `time_discretization_state_perturbation` valid to
        `ode` with `step_size`.

        Note that this happens in place, meaning the
        argument `time_discretization_state_perturbation` contains the new data after
        the call and is returned additionally for convenience.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE implementation of which the adjoint jacvec product is computed.
        time_discretization_state: TimeDiscretizationStateInterface
            Primal discretization state acting as linearization point.
        time_discretization_state_perturbation: TimeDiscretizationStateInterface
            (Linear) discretization state on which computations take place.
        step_size: float
            Step size used for the computation of the time step.

        Returns
        -------
        time_discretization_state_perturbation: TimeDiscretizationStateInterface
        """

    @abstractmethod
    def time_discretization_starting_scheme(
        self,
        ode: DiscretizedODE,
        starting_values: StartingValues,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """
        Starting scheme of the time discretization for converting usual data
        representation of state of ODEs to one compatible to the used time
        discretization.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        starting_values: StartingValues
            Values on which the starting scheme is performed.
        step_size: float
            Step size for the starting scheme.

        Returns
        -------
        started_discretization_state: TimeDiscretizationStateInterface
            Converted version of `starting_values` compatible with the used time
            discretization.
        """

    @abstractmethod
    def time_discretization_starting_scheme_derivative(
        self,
        ode: DiscretizedODE,
        starting_values: StartingValues,
        starting_value_perturbations: StartingValues,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """
        Forward-mode differentiated version of starting scheme.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        starting_values: StartingValues
            Linearization point for the jacobian of the starting scheme.
        starting_value_perturbations: StartingValues
            Perturbations to be multiplied with the jacobian of the starting scheme.
        step_size: float
            Step size for the starting scheme.

        Returns
        -------
        started_discretization_state_perturbations: TimeDiscretizationStateInterface
            Result of the jacvec-product of the starting scheme.
        """

    @abstractmethod
    def time_discretization_starting_scheme_adjoint_derivative(
        self,
        ode: DiscretizedODE,
        starting_values: StartingValues,
        started_discretization_state_perturbations: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> StartingValues:
        """
        Reverse-mode differentiated version of starting scheme.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        starting_values: StartingValues
            Linearization point for the adjoint jacobian of the starting scheme.
        started_discretization_state_perturbations: TimeDiscretizationStateInterface
            Perturbations to be multiplied with the adjoint jacobian of the starting
            scheme.
        step_size: float
            Step size for the starting scheme.

        Returns
        -------
        starting_value_perturbations: StartingValues
            Result of the adjoint jacvec-product of the starting scheme.
        """

    @abstractmethod
    def time_discretization_finalization_scheme(
        self,
        ode: DiscretizedODE,
        discretization_state: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> FinalizationValues:
        """
        Finalization scheme of the time discretization for converting a state specific
        to the discretization back to one compatible with an ODE.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        discretization_state: TimeDiscretizationStateInterface
            Values on which the finalization scheme is performed.
        step_size: float
            Step size for the finalization scheme.

        Returns
        -------
        finalization_values: FinalizationValues
            Converted version of `final_time_discretization_state` compatible with the
            used ODE.
        """

    @abstractmethod
    def time_discretization_finalization_scheme_derivative(
        self,
        ode: DiscretizedODE,
        discretization_state: TimeDiscretizationStateInterface,
        discretization_state_perturbations: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> FinalizationValues:
        """
        Forward-mode differentiated version of finalization scheme.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        discretization_state: TimeDiscretizationStateInterface
            Linearization point for the jacobian of the finalization scheme.
        discretization_state_perturbations: TimeDiscretizationStateInterface
            Perturbations to be multiplied with the jacobian of the finalization
            scheme.
        step_size: float
            Step size for the finalization scheme.

        Returns
        -------
        finalization_value_perturbations: FinalizationValues
            Result of the jacvec-product of the finalization scheme.
        """

    @abstractmethod
    def time_discretization_finalization_scheme_adjoint_derivative(
        self,
        ode: DiscretizedODE,
        discretization_state: TimeDiscretizationStateInterface,
        finalization_value_perturbations: FinalizationValues,
        step_size: float,
    ) -> TimeDiscretizationStateInterface:
        """
        Reverse-mode differentiated version of finalization scheme.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        discretization_state: TimeDiscretizationStateInterface
            Linearization point for the adjoint jacobian of the finalization scheme.
        finalization_value_perturbations: FinalizationValues
            Perturbations to be multiplied with the adjoint jacobian of the
            finalization scheme.
        step_size: float
            Step size for the finalization scheme.

        Returns
        -------
        discretization_state_perturbations: TimeDiscretizationStateInterface
            Result of the adjoint jacvec-product of the finalization scheme.
        """

    @abstractmethod
    def get_ode_state(
        self,
        ode: DiscretizedODE,
        discretization_state: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> DiscretizedODEResultState:
        """
        Uses the contents of `discretization_state` to create a valid result state for
        the passed ode containing state data.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        discretization_state: TimeDiscretizationStateInterface
            Discretization state from which data is used.
        step_size: float
            Step size that is possibly necessary for some calculations.

        Returns
        -------
        ode_state: DiscretizedODEResultState
            ODE-compatible result state containing discretization state data.
        """

    @abstractmethod
    def get_ode_error_estimate(
        self,
        ode: DiscretizedODE,
        discretization_state: TimeDiscretizationStateInterface,
        step_size: float,
    ) -> DiscretizedODEResultState | None:
        """
        Uses the contents of `discretization_state` to create a valid result state for
        the passed ode containing error estimate data.

        Parameters
        ----------
        ode: DiscretizedODE
            ODE on which time integration is performed.
        discretization_state: TimeDiscretizationStateInterface
            Discretization state from which data is used.
        step_size: float
            Step size that is possibly necessary for some calculations.

        Returns
        -------
        ode_error_estimate: DiscretizedODEResultState
            ODE-compatible result state containing error estimate data.
        """
