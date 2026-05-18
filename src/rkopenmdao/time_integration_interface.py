"""Abstract interface for time integration schemes with primal and derivative support."""

from __future__ import annotations
from abc import ABC, abstractmethod

from rkopenmdao.states import (
    FinalizationValues,
    StartingValues,
    TimeIntegrationState,
)


class TimeIntegrationInterface(ABC):
    """
    Abstract base class for time integration schemes.

    This interface defines the contract for time integration implementations that support:
    - Primal integration (forward in time)
    - Derivative integration (linearized forward in time)
    - Adjoint derivative integration (reverse in time)

    Implementations must provide methods for:
    - Creating empty integration states for primal and derivative computations
    - Integrating from initial states through the full time domain
    - Computing derivatives with respect to initial conditions and parameters
    - Computing adjoint (reverse-mode) derivatives for sensitivity analysis

    The interface supports both fixed-step and adaptive-step integration schemes
    through the IntegrationConfig and ErrorController parameters.

    Parameters
    ----------
    None
        This is an abstract base class; implementations are configured through
        their constructor parameters (see CheckpointedTimeIntegration for example).
    """

    @abstractmethod
    def create_empty_primal_integration_state(self) -> TimeIntegrationState:
        """
        Creates an empty integration state for primal (non-perturbed) integration.

        The returned state contains arrays sized for the full time integration
        process, but with zero-filled values that will be populated during
        integration.

        Returns
        -------
        TimeIntegrationState
            Empty state for primal integration with:
            - Zero-filled discretization_state matching ODE dimensions
            - Step size arrays sized for adaptive integration history
            - Error history arrays initialized for adaptive stepping
        """

    @abstractmethod
    def create_empty_derivative_integration_state(self) -> TimeIntegrationState:
        """
        Creates an empty integration state for derivative (perturbed) integration.

        This state is used for computing tangent-linear (first-order) derivatives
        of the solution with respect to initial conditions or parameters.

        Returns
        -------
        TimeIntegrationState
            Empty state for derivative integration with:
            - Zero-filled discretization_state matching ODE dimensions
            - Empty or zero step_size_suggestion (fixed step for derivative)
            - Empty step_size_history and error_history (derivative uses same steps as primal)

        Notes
        -----
        The derivative state has the same discretization dimensions as the primal state
        but may have different step size array sizes depending on implementation.
        """

    @abstractmethod
    def integrate(
        self, initial_state: TimeIntegrationState
    ) -> list[TimeIntegrationState]:
        """
        Performs primal (forward) time integration from initial state to final time.

        Integrates the ODE system forward in time using the configured time
        discretization scheme and error control strategy.

        Parameters
        ----------
        initial_state : TimeIntegrationState
            The initial state containing:
            - discretization_state: Initial values of state variables
            - step_size_suggestion: Initial step size suggestion
            - step_size_history: Initial step size history (often filled with same value)
            - error_history: Initial error history (often filled with tolerance)

        Returns
        -------
        list[TimeIntegrationState]
            List containing the final integration state after completing all time steps.
            The list contains a single element for most implementations.

        Notes
        -----
        The integration continues until the termination_criterion in time_integration_config
        indicates completion. This may be based on:
        - Fixed number of steps
        - Reaching a specific final time
        - Other convergence criteria
        """

    @abstractmethod
    def integrate_derivative(
        self,
        initial_state: TimeIntegrationState,
        initial_state_perturbation: TimeIntegrationState,
    ) -> tuple[list[TimeIntegrationState], list[TimeIntegrationState]]:
        """
        Performs derivative (tangent-linear) integration forward in time.

        Computes the first-order derivative of the solution with respect to initial
        conditions by integrating the linearized ODE system using forward-mode
        differentiation.

        Parameters
        ----------
        initial_state : TimeIntegrationState
            The primal integration state (linearization point) containing:
            - discretization_state: State variables at initial time
            - step_size_suggestion: Initial step size suggestion
            - step_size_history: Initial step size history (often filled with same value)
            - error_history: Initial error history (often filled with tolerance)

        initial_state_perturbation : TimeIntegrationState
            The initial perturbation to propagate through the derivative system.
            Contains:
            - discretization_state: Perturbation of initial conditions
            - Other fields may be empty or zero (derivative doesn't use step control)

        Returns
        -------
        tuple[list[TimeIntegrationState], list[TimeIntegrationState]]
            A tuple containing:
            - List with final primal state (same as integrate)
            - List with final perturbation state after integration
        """

    @abstractmethod
    def integrate_adjoint_derivative(
        self,
        initial_state: TimeIntegrationState,
        final_state_perturbations: list[TimeIntegrationState],
    ) -> TimeIntegrationState:
        """
        Performs adjoint (reverse-mode) derivative integration backward in time.

        Computes the first-order derivative of the solution with respect to initial
        conditions by integrating the linearized ODE system using
        reverse-mode differentiation.

        Parameters
        ----------
        initial_state : TimeIntegrationState
            The primal integration state (linearization point) containing:
            - discretization_state: State variables at initial time
            - step_size_suggestion: Initial step size suggestion
            - step_size_history: Initial step size history (often filled with same value)
            - error_history: Initial error history (often filled with tolerance)

        final_state_perturbations : list[TimeIntegrationState]
            List containing the final state perturbation (adjoint variable).
            Typically:
            - discretization_state: ∂J/∂x at final time
            - Other fields may be empty or zero

        Returns
        -------
        TimeIntegrationState
            The perturbation of the initial state after reverse integration.
            Contains:
            - discretization_state: Adjoint of the final state with respect to initial conditions
            - Empty or zero step_size_suggestion, step_size_history, error_history).
        """

    @abstractmethod
    def starting_scheme(self, starting_values: StartingValues) -> TimeIntegrationState:
        """
        Initializes the integration state from starting values.

        Creates the initial integration state that will be used as input to the
        integrate method. This handles the transition from problem-specific initial
        conditions to the time integration scheme's internal state representation.

        Parameters
        ----------
        starting_values : StartingValues
            Problem-specific initial conditions containing:
            - time: Initial time value
            - Values for each time integration quantity (state variables)

        Returns
        -------
        TimeIntegrationState
            Initialized integration state with:
            - discretization_state: Populated with starting_values
            - step_size_suggestion: Set to time_integration_config.initial_step_size
            - step_size_history: Filled with initial step size
            - error_history: Filled with tolerance values
        """

    @abstractmethod
    def starting_scheme_derivative(
        self,
        starting_values: StartingValues,
        starting_value_perturbations: StartingValues,
    ) -> TimeIntegrationState:
        """
        Computes the derivative of the initialization with respect to perturbed starting values.

        Computes how the initial integration state changes when starting values are
        perturbed. This is the Jacobian-vector product for the initialization step.

        Parameters
        ----------
        starting_values : StartingValues
            Unperturbed initial conditions (linearization point)
        starting_value_perturbations : StartingValues
            Perturbations to initial conditions (vector to multiply by Jacobian)

        Returns
        -------
        TimeIntegrationState
            Perturbed initial integration state with:
            - discretization_state: Derivative of initial state w.r.t. starting values
            - Empty or zero step_size_suggestion, step_size_history, error_history
        """

    @abstractmethod
    def starting_scheme_adjoint_derivative(
        self,
        starting_values: StartingValues,
        integration_state_perturbations: TimeIntegrationState,
    ) -> StartingValues:
        """
        Computes the adjoint (reverse-mode) derivative of the initialization.

        Computes how perturbations in the initial integration state affect the
        starting values. This is the vector-Jacobian product for the initialization step,
        used in reverse-mode differentiation.

        Parameters
        ----------
        starting_values : StartingValues
            Unperturbed initial conditions (linearization point)
        integration_state_perturbations : TimeIntegrationState
            Perturbation of the initial integration state (adjoint variable)

        Returns
        -------
        StartingValues
            Perturbation of starting values (gradient of initial integration state w.r.t. initial conditions)
        """

    @abstractmethod
    def finalization_scheme(
        self, integration_state: TimeIntegrationState
    ) -> FinalizationValues:
        """
        Finalizes the integration and extracts final values.

        Converts the final integration state to problem-specific final values.
        This handles the transition from time integration scheme's internal
        representation to problem-specific output format.

        Parameters
        ----------
        integration_state : TimeIntegrationState
            The final state after completing all time steps, containing:
            - discretization_state: Final values of state variables
            - step_size_history: Complete history of step sizes used
            - error_history: Complete history of error measures

        Returns
        -------
        FinalizationValues
            Problem-specific final values containing:
            - final_time: The final time reached
            - Values for each time integration quantity at final time
        """

    @abstractmethod
    def finalization_scheme_derivative(
        self,
        integration_state: TimeIntegrationState,
        integration_state_perturbations: TimeIntegrationState,
    ) -> FinalizationValues:
        """
        Computes the derivative of finalization with respect to perturbed final state.

        Computes how the final values change when the final integration state
        is perturbed. This is the Jacobian-vector product for the finalization step.

        Parameters
        ----------
        integration_state : TimeIntegrationState
            The final state (linearization point) after completing integration
        integration_state_perturbations : TimeIntegrationState
            Perturbation of the final state (vector to multiply by Jacobian)

        Returns
        -------
        FinalizationValues
            Perturbation of final values (Jacobian-vector product)

        See Also
        --------
        finalization_scheme : Computes unperturbed final values
        finalization_scheme_adjoint_derivative : Computes adjoint of finalization
        """

    @abstractmethod
    def finalization_scheme_adjoint_derivative(
        self,
        integration_state: TimeIntegrationState,
        finalization_value_perturbations: FinalizationValues,
    ) -> TimeIntegrationState:
        """
        Computes the adjoint (reverse-mode) derivative of finalization.

        Computes how perturbations in the final values affect the final integration
        state. This is the vector-Jacobian product for the finalization step,
        used in reverse-mode differentiation.

        Parameters
        ----------
        integration_state : TimeIntegrationState
            The final state (linearization point) after completing integration
        finalization_value_perturbations : FinalizationValues
            Perturbations of final values (adjoint variable)

        Returns
        -------
        TimeIntegrationState
            Perturbation of final integration state (vector-Jacobian product)
        """
