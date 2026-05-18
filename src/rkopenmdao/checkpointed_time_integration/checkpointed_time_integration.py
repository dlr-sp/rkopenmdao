"""Base class for checkpointed time integration implementations."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from rkopenmdao.callback import Callback
from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE
from rkopenmdao.error_controller import ErrorController
from rkopenmdao.error_measurer import ErrorMeasurer
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationSchemeInterface,
)
from rkopenmdao.states import TimeIntegrationState, StartingValues, FinalizationValues
from rkopenmdao.time_integration_interface import (
    TimeIntegrationInterface,
)


@dataclass
class CheckpointedTimeIntegration(TimeIntegrationInterface):
    """
    Base class for checkpointed time integration implementations.

    This abstract base class provides a partial implementation of the time integration
    interface with checkpointing support. It handles the forward integration loop,
    error control, and callbacks, while delegating checkpoint-specific operations
    to subclasses.

    Checkpointing strategies determine how intermediate states are stored and retrieved
    during reverse-mode (adjoint) integration. The base class provides common operations
    for:
    - Forward integration of single time steps with adaptive stepping
    - Derivative integration if single time steps following the same step size
    - Derivative integration for the whole time domain
    - Adjoint derivative integration of single time steps following the adaptive step size

    Subclasses must implement checkpoint-specific integration methods:
    - integrate(): Forward integration of the whole time domain with checkpointing
    - integrate_adjoint_derivative(): Reverse adjoint integration of the whole time
      domain with checkpointing

    Parameters
    ----------
    ode : DiscretizedODE
        The discretized ordinary differential equation system to integrate.
        Contains the problem definition, including methods for computing solution,
        error estimates, and derivatives

    time_discretization_scheme : TimeDiscretizationSchemeInterface
        The time discretization scheme to use.
        Provides methods for:
        - Advancing the solution by one step
        - Computing error estimates
        - Computing derivatives (forward and adjoint)

    error_controller : ErrorController
        The adaptive step size controller. Determines whether a step is accepted
        and suggests new step sizes based on error estimates.

    error_measurer : ErrorMeasurer
        Computes scalar error measures from error estimates and current state.
        Provides the norm or measure used by the error_controller to decide step
        acceptance.

    time_integration_config : IntegrationConfig
        Configuration for the integration process, including:
        - Termination criterion (fixed steps, final time, etc.)
        - Initial step size
        - Usage of adaptive time stepping

    integrate_callbacks : list[Callback]
        List of callback objects invoked during primal integration, called at:
        - before_iteration(): Before each integration step
        - after_iteration(): After each integration step

    integrate_derivative_callbacks : list[Callback]
        List of callback objects invoked during derivative integration, called at:
        - before_iteration(): Before each integration step
        - after_iteration(): After each integration step

    integrate_adjoint_derivative_callbacks : list[Callback]
        List of callback objects invoked during adjoint derivative integration,
        called at:
        - before_iteration(): Before each integration step
        - after_iteration(): After each integration step

    Notes
    -----
    This class provides default implementations for derivative integration that
    simply follow the same time steps as the primal integration. The checkpointing
    subclasses must override integrate() and integrate_adjoint_derivative() to
    implement their specific checkpointing strategies.

    The callback system allows monitoring of the integration process.
    Callbacks can inspect or modify the integration state, access error measures,
    or trigger external actions like visualization or data writing.
    """

    ode: DiscretizedODE
    time_discretization_scheme: TimeDiscretizationSchemeInterface

    error_controller: ErrorController
    error_measurer: ErrorMeasurer

    time_integration_config: IntegrationConfig

    integrate_callbacks: list[Callback]
    integrate_derivative_callbacks: list[Callback]
    integrate_adjoint_derivative_callbacks: list[Callback]

    def create_empty_primal_integration_state(self) -> TimeIntegrationState:
        """
        Creates an empty state for primal integration.

        The state is sized for the full time integration process with arrays for
        step size and error history.

        Returns
        -------
        TimeIntegrationState
            Empty primal state with:
            - Zero-filled discretization_state matching ODE dimensions
            - step_size_history and error_history arrays of size 2
            - step_size_suggestion array of size 1
        """
        return TimeIntegrationState(
            self.time_discretization_scheme.create_empty_discretization_state(self.ode),
            np.zeros(1),
            np.zeros(2),
            np.zeros(2),
        )

    def create_empty_derivative_integration_state(self) -> TimeIntegrationState:
        """
        Creates an empty state for derivative integration.

        The state is sized for the derivative computation, which may have different
        array sizes than the primal state.

        Returns
        -------
        TimeIntegrationState
            Empty derivative state with:
            - Zero-filled discretization_state matching ODE dimensions
            - Empty (size 0) step_size_suggestion, step_size_history, error_history
        """
        return TimeIntegrationState(
            self.time_discretization_scheme.create_empty_discretization_state(self.ode),
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
        )

    def integrate_derivative(
        self,
        initial_state: TimeIntegrationState,
        initial_state_perturbation: TimeIntegrationState,
    ) -> tuple[list[TimeIntegrationState], list[TimeIntegrationState]]:
        """
        Performs derivative integration following the same steps as primal integration.

        The derivative integration uses the exact same time steps as the primal
        integration (from initial_state.step_size_history), ensuring consistent
        linearization.

        Parameters
        ----------
        initial_state : TimeIntegrationState
            Primal integration state containing:
            - discretization_state: Linearization point (unperturbed initial values)
            - step_size_suggestion: Initial step size suggestion
            - step_size_history: Initial step size history (often filled with same value)
            - error_history: Initial error history (often filled with tolerance)

        initial_state_perturbation : TimeIntegrationState
            Initial perturbation state containing:
            - discretization_state: Perturbation of initial conditions

        Returns
        -------
        tuple[list[TimeIntegrationState], list[TimeIntegrationState]]
            Tuple of:
            - List with final primal state (same as integrate)
            - List containing the final perturbation state
        """
        iteration = 0
        while not self.time_integration_config.termination_criterion.is_iteration_finished(
            iteration, initial_state, self.ode, self.time_discretization_scheme
        ):
            iteration += 1
            self._run_step_derivative(
                iteration,
                initial_state,
                initial_state_perturbation,
            )

        return [initial_state], [initial_state_perturbation]

    def _run_step(
        self, iteration: int, time_integration_state: TimeIntegrationState
    ) -> None:
        """
        Executes one integration step with callback support.

        This method advances the integration state by one time step using the
        configured time discretization scheme and error controller. It invokes
        callbacks before and after the iteration for monitoring.

        Parameters
        ----------
        iteration : int
            The current iteration (time step) number, starting from 1.

        time_integration_state : TimeIntegrationState
            The state to advance by one step. Modified in-place:
            - discretization_state: Updated to new time level
            - step_size_suggestion: Updated with new step size suggestion
            - step_size_history: Updated with most recent step size
            - error_history: Updated with most recent error measure

        See Also
        --------
        _iterate_on_step : Core step advancement logic without callbacks
        """
        for callback in self.integrate_callbacks:
            callback.before_iteration(
                iteration,
                time_integration_state,
                self.ode,
                self.time_discretization_scheme,
            )
        self._iterate_on_step(time_integration_state)
        for callback in self.integrate_callbacks:
            callback.after_iteration(
                iteration,
                time_integration_state,
                self.ode,
                self.time_discretization_scheme,
            )

    def _run_step_derivative(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        time_integration_state_perturbations: TimeIntegrationState,
    ) -> None:
        """
        Executes one derivative integration step with callback support.

        Advances both the primal state and the perturbation state by one time step.
        The perturbation state is updated using the time discretization scheme's
        derivative computation method.

        Parameters
        ----------
        iteration : int
            The current iteration (time step) number.

        time_integration_state : TimeIntegrationState
            The primal state (linearization point). Modified in-place.

        time_integration_state_perturbations : TimeIntegrationState
            The perturbation state. Modified in-place with derivative values.
        """
        for callback in self.integrate_derivative_callbacks:
            callback.before_iteration(
                iteration,
                time_integration_state_perturbations,
                self.ode,
                self.time_discretization_scheme,
            )
        self._iterate_on_step(time_integration_state)
        time_integration_state_perturbations.discretization_state.set(
            self.time_discretization_scheme.compute_step_derivative(
                self.ode,
                time_integration_state.discretization_state,
                time_integration_state_perturbations.discretization_state,
                time_integration_state.step_size_history[0],
            )
        )
        for callback in self.integrate_derivative_callbacks:
            callback.after_iteration(
                iteration,
                time_integration_state_perturbations,
                self.ode,
                self.time_discretization_scheme,
            )

    def _run_step_adjoint_derivative(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        time_integration_state_perturbations: TimeIntegrationState,
    ) -> None:
        """
        Executes one adjoint derivative integration step with callback support.

        Advances the perturbation state backward in time using the adjoint of
        the time discretization scheme's Jacobian.

        Parameters
        ----------
        iteration : int
            The current iteration (time step) number, counting down during reverse.

        time_integration_state : TimeIntegrationState
            The primal state (linearization point) at this time level.

        time_integration_state_perturbations : TimeIntegrationState
            The perturbation state. Modified in-place with adjoint derivative values.
        """
        for callback in self.integrate_adjoint_derivative_callbacks:
            callback.before_iteration(
                iteration,
                time_integration_state_perturbations,
                self.ode,
                self.time_discretization_scheme,
            )
        time_integration_state_perturbations.discretization_state.set(
            self.time_discretization_scheme.compute_step_adjoint_derivative(
                self.ode,
                time_integration_state.discretization_state,
                time_integration_state_perturbations.discretization_state,
                time_integration_state.step_size_history[0],
            )
        )
        for callback in self.integrate_adjoint_derivative_callbacks:
            callback.after_iteration(
                iteration,
                time_integration_state_perturbations,
                self.ode,
                self.time_discretization_scheme,
            )

    def _iterate_on_step(
        self,
        time_integration_state: TimeIntegrationState,
    ) -> None:
        """
        Core logic for advancing the solution by one time step.

        This method implements the adaptive time stepping algorithm:
        1. Tentative step computation using the time discretization scheme
        2. Error estimation from the computed solution
        3. Error measurement (norm) computation
        4. Step acceptance decision using error_controller
        5. Step size adjustment for next step

        The method may iterate multiple times if the step is rejected, adjusting
        the step size suggestion each time until acceptance or a stall threshold.

        Parameters
        ----------
        time_integration_state : TimeIntegrationState
            The state to advance. Modified in-place:
            - discretization_state: Updated to new time level if step accepted
            - step_size_suggestion: Updated with new suggestion for next step
            - step_size_history: Updated with most recent accepted step size
            - error_history: Updated with most recent error measure
        """
        temp_discretization_state = deepcopy(
            time_integration_state.discretization_state
        )
        temp_discretization_state = self.time_discretization_scheme.compute_step(
            self.ode,
            temp_discretization_state,
            time_integration_state.step_size_suggestion[0],
        )
        ode_state = self.time_discretization_scheme.get_ode_state(
            self.ode,
            temp_discretization_state,
            time_integration_state.step_size_suggestion[0],
        )
        ode_error_estimate = self.time_discretization_scheme.get_ode_error_estimate(
            self.ode,
            temp_discretization_state,
            time_integration_state.step_size_suggestion[0],
        )
        if ode_error_estimate:
            error_measure = self.error_measurer.get_measure(
                ode_error_estimate, ode_state, self.ode
            )
        else:
            error_measure = 0.0

        stall_counter = 0
        while True:
            if hasattr(
                self.time_integration_config.termination_criterion,
                "remaining_time",
            ):
                remaining_time = self.time_integration_config.termination_criterion.remaining_time(
                    self.time_discretization_scheme.time_discretization_finalization_scheme(
                        self.ode,
                        temp_discretization_state,
                        time_integration_state.step_size_suggestion[0],
                    ).final_time
                )
            else:
                remaining_time = np.inf

            error_controller_status = self.error_controller(
                error_measure,
                time_integration_state.step_size_suggestion[0],
                remaining_time,
                time_integration_state.error_history,
                time_integration_state.step_size_history,
            )

            if error_controller_status.acceptance or stall_counter > 4:
                break
            stall_counter += 1
            time_integration_state.step_size_suggestion[0] = (
                error_controller_status.step_size_suggestion
            )

        if ode_error_estimate:
            new_step_size_history = np.roll(time_integration_state.step_size_history, 1)
            new_step_size_history[0] = time_integration_state.step_size_suggestion[0]
            new_error_history = np.roll(time_integration_state.error_history, 1)
            new_error_history[0] = error_measure
            time_integration_state.step_size_history[:] = new_step_size_history
            time_integration_state.error_history[:] = new_error_history
        time_integration_state.step_size_suggestion[0] = (
            error_controller_status.step_size_suggestion
        )
        time_integration_state.discretization_state.set(temp_discretization_state)

    def starting_scheme(self, starting_values: StartingValues) -> TimeIntegrationState:
        """
        Initializes the integration state from starting values.

        Creates the initial integration state that will be used as input to integrate.

        Parameters
        ----------
        starting_values : StartingValues
            Problem-specific initial conditions containing:
            - time: Initial time value
            - Values for each time integration quantity (state variables)

        Returns
        -------
        TimeIntegrationState
            Initialized state with:
            - discretization_state: Populated from starting_values
            - step_size_suggestion: Set to initial_step_size from config
            - step_size_history: Filled with initial_step_size
            - error_history: Filled with error_controller tolerance
        """
        return TimeIntegrationState(
            discretization_state=self.time_discretization_scheme.time_discretization_starting_scheme(
                self.ode,
                starting_values,
                self.time_integration_config.initial_step_size,
            ),
            step_size_suggestion=np.array(
                [self.time_integration_config.initial_step_size]
            ),
            step_size_history=np.full(
                2, self.time_integration_config.initial_step_size
            ),
            error_history=np.full(2, self.error_controller.config.tol),
        )

    def starting_scheme_derivative(
        self,
        starting_values: StartingValues,
        starting_value_perturbations: StartingValues,
    ) -> TimeIntegrationState:
        """
        Computes the derivative of initialization with respect to perturbed starting values.

        Parameters
        ----------
        starting_values : StartingValues
            Unperturbed initial conditions (linearization point)
        starting_value_perturbations : StartingValues
            Perturbations to initial conditions

        Returns
        -------
        TimeIntegrationState
            Perturbed initial integration state with:
            - discretization_state: Derivative of initial state w.r.t. starting values
            - Empty step_size_suggestion, step_size_history, error_history
        """
        return TimeIntegrationState(
            discretization_state=self.time_discretization_scheme.time_discretization_starting_scheme_derivative(
                self.ode,
                starting_values,
                starting_value_perturbations,
                self.time_integration_config.initial_step_size,
            ),
            step_size_suggestion=np.zeros(0),
            step_size_history=np.zeros(0),
            error_history=np.zeros(0),
        )

    def starting_scheme_adjoint_derivative(
        self,
        starting_values: StartingValues,
        integration_state_perturbations: TimeIntegrationState,
    ) -> StartingValues:
        """
        Computes the adjoint derivative of initialization.

        Parameters
        ----------
        starting_values : StartingValues
            Unperturbed initial conditions (linearization point)
        integration_state_perturbations : TimeIntegrationState
            Perturbation of initial integration state (adjoint variable)

        Returns
        -------
        StartingValues
            Perturbation of starting values (gradient)
        """
        return self.time_discretization_scheme.time_discretization_starting_scheme_adjoint_derivative(
            self.ode,
            starting_values,
            integration_state_perturbations.discretization_state,
            self.time_integration_config.initial_step_size,
        )

    def finalization_scheme(
        self, integration_state: TimeIntegrationState
    ) -> FinalizationValues:
        """
        Finalizes the integration and extracts final values.

        Parameters
        ----------
        integration_state : TimeIntegrationState
            The final state after completing all time steps

        Returns
        -------
        FinalizationValues
            Problem-specific final values (final time and state values)
        """
        return self.time_discretization_scheme.time_discretization_finalization_scheme(
            self.ode,
            integration_state.discretization_state,
            integration_state.step_size_history[0],
        )

    def finalization_scheme_derivative(
        self,
        integration_state: TimeIntegrationState,
        integration_state_perturbations: TimeIntegrationState,
    ) -> FinalizationValues:
        """
        Computes the derivative of finalization.

        Parameters
        ----------
        integration_state : TimeIntegrationState
            The final state (linearization point)
        integration_state_perturbations : TimeIntegrationState
            Perturbation of final state

        Returns
        -------
        FinalizationValues
            Perturbation of final values
        """
        return self.time_discretization_scheme.time_discretization_finalization_scheme_derivative(
            self.ode,
            integration_state.discretization_state,
            integration_state_perturbations.discretization_state,
            integration_state.step_size_history[0],
        )

    def finalization_scheme_adjoint_derivative(
        self,
        integration_state: TimeIntegrationState,
        finalization_value_perturbations: FinalizationValues,
    ) -> TimeIntegrationState:
        """
        Computes the adjoint derivative of finalization.

        Parameters
        ----------
        integration_state : TimeIntegrationState
            The final state (linearization point)
        finalization_value_perturbations : FinalizationValues
            Perturbations of final values (adjoint variable)

        Returns
        -------
        TimeIntegrationState
            Perturbation of final integration state
        """
        return TimeIntegrationState(
            discretization_state=self.time_discretization_scheme.time_discretization_finalization_scheme_adjoint_derivative(
                self.ode,
                integration_state.discretization_state,
                finalization_value_perturbations,
                integration_state.step_size_history[0],
            ),
            step_size_suggestion=np.zeros(0),
            step_size_history=np.zeros(0),
            error_history=np.zeros(0),
        )
