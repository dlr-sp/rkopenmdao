"""Interface for checkpointing implementations in RKOpenMDAO."""

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
    Abstract interface for checkpointing implementations.

    This class defines the interface required for different checkpointing strategies to
    be implemented in a consistent manner. It provides an interface for creating
    checkpointers and iterating forward and reverse through time.

    Parameters
    ----------
    TODO
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
        return TimeIntegrationState(
            self.time_discretization_scheme.create_empty_discretization_state(self.ode),
            np.zeros(1),
            np.zeros(2),
            np.zeros(2),
        )

    def create_empty_derivative_integration_state(self) -> TimeIntegrationState:
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
        Note: time integration state is changed, but not explicitely returned.
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
        return self.time_discretization_scheme.time_discretization_starting_scheme_adjoint_derivative(
            self.ode,
            starting_values,
            integration_state_perturbations.discretization_state,
            self.time_integration_config.initial_step_size,
        )

    def finalization_scheme(
        self, integration_state: TimeIntegrationState
    ) -> FinalizationValues:
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
