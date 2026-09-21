"""Callback interface and implementations for use during time integration."""

from abc import ABC
from dataclasses import dataclass, field
from time import perf_counter

from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationSchemeInterface,
)
from rkopenmdao.states import TimeIntegrationState


class Callback(ABC):
    """
    Abstract interface for callbacks during the time integration.

    Before and after each iteration of the time integration, a callback can gain access
    to core parts of the time integration, namely
    - the current iteration, in order to e.g. perform iteration-periodic tasks.
    - the current state of the time integration, e.g. to perform time-periodic tasks or
        use data created by the time integration.
    - the ODE to be integrated over time, in order to have information about the
        composition of the states.
    - the discretization used by the time integration, in order to make use of the
        discretization-specific parts of the `time_integration_state`.

    Note that in principle, the callbacks are able to modify the objects passed to
    them. This can enable features like post-processing, but can also be dangerous if
    e.g. the primal and differentiated integrations run out of sync because of that.
    Best practice would be to incorporate the post-processing into the ODE already.
    """

    def before_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: DiscretizedODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ) -> None:
        """
        Function to call before a time integration performs a step.

        Parameters
        ----------
        iteration: int
            The current iteration of the time integration.
        time_integration_state: TimeIntegrationState
            Current state of the time integration.
        ode: DiscretizedODE
            ODE intgrated over time.
        discretization_scheme: TImeDiscretizationSchemeInterface
            Time discretization scheme used by the time integration.
        """

    def after_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: DiscretizedODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ) -> None:
        """
        Function to call after a time integration performs a step.

        Parameters
        ----------
        iteration: int
            The current iteration of the time integration.
        time_integration_state: TimeIntegrationState
            Current state of the time integration.
        ode: DiscretizedODE
            ODE intgrated over time.
        discretization_scheme: TImeDiscretizationSchemeInterface
            Time discretization scheme used by the time integration.
        """


@dataclass
class IterationLogging(Callback):
    """
    Callback for logging messages about the current step of time integration.

    Parameters
    ----------
    logged_function_name: str
        How the function to be logged is called.
    """

    logged_function_name: str

    def before_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: DiscretizedODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ):
        """Logs a message at the start of a time step."""
        print(f"Starting step <{iteration}> of {self.logged_function_name}.")

    def after_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: DiscretizedODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ):
        """Logs a message at the end of a time step."""
        print(f"Finishing step <{iteration}> of {self.logged_function_name}.")


class WallClockMeasurement(Callback):
    """
    Callback for measuring the time taken by a step of time integration.
    """

    def __init__(self):
        """Initializes the wall-clock timer with a zero start time."""
        self._before_time: float = 0.0

    def before_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: DiscretizedODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ):
        """Records the start of a time step on the wall-clock timer."""
        self._before_time = perf_counter()

    def after_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: DiscretizedODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ):
        """Prints the wall-clock time taken by the last time step."""
        after_time = perf_counter()
        elapsed_time = after_time - self._before_time
        print(f"Iteration took {elapsed_time} seconds.")


@dataclass
class TimeStepsLog(Callback):
    """
    Callback for saving and printing step sizes taken for each
    step of time integration.
    """

    time_steps: list = field(default_factory=lambda: [])

    def after_iteration(
        self,
        iteration: int,
        time_integration_state: TimeIntegrationState,
        ode: DiscretizedODE,
        discretization_scheme: TimeDiscretizationSchemeInterface,
    ):
        step_size = time_integration_state.step_size_history[0]
        print(f"Step size: {step_size}")
        self.time_steps.append(step_size)
