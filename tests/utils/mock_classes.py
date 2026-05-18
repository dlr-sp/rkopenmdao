"""
Mock classes for testing time integration implementations.

This module provides simplified mock implementations of ODE and time
discretization interfaces for unit testing purposes. These mocks return
zero-filled results and are designed to test interface compliance rather
than numerical accuracy.
"""

import numpy as np

from rkopenmdao.discretized_ode.discretized_ode import (
    DiscretizedODE,
    DiscretizedODEInputState,
    DiscretizedODEResultState,
)
from rkopenmdao.states import TimeDiscretizationStateInterface
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationSchemeInterface,
    StartingValues,
    FinalizationValues,
)


class MockODE(DiscretizedODE):
    """Mock ODE implementation for testing time integration schemes.

    This class provides a minimal implementation of the DiscretizedODE
    interface that returns zero-filled results for all computations. It is
    designed for testing interface compliance and framework structure rather
    than numerical accuracy.

    Notes
    -----
    All method implementations return zero arrays with appropriate dimensions.
    The state size and input/output sizes are all zero, making this suitable
    for testing code paths that do not depend on actual ODE dynamics.
    """

    def compute_update(self, ode_input, step_size, stage_factor):
        return DiscretizedODEResultState(
            np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        )

    def compute_update_derivative(
        self, ode_input_perturbation, step_size, stage_factor
    ):
        return DiscretizedODEResultState(np.zeros(0), np.zeros(0), np.zeros(0))

    def compute_update_adjoint_derivative(
        self, ode_result_perturbation, step_size, stage_factor
    ):
        """Compute the adjoint derivative of the ODE update.

        Parameters
        ----------
        ode_result_perturbation : DiscretizedODEResultState
            Perturbation of the ODE result state.
        step_size : float
            Current time step size.
        stage_factor : float
            Stage factor for the Runge-Kutta scheme.

        Returns
        -------
        DiscretizedODEInputState
            Zero-filled adjoint derivative state.
        """
        return DiscretizedODEInputState(np.zeros(0), np.zeros(0), np.zeros(0), 0.0)

    def compute_state_norm(self, state):
        """Compute the norm of a state vector.

        Parameters
        ----------
        state : DiscretizedODEResultState
            State vector to compute norm of.

        Returns
        -------
        float
            Zero norm (since state size is zero).
        """
        return 0.0

    def get_state_size(self):
        """Return the size of the ODE state vector.

        Returns
        -------
        int
            Zero (this mock has no state variables).
        """
        return 0

    def get_independent_input_size(self):
        """Return the size of independent input variables.

        Returns
        -------
        int
            Zero (this mock has no independent inputs).
        """
        return 0

    def get_independent_output_size(self):
        """Return the size of independent output variables.

        Returns
        -------
        int
            Zero (this mock has no independent outputs).
        """
        return 0

    def get_linearization_point_size(self):
        """Return the size of the linearization point.

        Returns
        -------
        int
            Zero (this mock has no linearization variables).
        """
        return 0


class MockDiscretizationState(TimeDiscretizationStateInterface):
    """Mock time discretization state for testing.

    This class provides a minimal implementation of the
    TimeDiscretizationStateInterface that stores no actual state data.
    It is designed for testing the framework structure without committing
    to specific state representations.

    Notes
    -----
    All state operations are no-ops since this mock has no actual state
    data to manage.
    """

    def set(self, other):
        """Set this state from another state.

        Parameters
        ----------
        other : MockDiscretizationState
            Source state (ignored since this mock has no state data).
        """

    def to_dict(self):
        return {}

    @classmethod
    def from_dict(cls, state_dict):
        """Create a MockDiscretizationState from a dictionary.

        Parameters
        ----------
        state_dict : dict
            Dictionary of state data (ignored since this mock has no state).

        Returns
        -------
        MockDiscretizationState
            New mock state instance.
        """
        return MockDiscretizationState()


class MockDiscretization(TimeDiscretizationSchemeInterface):
    """Mock time discretization scheme for testing.

    This class provides a minimal implementation of the
    TimeDiscretizationSchemeInterface that returns zero-filled results
    for all computations. It is designed for testing interface compliance
    and framework structure rather than numerical accuracy.

    Notes
    -----
    All method implementations return zero-filled or empty results.
    This mock is suitable for testing code paths that do not depend on
    actual time discretization dynamics.
    """

    def create_empty_discretization_state(self, ode):
        """Create an empty discretization state.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object (ignored).

        Returns
        -------
        MockDiscretizationState
            Empty mock state.
        """
        return MockDiscretizationState()

    def compute_step(self, ode, time_discretization_state, step_size):
        """Compute a single time step.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object (ignored).
        time_discretization_state : TimeDiscretizationStateInterface
            Current state (ignored).
        step_size : float
            Time step size (ignored).

        Returns
        -------
        MockDiscretizationState
            Empty mock state.
        """
        return MockDiscretizationState()

    def compute_step_derivative(
        self,
        ode,
        time_discretization_state,
        time_discretization_state_perturbation,
        step_size,
    ):
        """Compute derivative of a time step.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object (ignored).
        time_discretization_state : TimeDiscretizationStateInterface
            Current state (ignored).
        time_discretization_state_perturbation : TimeDiscretizationStateInterface
            State perturbation (ignored).
        step_size : float
            Time step size (ignored).

        Returns
        -------
        MockDiscretizationState
            Empty mock state.
        """
        return MockDiscretizationState()

    def compute_step_adjoint_derivative(
        self,
        ode,
        time_discretization_state,
        time_discretization_state_perturbation,
        step_size,
    ):
        """Compute adjoint derivative of a time step.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object (ignored).
        time_discretization_state : TimeDiscretizationStateInterface
            Current state (ignored).
        time_discretization_state_perturbation : TimeDiscretizationStateInterface
            State perturbation (ignored).
        step_size : float
            Time step size (ignored).

        Returns
        -------
        MockDiscretizationState
            Empty mock state.
        """
        return MockDiscretizationState()

    def time_discretization_starting_scheme(self, ode, starting_values, step_size):
        """Compute starting values for time discretization.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object (ignored).
        starting_values : StartingValues
            Initial values (ignored).
        step_size : float
            Time step size (ignored).

        Returns
        -------
        MockDiscretizationState
            Empty mock state.
        """
        return MockDiscretizationState()

    def time_discretization_starting_scheme_derivative(
        self, ode, starting_values, starting_value_perturbations, step_size
    ):
        """Compute derivative of starting values.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object (ignored).
        starting_values : StartingValues
            Initial values (ignored).
        starting_value_perturbations : StartingValues
            Initial value perturbations (ignored).
        step_size : float
            Time step size (ignored).

        Returns
        -------
        MockDiscretizationState
            Empty mock state.
        """
        return MockDiscretizationState()

    def time_discretization_starting_scheme_adjoint_derivative(
        self,
        ode,
        starting_values,
        started_discretization_state_perturbations,
        step_size,
    ):
        """Compute adjoint derivative of starting values.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object (ignored).
        starting_values : StartingValues
            Initial values (ignored).
        started_discretization_state_perturbations : MockDiscretizationState
            Started discretization state perturbations (ignored).
        step_size : float
            Time step size (ignored).

        Returns
        -------
        StartingValues
            Zero-filled starting values.
        """
        return StartingValues(0.0, np.zeros(0), np.zeros(0))

    def time_discretization_finalization_scheme(
        self, ode, discretization_state, step_size
    ):
        """Compute finalization values for time discretization.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object (ignored).
        discretization_state : TimeDiscretizationStateInterface
            Final state (ignored).
        step_size : float
            Time step size (ignored).

        Returns
        -------
        FinalizationValues
            Zero-filled finalization values.
        """
        return FinalizationValues(0.0, np.zeros(0), np.zeros(0))

    def time_discretization_finalization_scheme_derivative(
        self, ode, discretization_state, discretization_state_perturbations, step_size
    ):
        """Compute derivative of finalization values.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object (ignored).
        discretization_state : TimeDiscretizationStateInterface
            Final state (ignored).
        discretization_state_perturbations : TimeDiscretizationStateInterface
            Final state perturbations (ignored).
        step_size : float
            Time step size (ignored).

        Returns
        -------
        FinalizationValues
            Zero-filled finalization values.
        """
        return FinalizationValues(0.0, np.zeros(0), np.zeros(0))

    def time_discretization_finalization_scheme_adjoint_derivative(
        self, ode, discretization_state, finalization_value_perturbations, step_size
    ):
        """Compute adjoint derivative of finalization values.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object (ignored).
        discretization_state : TimeDiscretizationStateInterface
            Final state (ignored).
        finalization_value_perturbations : FinalizationValues
            Finalization value perturbations (ignored).
        step_size : float
            Time step size (ignored).

        Returns
        -------
        MockDiscretizationState
            Empty mock state.
        """
        return MockDiscretizationState()

    def get_ode_state(self, ode, discretization_state, step_size):
        """Extract ODE state from discretization state.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object.
        discretization_state : TimeDiscretizationStateInterface
            Discretization state (ignored).
        step_size : float
            Time step size (ignored).

        Returns
        -------
        DiscretizedODEResultState
            Zero-filled ODE state with appropriate dimensions.
        """
        return DiscretizedODEResultState(
            np.zeros(ode.get_state_size()),
            np.zeros(ode.get_state_size()),
            np.zeros(ode.get_independent_input_size()),
        )

    def get_ode_error_estimate(self, ode, discretization_state, step_size):
        """Extract ODE error estimate from discretization state.

        Parameters
        ----------
        ode : DiscretizedODE
            ODE object.
        discretization_state : TimeDiscretizationStateInterface
            Discretization state (ignored).
        step_size : float
            Time step size (ignored).

        Returns
        -------
        DiscretizedODEResultState
            Zero-filled error estimate with appropriate dimensions.
        """
        return DiscretizedODEResultState(
            np.zeros(ode.get_state_size()),
            np.zeros(ode.get_state_size()),
            np.zeros(ode.get_independent_input_size()),
        )
