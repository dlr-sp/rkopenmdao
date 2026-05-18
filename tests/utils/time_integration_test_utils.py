"""
Utility functions and abstract test classes for time integration testing.

This module provides helper functions and abstract base classes for writing
tests that verify the correctness of time integration implementations. It
includes convergence testing utilities, norm and inner product computations
for various state types, and abstract test classes for verifying integration
schemes.

The abstract test classes define comprehensive test suites for:
- Unit tests of TimeIntegrationInterface implementations
- System tests of time integration with homogeneous and adaptive schemes
- Verification of convergence orders and duality relationships
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pytest

from rkopenmdao.states import (
    DiscretizedODEResultState,
    StartingValues,
    FinalizationValues,
    TimeIntegrationState,
)
from rkopenmdao.time_integration_interface import TimeIntegrationInterface

from .convergence_utils import assert_function_convergence_rate
from .mock_classes import MockDiscretizationState


def _compute_finalization_values_inner_product(
    a: FinalizationValues, b: FinalizationValues
) -> float:
    """Compute the Euclidean inner product of two FinalizationValues objects.

    Parameters
    ----------
    a : FinalizationValues
        First FinalizationValues object.
    b : FinalizationValues
        Second FinalizationValues object.

    Returns
    -------
    float
        Inner product :math:`\\langle a, b \\rangle = a_{time} b_{time}
        + \\mathbf{a}_{values} \\cdot \\mathbf{b}_{values}
        + \\mathbf{a}_{outputs} \\cdot \\mathbf{b}_{outputs}`.
    """
    inner = a.final_time * b.final_time
    inner += np.dot(a.final_values, b.final_values)
    inner += np.dot(a.final_independent_outputs, b.final_independent_outputs)
    return inner


def _compute_finalization_values_norm(values: FinalizationValues) -> float:
    """Compute the Euclidean norm of a FinalizationValues object.

    Parameters
    ----------
    values : FinalizationValues
        FinalizationValues object to compute norm of.

    Returns
    -------
    float
        Norm :math:`\\|values\\| = \\sqrt{\\langle values, values \\rangle}`.
    """
    return _compute_finalization_values_inner_product(values, values) ** 0.5


def _compute_starting_values_inner_product(
    a: StartingValues, b: StartingValues
) -> float:
    """Compute the Euclidean inner product of two StartingValues objects.

    Parameters
    ----------
    a : StartingValues
        First StartingValues object.
    b : StartingValues
        Second StartingValues object.

    Returns
    -------
    float
        Inner product :math:`\\langle a, b \\rangle = a_{time} b_{time}
        + \\mathbf{a}_{values} \\cdot \\mathbf{b}_{values}
        + \\mathbf{a}_{inputs} \\cdot \\mathbf{b}_{inputs}`.
    """
    inner = a.initial_time * b.initial_time
    inner += np.dot(a.initial_values, b.initial_values)
    inner += np.dot(a.independent_inputs, b.independent_inputs)
    return inner


def _compute_starting_values_norm(values: StartingValues) -> float:
    """Compute the Euclidean norm of a StartingValues object.

    Parameters
    ----------
    values : StartingValues
        StartingValues object to compute norm of.

    Returns
    -------
    float
        Norm :math:`\\|values\\| = \\sqrt{\\langle values, values \\rangle}`.
    """
    return _compute_starting_values_inner_product(values, values) ** 0.5


class AbstractTestTimeIntegrationUnit(ABC):
    """Abstract base class for unit tests of TimeIntegrationInterface implementations.

    This class provides a template for writing unit tests that verify the
    correct behavior of individual time integration methods. It defines test
    methods for:

    - Creating empty integration states (primal and derivative)
    - Primal integration
    - Forward derivative integration
    - Adjoint derivative integration
    - Starting scheme and its derivatives
    - Finalization scheme and its derivatives

    Subclasses must implement the ``time_integrator`` fixture to return the
    TimeIntegrationInterface implementation under test.

    Parameters
    ----------
    None
        This is an abstract base class; implementations are configured through
        the ``time_integrator`` fixture.

    Notes
    -----
    This class is designed for testing low-level interface compliance rather
    than numerical accuracy. It verifies that methods return objects of the
    correct types and handle various input combinations correctly.
    """

    @abstractmethod
    @pytest.fixture
    def time_integrator(
        self,
    ) -> TimeIntegrationInterface:
        """Return the TimeIntegrationInterface implementation under test.

        Returns
        -------
        TimeIntegrationInterface
            The time integration implementation to be tested.
        """

    @pytest.fixture
    def initial_state(self) -> TimeIntegrationState:
        """Create a mock initial state for testing.

        Returns
        -------
        TimeIntegrationState
            Mock initial state with zero-filled discretization state and
            small zero-filled arrays for step sizes and perturbations.
        """
        return TimeIntegrationState(
            MockDiscretizationState(), np.zeros(1), np.zeros(2), np.zeros(2)
        )

    @pytest.fixture
    def initial_state_perturbation(self) -> TimeIntegrationState:
        """Create a mock initial state perturbation for testing.

        Returns
        -------
        TimeIntegrationState
            Mock perturbation state with zero-filled arrays.
        """
        return TimeIntegrationState(
            MockDiscretizationState(), np.zeros(0), np.zeros(0), np.zeros(0)
        )

    @pytest.fixture
    def final_state_perturbations(self) -> list[TimeIntegrationState]:
        """Create a list of mock final state perturbations for testing.

        Returns
        -------
        list[TimeIntegrationState]
            List containing a single mock perturbation state with zero-filled arrays.
        """
        return [
            TimeIntegrationState(
                MockDiscretizationState(), np.zeros(0), np.zeros(0), np.zeros(0)
            )
        ]

    @pytest.fixture
    def starting_values(self) -> StartingValues:
        """Create mock starting values for testing.

        Returns
        -------
        StartingValues
            Mock starting values with zero-filled arrays.
        """
        return StartingValues(0.0, np.zeros(0), np.zeros(0))

    @pytest.fixture
    def starting_value_perturbations(self) -> StartingValues:
        """Create mock starting value perturbations for testing.

        Returns
        -------
        StartingValues
            Mock perturbations with zero-filled arrays.
        """
        return StartingValues(0.0, np.zeros(0), np.zeros(0))

    @pytest.fixture
    def finalization_value_perturbations(self) -> FinalizationValues:
        """Create mock finalization value perturbations for testing.

        Returns
        -------
        FinalizationValues
            Mock perturbations with zero-filled arrays.
        """
        return FinalizationValues(0.0, np.zeros(0), np.zeros(0))

    def test_create_empty_primal_integration_state(
        self, time_integrator: TimeIntegrationInterface
    ):
        """Test that primal integration state creation returns correct type.

        Verifies that the ``create_empty_primal_integration_state`` method
        returns a TimeIntegrationState instance.

        Parameters
        ----------
        time_integrator : TimeIntegrationInterface
            The time integration implementation under test.

        Returns
        -------
        None
            The test passes if the returned object is a TimeIntegrationState.
        """
        assert isinstance(
            time_integrator.create_empty_primal_integration_state(),
            TimeIntegrationState,
        )

    def test_create_empty_derivative_integration_state(
        self, time_integrator: TimeIntegrationInterface
    ):
        """Test that derivative integration state creation returns correct type.

        Verifies that the ``create_empty_derivative_integration_state`` method
        returns a TimeIntegrationState instance.

        Parameters
        ----------
        time_integrator : TimeIntegrationInterface
            The time integration implementation under test.

        Returns
        -------
        None
            The test passes if the returned object is a TimeIntegrationState.
        """
        assert isinstance(
            time_integrator.create_empty_derivative_integration_state(),
            TimeIntegrationState,
        )

    def test_integrate(
        self,
        subtests,
        time_integrator: TimeIntegrationInterface,
        initial_state: TimeIntegrationState,
    ):
        """Test primal integration returns correct type.

        Verifies that the ``integrate`` method returns a list of
        TimeIntegrationState instances.

        Parameters
        ----------
        subtests : pytest.Subtests
            Pytest subtest context for granular assertions.
        time_integrator : TimeIntegrationInterface
            The time integration implementation under test.
        initial_state : TimeIntegrationState
            Initial state for integration.

        Returns
        -------
        None
            The test passes if the output is a list containing only
            TimeIntegrationState instances.
        """
        output = time_integrator.integrate(initial_state)
        with subtests.test():
            assert isinstance(output, list)
        for state in output:
            with subtests.test():
                assert isinstance(state, TimeIntegrationState)

    def test_integrate_derivative(
        self,
        subtests,
        time_integrator: TimeIntegrationInterface,
        initial_state: TimeIntegrationState,
        initial_state_perturbation: TimeIntegrationState,
    ):
        """Test forward derivative integration returns correct types.

        Verifies that the ``integrate_derivative`` method returns a tuple
        containing two lists of TimeIntegrationState instances.

        Parameters
        ----------
        subtests : pytest.Subtests
            Pytest subtest context for granular assertions.
        time_integrator : TimeIntegrationInterface
            The time integration implementation under test.
        initial_state : TimeIntegrationState
            Primal integration state (linearization point).
        initial_state_perturbation : TimeIntegrationState
            Initial perturbation for derivative computation.

        Returns
        -------
        None
            The test passes if the output is a tuple of two lists, each
            containing only TimeIntegrationState instances.
        """
        output = time_integrator.integrate_derivative(
            initial_state, initial_state_perturbation
        )
        with subtests.test():
            assert isinstance(output, tuple)
        with subtests.test():
            assert isinstance(output[0], list)
        with subtests.test():
            assert isinstance(output[1], list)
        for state in output[0]:
            assert isinstance(state, TimeIntegrationState)
        for state in output[1]:
            assert isinstance(state, TimeIntegrationState)

    def test_integrate_adjoint_derivative(
        self,
        time_integrator: TimeIntegrationInterface,
        initial_state: TimeIntegrationState,
        final_state_perturbations: list[TimeIntegrationState],
    ):
        """Test adjoint derivative integration returns correct type.

        Verifies that the ``integrate_adjoint_derivative`` method returns
        a TimeIntegrationState instance.

        Parameters
        ----------
        time_integrator : TimeIntegrationInterface
            The time integration implementation under test.
        initial_state : TimeIntegrationState
            Initial state for adjoint integration.
        final_state_perturbations : list[TimeIntegrationState]
            List of final state perturbations for adjoint computation.

        Returns
        -------
        None
            The test passes if the returned object is a TimeIntegrationState.
        """
        assert isinstance(
            time_integrator.integrate_adjoint_derivative(
                initial_state, final_state_perturbations
            ),
            TimeIntegrationState,
        )

    def test_starting_scheme(
        self, time_integrator: TimeIntegrationInterface, starting_values: StartingValues
    ):
        """Test starting scheme returns correct type.

        Verifies that the ``starting_scheme`` method returns a
        TimeIntegrationState instance.

        Parameters
        ----------
        time_integrator : TimeIntegrationInterface
            The time integration implementation under test.
        starting_values : StartingValues
            Starting values for the scheme.

        Returns
        -------
        None
            The test passes if the returned object is a TimeIntegrationState.
        """
        assert isinstance(
            time_integrator.starting_scheme(starting_values), TimeIntegrationState
        )

    def test_starting_scheme_derivative(
        self,
        time_integrator: TimeIntegrationInterface,
        starting_values: StartingValues,
        starting_value_perturbations: StartingValues,
    ):
        """Test starting scheme derivative returns correct type.

        Verifies that the ``starting_scheme_derivative`` method returns a
        TimeIntegrationState instance.

        Parameters
        ----------
        time_integrator : TimeIntegrationInterface
            The time integration implementation under test.
        starting_values : StartingValues
            Starting values for the scheme (linearization point).
        starting_value_perturbations : StartingValues
            Perturbation of starting values.

        Returns
        -------
        None
            The test passes if the returned object is a TimeIntegrationState.
        """
        assert isinstance(
            time_integrator.starting_scheme_derivative(
                starting_values, starting_value_perturbations
            ),
            TimeIntegrationState,
        )

    def test_starting_scheme_adjoint_derivative(
        self,
        time_integrator: TimeIntegrationInterface,
        starting_values: StartingValues,
        initial_state_perturbation: TimeIntegrationState,
    ):
        """Test starting scheme adjoint derivative returns correct type.

        Verifies that the ``starting_scheme_adjoint_derivative`` method
        returns a StartingValues instance.

        Parameters
        ----------
        time_integrator : TimeIntegrationInterface
            The time integration implementation under test.
        starting_values : StartingValues
            Starting values for the scheme (linearization point).
        initial_state_perturbation : TimeIntegrationState
            Perturbation of initial state.

        Returns
        -------
        None
            The test passes if the returned object is a StartingValues.
        """
        assert isinstance(
            time_integrator.starting_scheme_adjoint_derivative(
                starting_values, initial_state_perturbation
            ),
            StartingValues,
        )

    def test_finalization_scheme(
        self,
        time_integrator: TimeIntegrationInterface,
        initial_state: TimeIntegrationState,
    ):
        """Test finalization scheme returns correct type.

        Verifies that the ``finalization_scheme`` method returns a
        FinalizationValues instance.

        Parameters
        ----------
        time_integrator : TimeIntegrationInterface
            The time integration implementation under test.
        initial_state : TimeIntegrationState
            Final state for finalization.

        Returns
        -------
        None
            The test passes if the returned object is a FinalizationValues.
        """
        assert isinstance(
            time_integrator.finalization_scheme(initial_state), FinalizationValues
        )

    def test_finalization_scheme_derivative(
        self,
        time_integrator: TimeIntegrationInterface,
        initial_state: TimeIntegrationState,
        initial_state_perturbation: TimeIntegrationState,
    ):
        """Test finalization scheme derivative returns correct type.

        Verifies that the ``finalization_scheme_derivative`` method returns
        a FinalizationValues instance.

        Parameters
        ----------
        time_integrator : TimeIntegrationInterface
            The time integration implementation under test.
        initial_state : TimeIntegrationState
            Final state for finalization (linearization point).
        initial_state_perturbation : TimeIntegrationState
            Perturbation of initial state.

        Returns
        -------
        None
            The test passes if the returned object is a FinalizationValues.
        """
        assert isinstance(
            time_integrator.finalization_scheme_derivative(
                initial_state, initial_state_perturbation
            ),
            FinalizationValues,
        )

    def test_finalization_scheme_adjoint_derivative(
        self,
        time_integrator: TimeIntegrationInterface,
        initial_state: TimeIntegrationState,
        finalization_value_perturbations: FinalizationValues,
    ):
        """Test finalization scheme adjoint derivative returns correct type.

        Verifies that the ``finalization_scheme_adjoint_derivative`` method
        returns a TimeIntegrationState instance.

        Parameters
        ----------
        time_integrator : TimeIntegrationInterface
            The time integration implementation under test.
        initial_state : TimeIntegrationState
            Final state for finalization (linearization point).
        finalization_value_perturbations : FinalizationValues
            Perturbation of finalization values.

        Returns
        -------
        None
            The test passes if the returned object is a TimeIntegrationState.
        """
        assert isinstance(
            time_integrator.finalization_scheme_adjoint_derivative(
                initial_state, finalization_value_perturbations
            ),
            TimeIntegrationState,
        )


class AbstractTestTimeIntegrationSystem(ABC):
    """Abstract base class for system tests of time integration implementations.

    This class provides a template for writing system tests that verify the
    correctness of time integration schemes. It defines test methods for:

    - Duality relationship between forward and adjoint derivatives

    Subclasses must implement fixtures for:
    - ``time_integrator_creator``: Function that creates integrators with given step size
    - ``initial_state``: Initial state for the integration problem
    - ``initial_state_perturbations``: Perturbation for derivative tests
    - ``final_state_perturbations``: Perturbation for adjoint derivative tests
    - ``expected_order``: Expected convergence order
    - ``reference_solution``: Reference solution for error computation
    - ``reference_derivative``: Reference derivative for error computation
    - ``reference_adjoint_derivative``: Reference adjoint derivative for error computation

    Parameters
    ----------
    None
        This is an abstract base class; implementations are configured through
        the required fixtures.
    """

    @abstractmethod
    @pytest.fixture
    def time_integrator_creator(
        self,
    ) -> Callable[[float], TimeIntegrationInterface]:
        """Create a time integrator with a given step size.

        Returns
        -------
        Callable[[float], TimeIntegrationInterface]
            Factory function that takes a step size and returns a
            TimeIntegrationInterface instance.
        """

    @abstractmethod
    @pytest.fixture
    def initial_state(self) -> StartingValues:
        """Return the initial state for the time integration problem.

        Returns
        -------
        StartingValues
            Initial state containing initial time, values, and independent inputs.
        """

    @abstractmethod
    @pytest.fixture
    def initial_state_perturbations(self) -> StartingValues:
        """Return a perturbation of the initial state for derivative tests.

        Returns
        -------
        StartingValues
            Perturbation of initial state for computing derivatives.
        """

    @abstractmethod
    @pytest.fixture
    def final_state_perturbations(self) -> FinalizationValues:
        """Return a perturbation of the final state for adjoint derivative tests.

        Returns
        -------
        FinalizationValues
            Perturbation of final state for computing adjoint derivatives.
        """

    @abstractmethod
    @pytest.fixture
    def expected_order(self) -> float:
        """Return the expected convergence order for the time integration scheme.

        Returns
        -------
        float
            Expected order of convergence (e.g., 2.0 for second-order scheme).
        """

    @abstractmethod
    @pytest.fixture
    def reference_solution(self) -> FinalizationValues:
        """Return a reference solution for error computation.

        Returns
        -------
        FinalizationValues
            Reference solution at final time for comparing computed solutions.
        """

    @abstractmethod
    @pytest.fixture
    def reference_derivative(self) -> FinalizationValues:
        """Return a reference derivative for error computation.

        Returns
        -------
        FinalizationValues
            Reference derivative at final time for comparing computed derivatives.
        """

    @abstractmethod
    @pytest.fixture
    def reference_adjoint_derivative(self) -> StartingValues:
        """Return a reference adjoint derivative for error computation.

        Returns
        -------
        StartingValues
            Reference adjoint derivative at initial time for comparing computed adjoint derivatives.
        """

    def test_derivative_duality(
        self,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        initial_state: StartingValues,
        initial_state_perturbations: StartingValues,
        final_state_perturbations: FinalizationValues,
    ):
        """Test the duality relationship between forward and adjoint derivatives.

        Verifies that the discrete duality relationship holds:
        :math:`\\langle dy, J dx \\rangle = \\langle J^T dy, dx \\rangle`,
        where :math:`J` is the Jacobian of the full time integration map,
        :math:`dx` is a perturbation of initial conditions, and :math:`dy`
        is a perturbation of final conditions.

        Parameters
        ----------
        time_integrator_creator : Callable[[float], TimeIntegrationInterface]
            Factory function that creates integrators with given step size.
        initial_state : StartingValues
            Initial state for integration.
        initial_state_perturbations : StartingValues
            Perturbation of initial conditions.
        final_state_perturbations : FinalizationValues
            Perturbation of final conditions.

        Returns
        -------
        None
            The test passes if the forward and reverse duality computations
            match within numerical precision.

        Notes
        -----
        This test verifies the correctness of the implementation of both
        forward and adjoint derivative modes. The duality relationship is
        a fundamental property of tangent-linear and adjoint systems.
        """
        time_integrator = time_integrator_creator(0.01)
        initial_tstate = time_integrator.starting_scheme(initial_state)
        initial_tstate_backup = deepcopy(initial_tstate)

        initial_perturbation_tstate = time_integrator.starting_scheme_derivative(
            initial_state, initial_state_perturbations
        )
        primal_states, derivative_states = time_integrator.integrate_derivative(
            initial_tstate, initial_perturbation_tstate
        )
        final_derivative = time_integrator.finalization_scheme_derivative(
            primal_states[-1], derivative_states[-1]
        )

        dual_forward = _compute_finalization_values_inner_product(
            final_state_perturbations, final_derivative
        )

        adjoint_perturbation_tstate = (
            time_integrator.finalization_scheme_adjoint_derivative(
                primal_states[-1], final_state_perturbations
            )
        )
        adjoint_state = time_integrator.integrate_adjoint_derivative(
            initial_tstate_backup, [adjoint_perturbation_tstate]
        )
        final_adjoint = time_integrator.starting_scheme_adjoint_derivative(
            initial_state, adjoint_state
        )

        dual_reverse = _compute_starting_values_inner_product(
            initial_state_perturbations, final_adjoint
        )
        assert dual_reverse == pytest.approx(dual_forward)


class AbstractTestHomogeneousTimeIntegrationSystem(AbstractTestTimeIntegrationSystem):
    """Abstract base class for testing homogeneous time integration implementations.

    This class extends AbstractTestTimeIntegrationSystem with concrete implementations
    of helper methods for testing homogeneous time integration schemes (those with
    fixed time stepping). It provides methods for:

    - Computing errors with respect to reference solutions
    - Testing convergence orders for primal, derivative, and adjoint integration

    This class is suitable for testing time integration schemes that use fixed
    step sizes (e.g., explicit or implicit Runge-Kutta methods with predefined
    step sizes).

    Parameters
    ----------
    None
        This is an abstract base class; implementations are configured through
        the required fixtures of AbstractTestTimeIntegrationSystem.

    Notes
    -----
    Subclasses should override methods if they need custom error computation
    or convergence testing behavior.
    """

    def _integrate_with_step_size_and_compare_to_reference(
        self,
        step_size: float,
        initial_state: StartingValues,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        reference_solution: FinalizationValues,
    ) -> float:
        """Integrate with given step size and compute error relative to reference.

        Parameters
        ----------
        step_size : float
            Time step size for integration.
        initial_state : StartingValues
            Initial state for integration.
        time_integrator_creator : Callable[[float], TimeIntegrationInterface]
            Factory function that creates integrators with given step size.
        reference_solution : FinalizationValues
            Reference solution for computing error.

        Returns
        -------
        float
            Norm of the error between computed solution and reference.
        """
        time_integrator = time_integrator_creator(step_size)
        initial_tstate = time_integrator.starting_scheme(initial_state)
        initial_tstate.step_size_suggestion[0] = step_size
        time_states = time_integrator.integrate(initial_tstate)
        final_state = time_integrator.finalization_scheme(time_states[-1])

        error_values = FinalizationValues(
            final_state.final_time - reference_solution.final_time,
            final_state.final_values - reference_solution.final_values,
            final_state.final_independent_outputs
            - reference_solution.final_independent_outputs,
        )
        return _compute_finalization_values_norm(error_values)

    def _integrate_derivative_with_step_size_and_compare_to_reference(
        self,
        step_size: float,
        initial_state: StartingValues,
        initial_state_perturbations: StartingValues,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        reference_derivative: FinalizationValues,
    ) -> float:
        """Integrate derivative with given step size and compute error relative to reference.

        Parameters
        ----------
        step_size : float
            Time step size for integration.
        initial_state : StartingValues
            Initial state for integration (linearization point).
        initial_state_perturbations : StartingValues
            Perturbation of initial conditions for derivative computation.
        time_integrator_creator : Callable[[float], TimeIntegrationInterface]
            Factory function that creates integrators with given step size.
        reference_derivative : FinalizationValues
            Reference derivative for computing error.

        Returns
        -------
        float
            Norm of the error between computed derivative and reference.
        """
        time_integrator = time_integrator_creator(step_size)
        initial_tstate = time_integrator.starting_scheme(initial_state)
        initial_tstate.step_size_suggestion[0] = step_size
        initial_perturbation_tstate = time_integrator.starting_scheme_derivative(
            initial_state, initial_state_perturbations
        )
        primal_states, derivative_states = time_integrator.integrate_derivative(
            initial_tstate, initial_perturbation_tstate
        )
        final_derivative = time_integrator.finalization_scheme_derivative(
            primal_states[-1], derivative_states[-1]
        )

        error_derivate_value = FinalizationValues(
            final_derivative.final_time - reference_derivative.final_time,
            final_derivative.final_values - reference_derivative.final_values,
            final_derivative.final_independent_outputs
            - reference_derivative.final_independent_outputs,
        )
        return _compute_finalization_values_norm(error_derivate_value)

    def _integrate_adjoint_derivative_with_step_size_and_compare_to_reference(
        self,
        step_size: float,
        initial_state: StartingValues,
        final_state_perturbations: FinalizationValues,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        reference_adjoint_derivative: StartingValues,
    ) -> float:
        """Integrate adjoint derivative with given step size and compute error relative to reference.

        Parameters
        ----------
        step_size : float
            Time step size for integration.
        initial_state : StartingValues
            Initial state for integration (linearization point).
        final_state_perturbations : FinalizationValues
            Perturbation of final conditions for adjoint computation.
        time_integrator_creator : Callable[[float], TimeIntegrationInterface]
            Factory function that creates integrators with given step size.
        reference_adjoint_derivative : StartingValues
            Reference adjoint derivative for computing error.

        Returns
        -------
        float
            Norm of the error between computed adjoint derivative and reference.
        """
        time_integrator = time_integrator_creator(step_size)
        initial_tstate = time_integrator.starting_scheme(initial_state)
        initial_tstate.step_size_suggestion[0] = step_size
        initial_tstate_backup = deepcopy(initial_tstate)

        final_state = time_integrator.integrate(initial_tstate)

        adjoint_perturbation_tstate = (
            time_integrator.finalization_scheme_adjoint_derivative(
                final_state[0], final_state_perturbations
            )
        )

        adjoint_state = time_integrator.integrate_adjoint_derivative(
            initial_tstate_backup, [adjoint_perturbation_tstate]
        )
        final_adjoint = time_integrator.starting_scheme_adjoint_derivative(
            initial_state, adjoint_state
        )
        error_derivative_value = StartingValues(
            final_adjoint.initial_time - reference_adjoint_derivative.initial_time,
            final_adjoint.initial_values - reference_adjoint_derivative.initial_values,
            (
                final_adjoint.independent_inputs
                - reference_adjoint_derivative.independent_inputs
            ),
        )
        return _compute_starting_values_norm(error_derivative_value)

    def test_integrate_order(
        self,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        initial_state: StartingValues,
        expected_order: float,
        reference_solution: FinalizationValues,
    ):
        """Test that primal integration converges at the expected order.

        Performs a convergence study with successively halving step sizes
        and verifies that the error decreases at the expected asymptotic rate.
        The convergence rate is verified using assert_function_convergence_rate.

        Parameters
        ----------
        time_integrator_creator : Callable[[float], TimeIntegrationInterface]
            Factory function that creates integrators with given step size.
        initial_state : StartingValues
            Initial state for integration.
        expected_order : float
            Expected order of convergence (e.g., 2.0 for second-order scheme).
        reference_solution : FinalizationValues
            Reference solution for computing errors.

        Returns
        -------
        None
            The test passes if the observed convergence rate matches the
            expected order within tolerance.

        Notes
        -----
        This test uses a logarithmic convergence study with step sizes
        ``h, h/2, h/4, h/8, h/16`` where ``h`` is the base step size (0.1).
        The convergence rate is estimated by comparing consecutive errors.
        """
        assert_function_convergence_rate(
            self._integrate_with_step_size_and_compare_to_reference,
            0.1,
            {
                "initial_state": initial_state,
                "time_integrator_creator": time_integrator_creator,
                "reference_solution": reference_solution,
            },
            expected_order,
        )

    def test_integrate_derivative_order(
        self,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        initial_state: StartingValues,
        initial_state_perturbations: StartingValues,
        expected_order: float,
        reference_derivative: FinalizationValues,
    ):
        """Test that forward derivative integration converges at the expected order.

        Performs a convergence study with successively halving step sizes
        and verifies that the error in the derivative decreases at the
        expected asymptotic rate.

        Parameters
        ----------
        time_integrator_creator : Callable[[float], TimeIntegrationInterface]
            Factory function that creates integrators with given step size.
        initial_state : StartingValues
            Initial state for integration (linearization point).
        initial_state_perturbations : StartingValues
            Perturbation of initial conditions for derivative computation.
        expected_order : float
            Expected order of convergence for the derivative.
        reference_derivative : FinalizationValues
            Reference derivative for computing errors.

        Returns
        -------
        None
            The test passes if the observed convergence rate matches the
            expected order within tolerance.

        Notes
        -----
        This test verifies that the tangent-linear (first-order) derivative
        computation converges at the expected rate.
        """
        assert_function_convergence_rate(
            self._integrate_derivative_with_step_size_and_compare_to_reference,
            0.1,
            {
                "initial_state": initial_state,
                "initial_state_perturbations": initial_state_perturbations,
                "time_integrator_creator": time_integrator_creator,
                "reference_derivative": reference_derivative,
            },
            expected_order,
        )

    def test_integrate_adjoint_derivative_order(
        self,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        initial_state: StartingValues,
        final_state_perturbations: FinalizationValues,
        expected_order: float,
        reference_adjoint_derivative: StartingValues,
    ):
        """Test that adjoint derivative integration converges at the expected order.

        Performs a convergence study with successively halving step sizes
        and verifies that the error in the adjoint derivative decreases at
        the expected asymptotic rate.

        Parameters
        ----------
        time_integrator_creator : Callable[[float], TimeIntegrationInterface]
            Factory function that creates integrators with given step size.
        initial_state : StartingValues
            Initial state for integration (linearization point).
        final_state_perturbations : FinalizationValues
            Perturbation of final conditions for adjoint computation.
        expected_order : float
            Expected order of convergence for the adjoint derivative.
        reference_adjoint_derivative : StartingValues
            Reference adjoint derivative for computing errors.

        Returns
        -------
        None
            The test passes if the observed convergence rate matches the
            expected order within tolerance.

        Notes
        -----
        This test verifies that the adjoint (reverse-mode) derivative
        computation converges at the expected rate.
        """
        assert_function_convergence_rate(
            self._integrate_adjoint_derivative_with_step_size_and_compare_to_reference,
            0.1,
            {
                "initial_state": initial_state,
                "final_state_perturbations": final_state_perturbations,
                "time_integrator_creator": time_integrator_creator,
                "reference_adjoint_derivative": reference_adjoint_derivative,
            },
            expected_order,
        )


class AbstractTestAdaptiveTimeIntegrationSystem(AbstractTestTimeIntegrationSystem):
    """Abstract base class for testing adaptive time integration implementations.

    This class extends AbstractTestTimeIntegrationSystem with methods specific
    to adaptive time integration schemes that adjust step sizes based on error
    estimates. It provides tests for:

    - Accuracy requirements (error within tolerance)
    - Adaptiveness (step sizes vary during integration)

    Parameters
    ----------
    None
        This is an abstract base class; implementations are configured through
        the required fixtures of AbstractTestTimeIntegrationSystem.

    Notes
    -----
    This class is suitable for testing adaptive time integration schemes
    that use error controllers to adjust step sizes dynamically.
    """

    def test_integrate_accuracy(
        self,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        initial_state: StartingValues,
        reference_solution: FinalizationValues,
    ):
        """Test that adaptive integration meets accuracy requirements.

        Verifies that the global error of the adaptive integration is within
        an acceptable factor of the requested tolerance. For adaptive schemes,
        the error should be roughly proportional to the tolerance parameter.

        Parameters
        ----------
        time_integrator_creator : Callable[[float], TimeIntegrationInterface]
            Factory function that creates integrators with given step size
            (initial step size suggestion, not used by adaptive schemes).
        initial_state : StartingValues
            Initial state for integration.
        reference_solution : FinalizationValues
            Reference solution for computing error.

        Returns
        -------
        None
            The test passes if the global error is less than 100 times the
            requested tolerance.

        Notes
        -----
        Adaptive schemes aim to achieve error approximately equal to the
        tolerance. The test allows a factor of 100 margin for numerical
        variability and different error estimation strategies.
        """
        time_integrator = time_integrator_creator(0.01)
        initial_tstate = time_integrator.starting_scheme(initial_state)
        time_states = time_integrator.integrate(initial_tstate)
        final_state = time_integrator.finalization_scheme(time_states[-1])
        global_error = DiscretizedODEResultState(
            np.zeros(time_integrator.ode.get_state_size()),
            final_state.final_values - reference_solution.final_values,
            np.zeros(time_integrator.ode.get_independent_output_size()),
        )
        assert (
            time_integrator.ode.compute_state_norm(global_error)
            < 100 * time_integrator.error_controller.config.tol
        )

    def test_adaptiveness(
        self,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        initial_state: StartingValues,
    ):
        """Test that the integrator uses adaptive step sizing.

        Verifies that the integrator varies its step sizes during integration
        rather than using a fixed step size throughout. This is a sanity check
        to ensure the callback system is working correctly.

        Parameters
        ----------
        time_integrator_creator : Callable[[float], TimeIntegrationInterface]
            Factory function that creates integrators with given step size
            (initial step size suggestion, not used by adaptive scheme).
        initial_state : StartingValues
            Initial state for integration.

        Returns
        -------
        None
            The test passes if more than one distinct step size is used
            during integration.

        Notes
        -----
        This test requires that the time_integrator_creator configures a
        callback that records step sizes (e.g., RecordStepSizes).
        """
        time_integrator = time_integrator_creator(0.01)
        initial_tstate = time_integrator.starting_scheme(initial_state)
        time_integrator.integrate(initial_tstate)
        # TODO: make this not hard-coded on the callbacks
        assert len(time_integrator.integrate_callbacks[0].step_sizes) > 1
