"""
Tests for checkpointed time integration implementations.

This module provides comprehensive test suites for verifying the correctness
of checkpointed time integration schemes that support primal, derivative,
and adjoint derivative computations. It tests:

- Unit-level interface compliance for different checkpointing strategies
- System-level numerical accuracy and convergence orders
- Adaptive-step system tests (error within tolerance, varying step sizes)
- Special cases such as NotImplementedError for unsupported features

The tests cover three checkpointing strategies:
- NoCheckpointTimeIntegration (no checkpointing)
- AllCheckpointTimeIntegration (store all steps)
- PyrevolveTimeIntegration (use pyrevolve for checkpointing)
"""

# Tests for all checkpointed time integration implementations should reside in
# one file, artificially splitting this will only hinder readability.
# pylint: disable=too-many-lines

# Different number of arguments for derived test classes is intentional, as that allows
# for more flexibility for different time integrators
# pylint: disable=arguments-differ
from dataclasses import dataclass

import pytest

from rkopenmdao.callback import TimeStepsLog
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.termination_criterion import (
    PredefinedNumberOfSteps,
    PredefinedFinalTime,
)
from rkopenmdao.checkpointed_time_integration.no_checkpoint_time_integration import (
    NoCheckpointTimeIntegration,
)
from rkopenmdao.checkpointed_time_integration.all_checkpoint_time_integration import (
    AllCheckpointTimeIntegration,
)
from rkopenmdao.checkpointed_time_integration.pyrevolve_time_integration import (
    PyrevolveTimeIntegration,
)

from rkopenmdao.error_controllers import pseudo
from rkopenmdao.error_measurer import SimpleErrorMeasurer

from .conftest import (
    DiscretizationOrderInfo,
    ErrorControllerMeasurerPair,
    ODEWithReferenceStatesAndSolutions,
)
from .utils.mock_classes import MockODE, MockDiscretization
from .utils.time_integration_test_utils import (
    AbstractTestHomogeneousTimeIntegrationSystem,
    AbstractTestTimeIntegrationUnit,
    AbstractTestAdaptiveTimeIntegrationSystem,
)


@pytest.fixture(name="homogeneous_error_controller_and_measurer")
def homogeneous_error_controller_and_measurer_fixture():
    """Create error controller and measurer pair for homogeneous tests.

    The homogeneous (fixed step size) suites do not need an
    order-dependent controller, so the factory ignores its order
    argument and always creates a pseudo controller of order 1.

    Returns
    -------
    ErrorControllerMeasurerPair
        Pair containing:
        - controller_factory: Callable[[float], ErrorController]
          ignoring its argument and returning ``pseudo(1)``.
        - error_measurer: SimpleErrorMeasurer
    """
    return ErrorControllerMeasurerPair(lambda p: pseudo(1), SimpleErrorMeasurer())


class TestNoCheckpointTimeIntegrationUnit(AbstractTestTimeIntegrationUnit):
    """Unit tests for NoCheckpointTimeIntegration implementation.

    This class tests the unit-level interface compliance of the
    NoCheckpointTimeIntegration implementation. It verifies that all
    required methods are implemented and return correct types.

    Notes
    -----
    This class does not test numerical accuracy or convergence, only
    interface compliance and basic functionality.
    """

    @pytest.fixture
    def time_integrator(self, homogeneous_error_controller_and_measurer):
        """Create NoCheckpointTimeIntegration for unit testing.

        Parameters
        ----------
        homogeneous_error_controller_and_measurer : ErrorControllerMeasurerPair
            Error controller and measurer pair.

        Returns
        -------
        NoCheckpointTimeIntegration
            Time integrator configured for unit tests.
        """
        return NoCheckpointTimeIntegration(
            MockODE(),
            MockDiscretization(),
            IntegrationConfig(False, PredefinedNumberOfSteps(5), 1.0),
            [],
            [],
            [],
            homogeneous_error_controller_and_measurer.controller_factory(1),
            homogeneous_error_controller_and_measurer.error_measurer,
        )

    def test_integrate_adjoint_derivative(
        self, time_integrator, initial_state, final_state_perturbations
    ):
        """Test that adjoint derivative integration raises NotImplementedError.

        NoCheckpointTimeIntegration does not support adjoint derivative
        computations, so this test verifies that the appropriate exception
        is raised.

        Parameters
        ----------
        time_integrator : NoCheckpointTimeIntegration
            Time integrator under test.
        initial_state : TimeIntegrationState
            Initial state (ignored).
        final_state_perturbations : list[TimeIntegrationState]
            Final state perturbations (ignored).

        Returns
        -------
        None
            The test passes if NotImplementedError is raised.
        """
        with pytest.raises(NotImplementedError):
            super().test_integrate_adjoint_derivative(
                time_integrator, initial_state, final_state_perturbations
            )


class TestAllCheckpointTimeIntegrationUnit(AbstractTestTimeIntegrationUnit):
    """Unit tests for AllCheckpointTimeIntegration implementation.

    This class tests the unit-level interface compliance of the
    AllCheckpointTimeIntegration implementation. It verifies that all
    required methods are implemented and return correct types.

    Notes
    -----
    This class does not test numerical accuracy or convergence, only
    interface compliance and basic functionality.
    """

    @pytest.fixture
    def time_integrator(self, homogeneous_error_controller_and_measurer):
        """Create AllCheckpointTimeIntegration for unit testing.

        Parameters
        ----------
        homogeneous_error_controller_and_measurer : ErrorControllerMeasurerPair
            Error controller and measurer pair.

        Returns
        -------
        AllCheckpointTimeIntegration
            Time integrator configured for unit tests.
        """
        return AllCheckpointTimeIntegration(
            MockODE(),
            MockDiscretization(),
            IntegrationConfig(False, PredefinedNumberOfSteps(5), 1.0),
            [],
            [],
            [],
            homogeneous_error_controller_and_measurer.controller_factory(1),
            homogeneous_error_controller_and_measurer.error_measurer,
        )


class TestPyrevolveTimeIntegrationUnit(AbstractTestTimeIntegrationUnit):
    """Unit tests for PyrevolveTimeIntegration implementation.

    This class tests the unit-level interface compliance of the
    PyrevolveTimeIntegration implementation. It verifies that all
    required methods are implemented and return correct types.

    Notes
    -----
    This class does not test numerical accuracy or convergence, only
    interface compliance and basic functionality.
    """

    @pytest.fixture
    def time_integrator(self, homogeneous_error_controller_and_measurer):
        """Create PyrevolveTimeIntegration for unit testing.

        Parameters
        ----------
        homogeneous_error_controller_and_measurer : ErrorControllerMeasurerPair
            Error controller and measurer pair.

        Returns
        -------
        PyrevolveTimeIntegration
            Time integrator configured for unit tests.
        """
        return PyrevolveTimeIntegration(
            MockODE(),
            MockDiscretization(),
            IntegrationConfig(False, PredefinedNumberOfSteps(5), 1.0),
            [],
            [],
            [],
            homogeneous_error_controller_and_measurer.controller_factory(1),
            homogeneous_error_controller_and_measurer.error_measurer,
        )

    def test_setup_revolver_class_error(
        self, time_integrator: PyrevolveTimeIntegration
    ):
        """Test that _setup_revolver_class_type raises TypeError for invalid input.

        PyrevolveTimeIntegration requires a valid pyrevolve revolver class.
        This test verifies that an appropriate error is raised when an
        invalid string is provided.

        Parameters
        ----------
        time_integrator : PyrevolveTimeIntegration
            Time integrator under test.

        Returns
        -------
        None
            The test passes if TypeError is raised for invalid input.
        """
        # Access of that argument is the whole point of the test.
        # pylint: disable=protected-access
        with pytest.raises(TypeError):
            time_integrator._setup_revolver_class_type("foo")


@dataclass
class TimeIntegrationTestCase:
    """Test case bundling the parameterized system-test fixtures.

    Aggregates the ODE, the time discretization, and the error control
    fixtures used by the system-level time integration test suites into
    a single object.

    Attributes
    ----------
    ode_with_reference_state_and_solution : ODEWithReferenceStatesAndSolutions
        ODE problem with reference solutions for verification.
    discretization_order_pair : DiscretizationOrderInfo
        Time discretization scheme with its convergence orders.
    error_controller_and_measurer : ErrorControllerMeasurerPair
        Error controller factory and error measurer pair.
    """

    ode_with_reference_state_and_solution: ODEWithReferenceStatesAndSolutions
    discretization_order_pair: DiscretizationOrderInfo
    error_controller_and_measurer: ErrorControllerMeasurerPair


@pytest.fixture(name="homogeneous_time_integration_test_case")
def homogeneous_time_integration_test_case_fixture(
    ode_with_reference_state_and_solution,
    discretization_order_pair,
    homogeneous_error_controller_and_measurer,
):
    """Create test case for homogeneous time integration system tests.

    Parameters
    ----------
    ode_with_reference_state_and_solution : ODEWithReferenceStatesAndSolutions
        ODE with reference solution fixture containing ODE, initial values,
        reference solution, and perturbation information.
    discretization_order_pair : DiscretizationOrderInfo
        Discretization with order fixture containing time discretization
        scheme and convergence order.
    homogeneous_error_controller_and_measurer : ErrorControllerMeasurerPair
        Error controller and measurer pair fixture.

    Returns
    -------
    TimeIntegrationTestCase
        Test case bundling:
        - ODE with reference solution
        - Discretization with order
        - Error controller and measurer pair
    """
    return TimeIntegrationTestCase(
        ode_with_reference_state_and_solution,
        discretization_order_pair,
        homogeneous_error_controller_and_measurer,
    )


class AbstractTestHomogeneousCheckpointedTimeIntegrationSystem(
    AbstractTestHomogeneousTimeIntegrationSystem
):
    """Abstract base class for homogeneous checkpointed time integration system tests.

    This class provides fixtures and helper methods for testing homogeneous
    checkpointed time integration systems (NoCheckpoint, AllCheckpoint, Pyrevolve).

    Parameters
    ----------
    None
        This is an abstract base class; implementations are configured through
        the required fixtures.

    Notes
    -----
    This class tests checkpointed time integration schemes with fixed
    step sizes and predefined termination criteria.
    """

    @pytest.fixture
    def initial_state(self, homogeneous_time_integration_test_case):
        """Extract initial state from test case.

        Parameters
        ----------
            homogeneous_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Initial values from the ODE.
        """
        ode_with_reference_state_and_solution = (
            homogeneous_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.initial_values

    @pytest.fixture
    def initial_state_perturbations(self, homogeneous_time_integration_test_case):
        """Extract initial state perturbations from test case.

        Parameters
        ----------
            homogeneous_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Initial value perturbations from the ODE.
        """
        ode_with_reference_state_and_solution = (
            homogeneous_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.initial_value_perturbations

    @pytest.fixture
    def final_state_perturbations(self, homogeneous_time_integration_test_case):
        """Extract final state perturbations from test case.

        Parameters
        ----------
            homogeneous_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Final value perturbations from the ODE.
        """
        ode_with_reference_state_and_solution = (
            homogeneous_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.final_value_perturbations

    @pytest.fixture
    def expected_order(self, homogeneous_time_integration_test_case):
        """Compute expected convergence order.

        The expected order is the minimum of the discretization order and
        the ODE's order barrier (if any).

        Parameters
        ----------
            homogeneous_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        float
            Expected convergence order, bounded by the discretization order
            and the ODE's order barrier.
        """
        ode_with_reference_state_and_solution = (
            homogeneous_time_integration_test_case.ode_with_reference_state_and_solution
        )
        discretization_order_pair = (
            homogeneous_time_integration_test_case.discretization_order_pair
        )
        return min(
            discretization_order_pair.order,
            ode_with_reference_state_and_solution.order_barrier,
        )

    @pytest.fixture
    def reference_solution(self, homogeneous_time_integration_test_case):
        """Extract reference solution from test case.

        Parameters
        ----------
            homogeneous_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Reference solution computed at final time 1.0 using the
            ODE's reference_solution method.
        """
        ode_with_reference_state_and_solution = (
            homogeneous_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.reference_solution(
            ode_with_reference_state_and_solution.initial_values,
            1.0,
        )

    @pytest.fixture
    def reference_derivative(self, homogeneous_time_integration_test_case):
        """Extract reference derivative from test case.

        Parameters
        ----------
            homogeneous_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Reference derivative computed at final time 1.0 using the
            ODE's reference_derivative method with initial state and
            initial state perturbations.
        """
        ode_with_reference_state_and_solution = (
            homogeneous_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.reference_derivative(
            ode_with_reference_state_and_solution.initial_values,
            ode_with_reference_state_and_solution.initial_value_perturbations,
            1.0,
        )

    @pytest.fixture
    def reference_adjoint_derivative(self, homogeneous_time_integration_test_case):
        """Extract reference adjoint derivative from test case.

        Parameters
        ----------
            homogeneous_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Reference adjoint derivative computed at final time 1.0 using
            the ODE's reference_adjoint_derivative method with initial state
            and final state perturbations.
        """
        ode_with_reference_state_and_solution = (
            homogeneous_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.reference_adjoint_derivative(
            ode_with_reference_state_and_solution.initial_values,
            ode_with_reference_state_and_solution.final_value_perturbations,
            1.0,
        )


class TestHomogeneousNoCheckpointTimeIntegrationSystem(
    AbstractTestHomogeneousCheckpointedTimeIntegrationSystem
):
    """System tests for the NoCheckpointTimeIntegration implementation.

    This class tests the numerical accuracy of the
    NoCheckpointTimeIntegration implementation for homogeneous ODEs with
    fixed step sizes. It verifies the convergence orders of the primal
    solution and the forward derivative.

    Notes
    -----
    This class inherits the fixtures and tests from
    AbstractTestHomogeneousCheckpointedTimeIntegrationSystem and provides
    the time_integrator_creator fixture. It additionally overrides
    test_integrate_adjoint_derivative_order and test_derivative_duality,
    expecting NotImplementedError since NoCheckpointTimeIntegration does
    not support adjoint derivative computations.
    """

    @pytest.fixture
    def time_integrator_creator(self, homogeneous_time_integration_test_case):
        """Create NoCheckpointTimeIntegration for homogeneous time integration tests.

        Parameters
        ----------
            homogeneous_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        callable
            Factory function that creates NoCheckpointTimeIntegration
            configured with the given step_size.
        """
        ode_with_reference_state_and_solution = (
            homogeneous_time_integration_test_case.ode_with_reference_state_and_solution
        )
        discretization_order_pair = (
            homogeneous_time_integration_test_case.discretization_order_pair
        )
        error_controller_and_measurer = (
            homogeneous_time_integration_test_case.error_controller_and_measurer
        )
        return lambda step_size: NoCheckpointTimeIntegration(
            ode=ode_with_reference_state_and_solution.ode,
            time_discretization_scheme=discretization_order_pair.time_discretization,
            error_controller=error_controller_and_measurer.controller_factory(0),
            error_measurer=error_controller_and_measurer.error_measurer,
            time_integration_config=IntegrationConfig(
                False, PredefinedNumberOfSteps(int(1 / step_size)), step_size
            ),
            integrate_callbacks=[],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )

    def test_integrate_adjoint_derivative_order(
        self,
        time_integrator_creator,
        initial_state,
        final_state_perturbations,
        expected_order,
        reference_adjoint_derivative,
    ):
        """Test that adjoint derivative order test raises NotImplementedError.

        NoCheckpointTimeIntegration does not support adjoint derivative
        computations, so this test verifies that the appropriate exception
        is raised when calling the parent test.

        Parameters
        ----------
        time_integrator_creator : callable
            Factory function that creates NoCheckpointTimeIntegration.
        initial_state : StartingValues
            Initial state for integration.
        final_state_perturbations : FinalizationValues
            Final state perturbations for adjoint computation.
        expected_order : float
            Expected convergence order (ignored).
        reference_adjoint_derivative : StartingValues
            Reference adjoint derivative (ignored).

        Returns
        -------
        None
            The test passes if NotImplementedError is raised.
        """
        with pytest.raises(NotImplementedError):
            return super().test_integrate_adjoint_derivative_order(
                time_integrator_creator,
                initial_state,
                final_state_perturbations,
                expected_order,
                reference_adjoint_derivative,
            )

    def test_derivative_duality(
        self,
        time_integrator_creator,
        initial_state,
        initial_state_perturbations,
        final_state_perturbations,
    ):
        """Test that derivative duality test raises NotImplementedError.

        NoCheckpointTimeIntegration does not support derivative duality
        computations, so this test verifies that the appropriate exception
        is raised when calling the parent test.

        Parameters
        ----------
        time_integrator_creator : callable
            Factory function that creates NoCheckpointTimeIntegration.
        initial_state : StartingValues
            Initial state for integration.
        initial_state_perturbations : StartingValues
            Initial state perturbations for derivative computation.
        final_state_perturbations : FinalizationValues
            Final state perturbations for adjoint computation.

        Returns
        -------
        None
            The test passes if NotImplementedError is raised.
        """
        with pytest.raises(NotImplementedError):
            super().test_derivative_duality(
                time_integrator_creator,
                initial_state,
                initial_state_perturbations,
                final_state_perturbations,
            )


class TestHomogeneousAllCheckpointTimeIntegrationSystem(
    AbstractTestHomogeneousCheckpointedTimeIntegrationSystem
):
    """System tests for the AllCheckpointTimeIntegration implementation.

    This class tests the numerical accuracy of the
    AllCheckpointTimeIntegration implementation for homogeneous ODEs with
    fixed step sizes. It verifies the convergence orders of the solution,
    derivative, and adjoint derivative computations and the duality of the
    forward and adjoint derivative computations.

    Notes
    -----
    This class inherits the fixtures and tests from
    AbstractTestHomogeneousCheckpointedTimeIntegrationSystem and only
    provides the time_integrator_creator fixture.
    """

    @pytest.fixture
    def time_integrator_creator(self, homogeneous_time_integration_test_case):
        """Create AllCheckpointTimeIntegration for homogeneous time integration tests.

        Parameters
        ----------
            homogeneous_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        callable
            Factory function that creates AllCheckpointTimeIntegration
            configured with the given step_size.
        """
        ode_with_reference_state_and_solution = (
            homogeneous_time_integration_test_case.ode_with_reference_state_and_solution
        )
        discretization_order_pair = (
            homogeneous_time_integration_test_case.discretization_order_pair
        )
        error_controller_and_measurer = (
            homogeneous_time_integration_test_case.error_controller_and_measurer
        )
        return lambda step_size: AllCheckpointTimeIntegration(
            ode=ode_with_reference_state_and_solution.ode,
            time_discretization_scheme=discretization_order_pair.time_discretization,
            error_controller=error_controller_and_measurer.controller_factory(0),
            error_measurer=error_controller_and_measurer.error_measurer,
            time_integration_config=IntegrationConfig(
                False, PredefinedNumberOfSteps(int(1 / step_size)), step_size
            ),
            integrate_callbacks=[],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )


class TestHomogeneousPyrevolveTimeIntegrationSystem(
    AbstractTestHomogeneousCheckpointedTimeIntegrationSystem
):
    """System tests for the PyrevolveTimeIntegration implementation.

    This class tests the numerical accuracy of the
    PyrevolveTimeIntegration implementation for homogeneous ODEs with
    fixed step sizes. It verifies the convergence orders of the solution,
    derivative, and adjoint derivative computations and the duality of the
    forward and adjoint derivative computations.

    Notes
    -----
    This class inherits the fixtures and tests from
    AbstractTestHomogeneousCheckpointedTimeIntegrationSystem and only
    provides the time_integrator_creator fixture.
    """

    @pytest.fixture
    def time_integrator_creator(self, homogeneous_time_integration_test_case):
        """Create PyrevolveTimeIntegration for homogeneous time integration tests.

        Parameters
        ----------
            homogeneous_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        callable
            Factory function that creates PyrevolveTimeIntegration
            configured with the given step_size.
        """
        ode_with_reference_state_and_solution = (
            homogeneous_time_integration_test_case.ode_with_reference_state_and_solution
        )
        discretization_order_pair = (
            homogeneous_time_integration_test_case.discretization_order_pair
        )
        error_controller_and_measurer = (
            homogeneous_time_integration_test_case.error_controller_and_measurer
        )
        return lambda step_size: PyrevolveTimeIntegration(
            ode=ode_with_reference_state_and_solution.ode,
            time_discretization_scheme=discretization_order_pair.time_discretization,
            error_controller=error_controller_and_measurer.controller_factory(0),
            error_measurer=error_controller_and_measurer.error_measurer,
            time_integration_config=IntegrationConfig(
                False, PredefinedNumberOfSteps(int(1 / step_size)), step_size
            ),
            integrate_callbacks=[],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )


@pytest.fixture(name="adaptive_time_integration_test_case")
def adaptive_time_integration_test_case_fixture(
    ode_with_reference_state_and_solution_for_adaptive,
    adaptive_discretization_order_pair,
    adaptive_error_controller_and_measurer,
):
    """Create test case for adaptive time integration system tests.

    Parameters
    ----------
    ode_with_reference_state_and_solution_for_adaptive :
        ODEWithReferenceStatesAndSolutions
        ODE with reference solution fixture for adaptive tests.
    adaptive_discretization_order_pair : DiscretizationOrderInfo
        Adaptive discretization with order fixture.
    adaptive_error_controller_and_measurer : ErrorControllerMeasurerPair
        Adaptive error controller and measurer pair fixture.

    Returns
    -------
    TimeIntegrationTestCase
        Test case bundling:
        - ODE with reference solution for adaptive tests
        - Adaptive discretization with order
        - Adaptive error controller and measurer pair
    """
    return TimeIntegrationTestCase(
        ode_with_reference_state_and_solution_for_adaptive,
        adaptive_discretization_order_pair,
        adaptive_error_controller_and_measurer,
    )


class AbstractTestAdaptiveCheckpointedTimeIntegrationSystem(
    AbstractTestAdaptiveTimeIntegrationSystem
):
    """Abstract base class for adaptive checkpointed time integration system
    tests.

    This class provides fixtures for testing adaptive checkpointed time
    integration systems (NoCheckpoint, AllCheckpoint, Pyrevolve) and
    extracts the test data from the adaptive_time_integration_test_case
    fixture.

    Notes
    -----
    This class is suitable for testing checkpointed time integration
    schemes with adaptive step sizes and predefined termination criteria.
    """

    @pytest.fixture
    def initial_state(self, adaptive_time_integration_test_case):
        """Extract initial state from test case.

        Parameters
        ----------
            adaptive_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Initial values from the ODE.
        """
        ode_with_reference_state_and_solution = (
            adaptive_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.initial_values

    @pytest.fixture
    def initial_state_perturbations(self, adaptive_time_integration_test_case):
        """Extract initial state perturbations from test case.

        Parameters
        ----------
            adaptive_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Initial value perturbations from the ODE.
        """
        ode_with_reference_state_and_solution = (
            adaptive_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.initial_value_perturbations

    @pytest.fixture
    def final_state_perturbations(self, adaptive_time_integration_test_case):
        """Extract final state perturbations from test case.

        Parameters
        ----------
            adaptive_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Final value perturbations from the ODE.
        """
        ode_with_reference_state_and_solution = (
            adaptive_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.final_value_perturbations

    @pytest.fixture
    def expected_order(self, adaptive_time_integration_test_case):
        """Compute expected convergence order.

        The expected order is the minimum of the discretization order and
        the ODE's order barrier (if any).

        Parameters
        ----------
            adaptive_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        float
            Expected convergence order, bounded by the discretization order
            and the ODE's order barrier.
        """
        discretization_order_pair = (
            adaptive_time_integration_test_case.discretization_order_pair
        )
        ode_with_reference_state_and_solution = (
            adaptive_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return min(
            discretization_order_pair.order,
            ode_with_reference_state_and_solution.order_barrier,
        )

    @pytest.fixture
    def reference_solution(self, adaptive_time_integration_test_case):
        """Extract reference solution from test case.

        Parameters
        ----------
            adaptive_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Reference solution computed at final time 1.0 using the
            ODE's reference_solution method.
        """
        ode_with_reference_state_and_solution = (
            adaptive_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.reference_solution(
            ode_with_reference_state_and_solution.initial_values,
            1.0,
        )

    @pytest.fixture
    def reference_derivative(self, adaptive_time_integration_test_case):
        """Extract reference derivative from test case.

        Parameters
        ----------
            adaptive_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Reference derivative computed at final time 1.0 using the
            ODE's reference_derivative method with initial state and
            initial state perturbations.
        """
        ode_with_reference_state_and_solution = (
            adaptive_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.reference_derivative(
            ode_with_reference_state_and_solution.initial_values,
            ode_with_reference_state_and_solution.initial_value_perturbations,
            1.0,
        )

    @pytest.fixture
    def reference_adjoint_derivative(self, adaptive_time_integration_test_case):
        """Extract reference adjoint derivative from test case.

        Parameters
        ----------
            adaptive_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Reference adjoint derivative computed at final time 1.0 using
            the ODE's reference_adjoint_derivative method with initial state
            and final state perturbations.
        """
        ode_with_reference_state_and_solution = (
            adaptive_time_integration_test_case.ode_with_reference_state_and_solution
        )
        return ode_with_reference_state_and_solution.reference_adjoint_derivative(
            ode_with_reference_state_and_solution.initial_values,
            ode_with_reference_state_and_solution.final_value_perturbations,
            1.0,
        )


class TestAdaptiveNoCheckpointTimeIntegrationSystem(
    AbstractTestAdaptiveCheckpointedTimeIntegrationSystem
):
    """System tests for the adaptive NoCheckpointTimeIntegration
    implementation.

    This class tests the numerical accuracy of the adaptive
    NoCheckpointTimeIntegration implementation for homogeneous ODEs. It
    verifies that the global error stays within the requested tolerance and
    that the integrator varies its step sizes during the integration.

    Notes
    -----
    NoCheckpointTimeIntegration does not support derivative duality
    computations, so the corresponding test verifies that a
    NotImplementedError is raised.
    """

    @pytest.fixture
    def time_integrator_creator(self, adaptive_time_integration_test_case):
        """Create NoCheckpointTimeIntegration for adaptive time integration tests.

        Parameters
        ----------
            adaptive_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        callable
            Factory function that creates NoCheckpointTimeIntegration
            configured with the given step_size.
        """
        ode_with_reference_state_and_solution = (
            adaptive_time_integration_test_case.ode_with_reference_state_and_solution
        )
        discretization_order_pair = (
            adaptive_time_integration_test_case.discretization_order_pair
        )
        error_controller_and_measurer = (
            adaptive_time_integration_test_case.error_controller_and_measurer
        )
        return lambda step_size: NoCheckpointTimeIntegration(
            ode=ode_with_reference_state_and_solution.ode,
            time_discretization_scheme=discretization_order_pair.time_discretization,
            error_controller=error_controller_and_measurer.controller_factory(
                discretization_order_pair.min_order
            ),
            error_measurer=error_controller_and_measurer.error_measurer,
            time_integration_config=IntegrationConfig(
                True,
                PredefinedFinalTime(
                    1.0
                    + ode_with_reference_state_and_solution.initial_values.initial_time
                ),
                step_size,
            ),
            integrate_callbacks=[TimeStepsLog()],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )

    def test_derivative_duality(
        self,
        time_integrator_creator,
        initial_state,
        initial_state_perturbations,
        final_state_perturbations,
    ):
        """Test that derivative duality raises NotImplementedError.

        NoCheckpointTimeIntegration does not support derivative duality
        computations, so this test verifies that the appropriate exception
        is raised when calling the parent test.

        Parameters
        ----------
        time_integrator_creator : callable
            Factory function that creates NoCheckpointTimeIntegration.
        initial_state : StartingValues
            Initial state of the ODE.
        initial_state_perturbations : StartingValues
            Initial state perturbations for derivative computation.
        final_state_perturbations : FinalizationValues
            Final state perturbations for adjoint computation.

        Returns
        -------
        None
            The test passes if NotImplementedError is raised.
        """
        with pytest.raises(NotImplementedError):
            super().test_derivative_duality(
                time_integrator_creator,
                initial_state,
                initial_state_perturbations,
                final_state_perturbations,
            )


class TestAdaptiveAllCheckpointTimeIntegrationSystem(
    AbstractTestAdaptiveCheckpointedTimeIntegrationSystem
):
    """System tests for the adaptive AllCheckpointTimeIntegration
    implementation.

    This class tests the numerical accuracy of the adaptive
    AllCheckpointTimeIntegration implementation for homogeneous ODEs. It
    verifies that the global error stays within the requested tolerance and
    that the integrator varies its step sizes during the integration.
    """

    @pytest.fixture
    def time_integrator_creator(self, adaptive_time_integration_test_case):
        """Create AllCheckpointTimeIntegration for adaptive time integration tests.

        Parameters
        ----------
            adaptive_time_integration_test_case : TimeIntegrationTestCase
            Test case containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        callable
            Factory function that creates AllCheckpointTimeIntegration
            configured with the given step_size.
        """
        ode_with_reference_state_and_solution = (
            adaptive_time_integration_test_case.ode_with_reference_state_and_solution
        )
        discretization_order_pair = (
            adaptive_time_integration_test_case.discretization_order_pair
        )
        error_controller_and_measurer = (
            adaptive_time_integration_test_case.error_controller_and_measurer
        )
        return lambda step_size: AllCheckpointTimeIntegration(
            ode=ode_with_reference_state_and_solution.ode,
            time_discretization_scheme=discretization_order_pair.time_discretization,
            error_controller=error_controller_and_measurer.controller_factory(
                discretization_order_pair.min_order
            ),
            error_measurer=error_controller_and_measurer.error_measurer,
            time_integration_config=IntegrationConfig(
                True,
                PredefinedFinalTime(
                    1.0
                    + ode_with_reference_state_and_solution.initial_values.initial_time
                ),
                step_size,
            ),
            integrate_callbacks=[TimeStepsLog()],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )
