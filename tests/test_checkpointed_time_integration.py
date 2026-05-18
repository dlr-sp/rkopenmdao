"""
Tests for checkpointed time integration implementations.

This module provides comprehensive test suites for verifying the correctness
of checkpointed time integration schemes that support primal, derivative,
and adjoint derivative computations. It tests:

- Unit-level interface compliance for different checkpointing strategies
- System-level numerical accuracy and convergence orders
- Special cases such as NotImplementedError for unsupported features

The tests cover three checkpointing strategies:
- NoCheckpointTimeIntegration (no checkpointing)
- AllCheckpointTimeIntegration (store all steps)
- PyrevolveTimeIntegration (use pyrevolve for checkpointing)
"""

import pytest

from rkopenmdao.callback import Callback
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

from .utils.mock_classes import MockODE, MockDiscretization
from .utils.time_integration_test_utils import (
    AbstractTestHomogeneousTimeIntegrationSystem,
    AbstractTestTimeIntegrationUnit,
    AbstractTestAdaptiveTimeIntegrationSystem,
)


@pytest.fixture
def homogeneous_error_controller_and_measurer():
    """Create error controller and measurer for homogeneous time integration tests.

    Returns
    -------
    tuple
        Tuple containing:
        - ErrorController: Pseudo error controller with order 1
        - SimpleErrorMeasurer: Error measurer for computing errors
    """
    return (pseudo(1), SimpleErrorMeasurer())


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
        homogeneous_error_controller_and_measurer : tuple
            Error controller and measurer fixture.

        Returns
        -------
        NoCheckpointTimeIntegration
            Time integrator configured for unit tests.
        """
        return NoCheckpointTimeIntegration(
            MockODE(),
            MockDiscretization(),
            homogeneous_error_controller_and_measurer[0],
            homogeneous_error_controller_and_measurer[1],
            IntegrationConfig(False, PredefinedNumberOfSteps(5), 1.0),
            [],
            [],
            [],
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
        homogeneous_error_controller_and_measurer : tuple
            Error controller and measurer fixture.

        Returns
        -------
        AllCheckpointTimeIntegration
            Time integrator configured for unit tests.
        """
        return AllCheckpointTimeIntegration(
            MockODE(),
            MockDiscretization(),
            homogeneous_error_controller_and_measurer[0],
            homogeneous_error_controller_and_measurer[1],
            IntegrationConfig(False, PredefinedNumberOfSteps(5), 1.0),
            [],
            [],
            [],
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
        homogeneous_error_controller_and_measurer : tuple
            Error controller and measurer fixture.

        Returns
        -------
        PyrevolveTimeIntegration
            Time integrator configured for unit tests.
        """
        return PyrevolveTimeIntegration(
            MockODE(),
            MockDiscretization(),
            homogeneous_error_controller_and_measurer[0],
            homogeneous_error_controller_and_measurer[1],
            IntegrationConfig(False, PredefinedNumberOfSteps(5), 1.0),
            [],
            [],
            [],
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
        with pytest.raises(TypeError):
            time_integrator._setup_revolver_class_type("foo")


@pytest.fixture
def homogeneous_time_integration_test_case(
    ode_with_reference_state_and_solution,
    discretization_order_pair,
    homogeneous_error_controller_and_measurer,
):
    """Create test case tuple for homogeneous time integration system tests.

    Parameters
    ----------
    ode_with_reference_state_and_solution : tuple
        ODE with reference solution fixture containing ODE, initial values,
        reference solution, and perturbation information.
    discretization_order_pair : tuple
        Discretization with order fixture containing time discretization
        scheme and convergence order.
    homogeneous_error_controller_and_measurer : tuple
        Error controller and measurer fixture.

    Returns
    -------
    tuple
        Tuple containing:
        - ODE with reference solution
        - Discretization with order
        - Error controller and measurer
    """
    return (
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
        homogeneous_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Initial values from the ODE.
        """
        return homogeneous_time_integration_test_case[0].initial_values

    @pytest.fixture
    def initial_state_perturbations(self, homogeneous_time_integration_test_case):
        """Extract initial state perturbations from test case.

        Parameters
        ----------
        homogeneous_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Initial value perturbations from the ODE.
        """
        return homogeneous_time_integration_test_case[0].initial_value_perturbations

    @pytest.fixture
    def final_state_perturbations(self, homogeneous_time_integration_test_case):
        """Extract final state perturbations from test case.

        Parameters
        ----------
        homogeneous_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Final value perturbations from the ODE.
        """
        return homogeneous_time_integration_test_case[0].final_value_perturbations

    @pytest.fixture
    def expected_order(self, homogeneous_time_integration_test_case):
        """Compute expected convergence order.

        The expected order is the minimum of the discretization order and
        the ODE's order barrier (if any).

        Parameters
        ----------
        homogeneous_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        float
            Expected convergence order, bounded by the discretization order
            and the ODE's order barrier.
        """
        return min(
            homogeneous_time_integration_test_case[1].order,
            homogeneous_time_integration_test_case[0].order_barrier,
        )

    @pytest.fixture
    def reference_solution(self, homogeneous_time_integration_test_case):
        """Extract reference solution from test case.

        Parameters
        ----------
        homogeneous_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Reference solution computed at final time 1.0 using the
            ODE's reference_solution method.
        """
        return homogeneous_time_integration_test_case[0].reference_solution(
            homogeneous_time_integration_test_case[0].initial_values, 1.0
        )

    @pytest.fixture
    def reference_derivative(self, homogeneous_time_integration_test_case):
        """Extract reference derivative from test case.

        Parameters
        ----------
        homogeneous_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Reference derivative computed at final time 1.0 using the
            ODE's reference_derivative method with initial state and
            initial state perturbations.
        """
        return homogeneous_time_integration_test_case[0].reference_derivative(
            homogeneous_time_integration_test_case[0].initial_values,
            homogeneous_time_integration_test_case[0].initial_value_perturbations,
            1.0,
        )

    @pytest.fixture
    def reference_adjoint_derivative(self, homogeneous_time_integration_test_case):
        """Extract reference adjoint derivative from test case.

        Parameters
        ----------
        homogeneous_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Reference adjoint derivative computed at final time 1.0 using
            the ODE's reference_adjoint_derivative method with initial state
            and final state perturbations.
        """
        return homogeneous_time_integration_test_case[0].reference_adjoint_derivative(
            homogeneous_time_integration_test_case[0].initial_values,
            homogeneous_time_integration_test_case[0].final_value_perturbations,
            1.0,
        )


class TestHomogeneousNoCheckpointTimeIntegrationSystem(
    AbstractTestHomogeneousCheckpointedTimeIntegrationSystem
):

    @pytest.fixture
    def time_integrator_creator(self, homogeneous_time_integration_test_case):
        """Create NoCheckpointTimeIntegration for homogeneous time integration tests.

        Parameters
        ----------
        homogeneous_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        callable
            Factory function that creates NoCheckpointTimeIntegration
            configured with the given step_size.
        """
        return lambda step_size: NoCheckpointTimeIntegration(
            ode=homogeneous_time_integration_test_case[0].ode,
            time_discretization_scheme=homogeneous_time_integration_test_case[
                1
            ].time_discretization,
            error_controller=homogeneous_time_integration_test_case[2][0],
            error_measurer=homogeneous_time_integration_test_case[2][1],
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
        final_state_perturbations : StartingValues
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
        final_state_perturbations : StartingValues
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
    @pytest.fixture
    def time_integrator_creator(self, homogeneous_time_integration_test_case):
        """Create AllCheckpointTimeIntegration for homogeneous time integration tests.

        Parameters
        ----------
        homogeneous_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        callable
            Factory function that creates AllCheckpointTimeIntegration
            configured with the given step_size.
        """
        return lambda step_size: AllCheckpointTimeIntegration(
            ode=homogeneous_time_integration_test_case[0].ode,
            time_discretization_scheme=homogeneous_time_integration_test_case[
                1
            ].time_discretization,
            error_controller=homogeneous_time_integration_test_case[2][0],
            error_measurer=homogeneous_time_integration_test_case[2][1],
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
    @pytest.fixture
    def time_integrator_creator(self, homogeneous_time_integration_test_case):
        """Create PyrevolveTimeIntegration for homogeneous time integration tests.

        Parameters
        ----------
        homogeneous_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        callable
            Factory function that creates PyrevolveTimeIntegration
            configured with the given step_size.
        """
        return lambda step_size: PyrevolveTimeIntegration(
            ode=homogeneous_time_integration_test_case[0].ode,
            time_discretization_scheme=homogeneous_time_integration_test_case[
                1
            ].time_discretization,
            error_controller=homogeneous_time_integration_test_case[2][0],
            error_measurer=homogeneous_time_integration_test_case[2][1],
            time_integration_config=IntegrationConfig(
                False, PredefinedNumberOfSteps(int(1 / step_size)), step_size
            ),
            integrate_callbacks=[],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )


@pytest.fixture
def adaptive_time_integration_test_case(
    ode_with_reference_state_and_solution_for_adaptive,
    adaptive_discretization_order_pair,
    adaptive_error_controller_and_measurer,
):
    """Create test case tuple for adaptive time integration system tests.

    Parameters
    ----------
    ode_with_reference_state_and_solution_for_adaptive : tuple
        ODE with reference solution fixture for adaptive tests.
    adaptive_discretization_order_pair : tuple
        Adaptive discretization with order fixture.
    adaptive_error_controller_and_measurer : tuple
        Adaptive error controller and measurer fixture.

    Returns
    -------
    tuple
        Tuple containing:
        - ODE with reference solution for adaptive tests
        - Adaptive discretization with order
        - Adaptive error controller and measurer
    """
    return (
        ode_with_reference_state_and_solution_for_adaptive,
        adaptive_discretization_order_pair,
        adaptive_error_controller_and_measurer,
    )


class AbstractTestAdaptiveCheckpointedTimeIntegrationSystem(
    AbstractTestAdaptiveTimeIntegrationSystem
):
    @pytest.fixture
    def initial_state(self, adaptive_time_integration_test_case):
        """Extract initial state from test case.

        Parameters
        ----------
        adaptive_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Initial values from the ODE.
        """
        return adaptive_time_integration_test_case[0].initial_values

    @pytest.fixture
    def initial_state_perturbations(self, adaptive_time_integration_test_case):
        """Extract initial state perturbations from test case.

        Parameters
        ----------
        adaptive_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Initial value perturbations from the ODE.
        """
        return adaptive_time_integration_test_case[0].initial_value_perturbations

    @pytest.fixture
    def final_state_perturbations(self, adaptive_time_integration_test_case):
        """Extract final state perturbations from test case.

        Parameters
        ----------
        adaptive_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        StartingValues
            Final value perturbations from the ODE.
        """
        return adaptive_time_integration_test_case[0].final_value_perturbations

    @pytest.fixture
    def expected_order(self, adaptive_time_integration_test_case):
        """Compute expected convergence order.

        The expected order is the minimum of the discretization order and
        the ODE's order barrier (if any).

        Parameters
        ----------
        adaptive_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        float
            Expected convergence order, bounded by the discretization order
            and the ODE's order barrier.
        """
        return min(
            adaptive_time_integration_test_case[1].order,
            adaptive_time_integration_test_case[0].order_barrier,
        )

    @pytest.fixture
    def reference_solution(self, adaptive_time_integration_test_case):
        """Extract reference solution from test case.

        Parameters
        ----------
        adaptive_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Reference solution computed at final time 1.0 using the
            ODE's reference_solution method.
        """
        return adaptive_time_integration_test_case[0].reference_solution(
            adaptive_time_integration_test_case[0].initial_values, 1.0
        )

    @pytest.fixture
    def reference_derivative(self, adaptive_time_integration_test_case):
        """Extract reference derivative from test case.

        Parameters
        ----------
        adaptive_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Reference derivative computed at final time 1.0 using the
            ODE's reference_derivative method with initial state and
            initial state perturbations.
        """
        return adaptive_time_integration_test_case[0].reference_derivative(
            adaptive_time_integration_test_case[0].initial_values,
            adaptive_time_integration_test_case[0].initial_value_perturbations,
            1.0,
        )

    @pytest.fixture
    def reference_adjoint_derivative(self, adaptive_time_integration_test_case):
        """Extract reference adjoint derivative from test case.

        Parameters
        ----------
        adaptive_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        FinalizationValues
            Reference adjoint derivative computed at final time 1.0 using
            the ODE's reference_adjoint_derivative method with initial state
            and final state perturbations.
        """
        return adaptive_time_integration_test_case[0].reference_adjoint_derivative(
            adaptive_time_integration_test_case[0].initial_values,
            adaptive_time_integration_test_case[0].final_value_perturbations,
            1.0,
        )


class RecordStepSizes(Callback):
    def __init__(self):
        self.step_sizes = set()

    def after_iteration(
        self, iteration, time_integration_state, ode, discretization_scheme
    ):
        self.step_sizes.add(time_integration_state.step_size_history[0])


class TestAdaptiveNoCheckpointTimeIntegrationSystem(
    AbstractTestAdaptiveCheckpointedTimeIntegrationSystem
):
    @pytest.fixture
    def time_integrator_creator(self, adaptive_time_integration_test_case):
        """Create NoCheckpointTimeIntegration for adaptive time integration tests.

        Parameters
        ----------
        adaptive_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        callable
            Factory function that creates NoCheckpointTimeIntegration
            configured with the given step_size.
        """
        return lambda step_size: NoCheckpointTimeIntegration(
            ode=adaptive_time_integration_test_case[0].ode,
            time_discretization_scheme=adaptive_time_integration_test_case[
                1
            ].time_discretization,
            error_controller=adaptive_time_integration_test_case[2][0](
                adaptive_time_integration_test_case[1].order
            ),
            error_measurer=adaptive_time_integration_test_case[2][1],
            time_integration_config=IntegrationConfig(
                True,
                PredefinedFinalTime(
                    1.0
                    + adaptive_time_integration_test_case[0].initial_values.initial_time
                ),
                step_size,
            ),
            integrate_callbacks=[RecordStepSizes()],
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
        final_state_perturbations : StartingValues
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
    @pytest.fixture
    def time_integrator_creator(self, adaptive_time_integration_test_case):
        """Create AllCheckpointTimeIntegration for adaptive time integration tests.

        Parameters
        ----------
        adaptive_time_integration_test_case : tuple
            Test case tuple containing (ODE with reference solution,
            discretization with order, error controller and measurer).

        Returns
        -------
        callable
            Factory function that creates AllCheckpointTimeIntegration
            configured with the given step_size.
        """
        return lambda step_size: AllCheckpointTimeIntegration(
            ode=adaptive_time_integration_test_case[0].ode,
            time_discretization_scheme=adaptive_time_integration_test_case[
                1
            ].time_discretization,
            error_controller=adaptive_time_integration_test_case[2][0](
                adaptive_time_integration_test_case[1].order
            ),
            error_measurer=adaptive_time_integration_test_case[2][1],
            time_integration_config=IntegrationConfig(
                True,
                PredefinedFinalTime(
                    1.0
                    + adaptive_time_integration_test_case[0].initial_values.initial_time
                ),
                step_size,
            ),
            integrate_callbacks=[RecordStepSizes()],
            integrate_derivative_callbacks=[],
            integrate_adjoint_derivative_callbacks=[],
        )
