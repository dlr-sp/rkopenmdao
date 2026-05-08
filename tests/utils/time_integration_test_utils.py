"""
TODO
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
    """Compute the inner product of two FinalizationValues objects."""
    inner = a.final_time * b.final_time
    inner += np.dot(a.final_values, b.final_values)
    inner += np.dot(a.final_independent_outputs, b.final_independent_outputs)
    return inner


def _compute_finalization_values_norm(values: FinalizationValues) -> float:
    """Compute the norm of a FinalizationValues object."""
    return _compute_finalization_values_inner_product(values, values) ** 0.5


def _compute_starting_values_inner_product(
    a: StartingValues, b: StartingValues
) -> float:
    """Compute the inner product of two StartingValues objects."""
    inner = a.initial_time * b.initial_time
    inner += np.dot(a.initial_values, b.initial_values)
    inner += np.dot(a.independent_inputs, b.independent_inputs)
    return inner


def _compute_starting_values_norm(values: StartingValues) -> float:
    """Compute the norm of a StartingValues object."""
    return _compute_starting_values_inner_product(values, values) ** 0.5


class AbstractTestTimeIntegrationUnit(ABC):
    @abstractmethod
    @pytest.fixture
    def time_integrator(
        self,
    ) -> TimeIntegrationInterface:
        """Return the TimeIntegrationInterface implementation under test."""

    @pytest.fixture
    def initial_state(self) -> TimeIntegrationState:
        return TimeIntegrationState(
            MockDiscretizationState(), np.zeros(1), np.zeros(2), np.zeros(2)
        )

    @pytest.fixture
    def initial_state_perturbation(self) -> TimeIntegrationState:
        return TimeIntegrationState(
            MockDiscretizationState(), np.zeros(0), np.zeros(0), np.zeros(0)
        )

    @pytest.fixture
    def final_state_perturbations(self) -> list[TimeIntegrationState]:
        return [
            TimeIntegrationState(
                MockDiscretizationState(), np.zeros(0), np.zeros(0), np.zeros(0)
            )
        ]

    @pytest.fixture
    def starting_values(self) -> StartingValues:
        return StartingValues(0.0, np.zeros(0), np.zeros(0))

    @pytest.fixture
    def starting_value_perturbations(self) -> StartingValues:
        return StartingValues(0.0, np.zeros(0), np.zeros(0))

    @pytest.fixture
    def finalization_value_perturbations(self) -> FinalizationValues:
        return FinalizationValues(0.0, np.zeros(0), np.zeros(0))

    def test_create_empty_primal_integration_state(
        self, time_integrator: TimeIntegrationInterface
    ):
        assert isinstance(
            time_integrator.create_empty_primal_integration_state(),
            TimeIntegrationState,
        )

    def test_create_empty_derivative_integration_state(
        self, time_integrator: TimeIntegrationInterface
    ):
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
        assert isinstance(
            time_integrator.integrate_adjoint_derivative(
                initial_state, final_state_perturbations
            ),
            TimeIntegrationState,
        )

    def test_starting_scheme(
        self, time_integrator: TimeIntegrationInterface, starting_values: StartingValues
    ):
        assert isinstance(
            time_integrator.starting_scheme(starting_values), TimeIntegrationState
        )

    def test_starting_scheme_derivative(
        self,
        time_integrator: TimeIntegrationInterface,
        starting_values: StartingValues,
        starting_value_perturbations: StartingValues,
    ):
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
        assert isinstance(
            time_integrator.finalization_scheme(initial_state), FinalizationValues
        )

    def test_finalization_scheme_derivative(
        self,
        time_integrator: TimeIntegrationInterface,
        initial_state: TimeIntegrationState,
        initial_state_perturbation: TimeIntegrationState,
    ):
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
        assert isinstance(
            time_integrator.finalization_scheme_adjoint_derivative(
                initial_state, finalization_value_perturbations
            ),
            TimeIntegrationState,
        )


class AbstractTestTimeIntegrationSystem(ABC):
    @abstractmethod
    @pytest.fixture
    def time_integrator_creator(
        self,
    ) -> Callable[[float], TimeIntegrationInterface]:
        """Return the TimeIntegrationInterface implementation under test."""

    @abstractmethod
    @pytest.fixture
    def initial_state(self) -> StartingValues:
        """Return the initial state for the time integration problem."""

    @abstractmethod
    @pytest.fixture
    def initial_state_perturbations(self) -> StartingValues:
        """Return a perturbation of the initial state for derivative tests."""

    @abstractmethod
    @pytest.fixture
    def final_state_perturbations(self) -> FinalizationValues:
        """Return a perturbation of the final state for adjoint derivative tests."""

    @abstractmethod
    @pytest.fixture
    def expected_order(self) -> float:
        """"""

    @abstractmethod
    @pytest.fixture
    def reference_solution(self) -> FinalizationValues:
        """"""

    @abstractmethod
    @pytest.fixture
    def reference_derivative(self) -> FinalizationValues:
        """"""

    @abstractmethod
    @pytest.fixture
    def reference_adjoint_derivative(self) -> StartingValues:
        """"""

    def test_derivative_duality(
        self,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        initial_state: StartingValues,
        initial_state_perturbations: StartingValues,
        final_state_perturbations: FinalizationValues,
    ):
        """Test the duality relationship between forward and adjoint derivatives.

        Verifies that ``<dy, J*dx> = <J^T*dy, dx>`` where J is the Jacobian
        of the full time integration map.
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
    """Abstract base class for testing implementations of TimeIntegrationInterface.

    This class provides a template for writing tests that verify the correctness
    of time integration implementations, including convergence orders and
    duality relationships between forward and adjoint derivatives.
    """

    def _integrate_with_step_size_and_compare_to_reference(
        self,
        step_size: float,
        initial_state: StartingValues,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        reference_solution: FinalizationValues,
    ) -> float:
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
        and verifies that the error decreases at rate ``expected_order``.
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
        and verifies that the error in the derivative decreases at rate
        ``expected_order``.
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
        rate ``expected_order``.
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
    def test_integrate_accuracy(
        self,
        time_integrator_creator: Callable[[float], TimeIntegrationInterface],
        initial_state: StartingValues,
        reference_solution: FinalizationValues,
    ):
        time_integrator = time_integrator_creator(0.01)
        initial_tstate = time_integrator.starting_scheme(initial_state)
        time_states = time_integrator.integrate(initial_tstate)
        final_state = time_integrator.finalization_scheme(time_states[-1])
        global_error = DiscretizedODEResultState(
            np.zeros(time_integrator.ode.get_state_size()),
            final_state.final_values - reference_solution.final_values,
            np.zeros(time_integrator.ode.get_independent_output_size()),
        )
        print(
            time_integrator.ode.compute_state_norm(global_error),
            time_integrator.error_controller.config.tol,
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
        time_integrator = time_integrator_creator(0.01)
        initial_tstate = time_integrator.starting_scheme(initial_state)
        time_integrator.integrate(initial_tstate)
        print(time_integrator.integrate_callbacks[0].step_sizes)
        assert len(time_integrator.integrate_callbacks[0].step_sizes) > 1
