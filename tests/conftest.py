"""Test configuration and fixtures for RKOpenMDAO test suite.

This module provides:
- Pytest fixtures for parameterized testing of Runge-Kutta discretizations
- Fixture data for various ODE test problems with reference solutions
- Error controller and measurer fixtures for integration tests

The fixtures support testing of:
- Fixed-step and adaptive-step Runge-Kutta methods
- Primal, derivative, and adjoint derivative integration
- Error estimation and adaptive time stepping
"""

from collections.abc import Callable
from dataclasses import dataclass


import numpy as np
import pytest


from rkopenmdao.butcher_tableaux import butcher_tableau_collection
from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE
from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import error_controller_collection
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.states import StartingValues, FinalizationValues
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationSchemeInterface,
)
from rkopenmdao.time_discretization.stage_ordered_runge_kutta_discretization import (
    StageOrderedRungeKuttaDiscretization,
    StageOrderedEmbeddedRungeKuttaDiscretization,
)

from .odes import (
    IdentityODE,
    identity_ode_solution,
    identity_ode_solution_derivative,
    identity_ode_solution_adjoint_derivative,
    TimeODE,
    time_ode_solution,
    time_ode_solution_derivative,
    time_ode_solution_adjoint_derivative,
    TimeScaledIdentityODE,
    time_scaled_identity_ode_solution,
    time_scaled_identity_ode_solution_derivative,
    time_scaled_identity_ode_solution_adjoint_derivative,
    ParameterODE,
    parameter_ode_solution,
    parameter_ode_solution_derivative,
    parameter_ode_solution_adjoint_derivative,
    RootODE,
    root_ode_solution,
    root_ode_solution_derivative,
    root_ode_solution_adjoint_derivative,
)


@dataclass
class DiscretizationOrderPair:
    """Pair of time discretization scheme and its convergence order.

    Used to parameterize tests over different time discretization methods with
    their expected convergence orders.

    Attributes
    ----------
    time_discretization : TimeDiscretizationSchemeInterface
        The time discretization scheme (e.g. a Runge-Kutta method).
    order : float
        The expected convergence order of the discretization scheme.
    """

    time_discretization: TimeDiscretizationSchemeInterface
    order: float


@dataclass
class ODEWithReferenceStatesAndSolutions:
    """ODE problem with reference solutions for testing.

    Encapsulates an ODE along with reference solutions for verifying
    the accuracy of time integration schemes. Supports testing of
    primal integration, derivative computation, and adjoint derivative
    computation.

    Attributes
    ----------
    ode : DiscretizedODE
        The discretized ODE problem to be tested.
    initial_values : StartingValues
        Initial conditions for the ODE.
    initial_value_perturbations : StartingValues
        Perturbations to initial conditions for derivative tests.
    final_value_perturbations : FinalizationValues
        Perturbations to final values for adjoint tests.
    reference_solution : Callable[[StartingValues, float], FinalizationValues]
        Function computing the exact solution given initial values and final time.
    reference_derivative : Callable[[StartingValues, StartingValues, float], FinalizationValues]
        Function computing the derivative with respect to initial conditions.
    reference_adjoint_derivative : Callable[[StartingValues, FinalizationValues, float], StartingValues]
        Function computing the adjoint (reverse-mode) derivative.
    order_barrier : float, optional
        Minimum convergence order required for this ODE. ODEs with
        order_barrier < inf may not be suitable for error estimation with
        high-order methods. Default is np.inf.
    """

    ode: DiscretizedODE
    initial_values: StartingValues
    initial_value_perturbations: StartingValues
    final_value_perturbations: FinalizationValues
    reference_solution: Callable[[StartingValues, float], FinalizationValues]
    reference_derivative: Callable[
        [StartingValues, StartingValues, float], FinalizationValues
    ]
    reference_adjoint_derivative: Callable[
        [StartingValues, FinalizationValues, float], StartingValues
    ]
    order_barrier: float = np.inf


##### Available time discretizations #####


non_embedded_rk_pairs = [
    DiscretizationOrderPair(StageOrderedRungeKuttaDiscretization(tableau), tableau.p)
    for tableau in butcher_tableau_collection
    if not tableau.is_embedded
]

embedded_rk_tableaus = [
    DiscretizationOrderPair(
        StageOrderedEmbeddedRungeKuttaDiscretization(tableau), tableau.p
    )
    for tableau in butcher_tableau_collection
    if tableau.is_embedded
]


@pytest.fixture(params=non_embedded_rk_pairs + embedded_rk_tableaus)
def discretization_order_pair(
    request,
) -> DiscretizationOrderPair:
    """Parameterized fixture providing all Runge-Kutta discretizations.

    Combines non-embedded and embedded Runge-Kutta pairs to test
    both fixed-step and adaptive-step methods.

    Parameters
    ----------
    request : _pytest.fixtures.FixtureRequest
        Pytest request object containing the parameter for this fixture.

    Returns
    -------
    DiscretizationOrderPair
        Pair containing the time discretization scheme and its order.
    """
    print(request.param.time_discretization.butcher_tableau)
    return request.param


@pytest.fixture(params=embedded_rk_tableaus)
def adaptive_discretization_order_pair(
    request,
) -> DiscretizationOrderPair:
    """Parameterized fixture providing embedded Runge-Kutta discretizations.

    Embedded pairs are used for adaptive time stepping with error estimation.

    Parameters
    ----------
    request : _pytest.fixtures.FixtureRequest
        Pytest request object containing the parameter for this fixture.

    Returns
    -------
    DiscretizationOrderPair
        Pair containing the embedded Runge-Kutta discretization and its order.
    """
    print(request.param.time_discretization.butcher_tableau)
    return request.param


##### Available ODEs #####


suitable_odes_for_any_time_integration = [
    ODEWithReferenceStatesAndSolutions(
        IdentityODE(),
        StartingValues(0.0, np.array([1.0]), np.zeros(0)),
        StartingValues(0.0, np.array([1.0]), np.zeros(0)),
        FinalizationValues(0.0, np.array([1.0]), np.zeros(0)),
        identity_ode_solution,
        identity_ode_solution_derivative,
        identity_ode_solution_adjoint_derivative,
    ),
    ODEWithReferenceStatesAndSolutions(
        TimeScaledIdentityODE(),
        StartingValues(0.0, np.array([1.0]), np.zeros(0)),
        StartingValues(0.0, np.array([1.0]), np.zeros(0)),
        FinalizationValues(0.0, np.array([1.0]), np.zeros(0)),
        time_scaled_identity_ode_solution,
        time_scaled_identity_ode_solution_derivative,
        time_scaled_identity_ode_solution_adjoint_derivative,
        order_barrier=2.0,
    ),
    ODEWithReferenceStatesAndSolutions(
        TimeScaledIdentityODE(),
        StartingValues(1.0, np.array([0.5]), np.zeros(0)),
        StartingValues(1.0, np.array([0.5]), np.zeros(0)),
        FinalizationValues(1.0, np.array([0.5]), np.zeros(0)),
        time_scaled_identity_ode_solution,
        time_scaled_identity_ode_solution_derivative,
        time_scaled_identity_ode_solution_adjoint_derivative,
        order_barrier=2.0,
    ),
    ODEWithReferenceStatesAndSolutions(
        RootODE(),
        StartingValues(0.0, np.array([1.0]), np.zeros(0)),
        StartingValues(0.0, np.array([1.0]), np.zeros(0)),
        FinalizationValues(0.0, np.array([1.0]), np.zeros(0)),
        root_ode_solution,
        root_ode_solution_derivative,
        root_ode_solution_adjoint_derivative,
    ),
]


# These get integrated exactly from methods beyond a certain order
# -> error estimation breaks down
suitable_odes_only_for_homogeneous_integration = [
    ODEWithReferenceStatesAndSolutions(
        TimeODE(),
        StartingValues(0.0, np.array([1.0]), np.zeros(0)),
        StartingValues(0.0, np.array([1.0]), np.zeros(0)),
        FinalizationValues(0.0, np.array([1.0]), np.zeros(0)),
        time_ode_solution,
        time_ode_solution_derivative,
        time_ode_solution_adjoint_derivative,
    ),
    ODEWithReferenceStatesAndSolutions(
        TimeODE(),
        StartingValues(1.0, np.array([1.0]), np.zeros(0)),
        StartingValues(1.0, np.array([1.0]), np.zeros(0)),
        FinalizationValues(1.0, np.array([1.0]), np.zeros(0)),
        time_ode_solution,
        time_ode_solution_derivative,
        time_ode_solution_adjoint_derivative,
    ),
    ODEWithReferenceStatesAndSolutions(
        ParameterODE(),
        StartingValues(0.0, np.array([1.0]), np.zeros(1)),
        StartingValues(0.0, np.array([1.0]), np.zeros(1)),
        FinalizationValues(0.0, np.array([1.0]), np.zeros(0)),
        parameter_ode_solution,
        parameter_ode_solution_derivative,
        parameter_ode_solution_adjoint_derivative,
    ),
    ODEWithReferenceStatesAndSolutions(
        ParameterODE(),
        StartingValues(0.0, np.array([1.0]), np.ones(1)),
        StartingValues(0.0, np.array([1.0]), np.ones(1)),
        FinalizationValues(0.0, np.array([1.0]), np.zeros(0)),
        parameter_ode_solution,
        parameter_ode_solution_derivative,
        parameter_ode_solution_adjoint_derivative,
    ),
]


@pytest.fixture(
    params=suitable_odes_for_any_time_integration
    + suitable_odes_only_for_homogeneous_integration
)
def ode_with_reference_state_and_solution(
    request,
) -> ODEWithReferenceStatesAndSolutions:
    """Parameterized fixture providing ODE problems with reference solutions.

    Combines ODEs suitable for any time integration with those suitable
    only for homogeneous integration. Used for testing primal integration
    accuracy across different ODE types.

    Parameters
    ----------
    request : _pytest.fixtures.FixtureRequest
        Pytest request object containing the parameter for this fixture.

    Returns
    -------
    ODEWithReferenceStatesAndSolutions
        ODE problem with reference solutions for verification.
    """
    return request.param


@pytest.fixture(params=suitable_odes_for_any_time_integration)
def ode_with_reference_state_and_solution_for_adaptive(
    request,
) -> ODEWithReferenceStatesAndSolutions:
    """Parameterized fixture for adaptive integration tests.

    Provides ODEs suitable for testing adaptive time integration
    with error estimation. Excludes ODEs that get integrated exactly
    at high orders where error estimation breaks down.

    Parameters
    ----------
    request : _pytest.fixtures.FixtureRequest
        Pytest request object containing the parameter for this fixture.

    Returns
    -------
    ODEWithReferenceStatesAndSolutions
        ODE problem suitable for adaptive integration testing.
    """
    return request.param


##### Error control #####


@pytest.fixture(params=[SimpleErrorMeasurer(), ImprovedErrorMeasurer()])
def error_measurer(
    request,
) -> SimpleErrorMeasurer | ImprovedErrorMeasurer:
    """Parameterized fixture providing error measurer implementations.

    Tests both simple and improved error measurement strategies.

    Parameters
    ----------
    request : _pytest.fixtures.FixtureRequest
        Pytest request object containing the parameter for this fixture.

    Returns
    -------
    SimpleErrorMeasurer or ImprovedErrorMeasurer
        The error measurer implementation.
    """
    return request.param


# pylint: disable=redefined-outer-name
@pytest.fixture(params=error_controller_collection)
def adaptive_error_controller_and_measurer(
    request, error_measurer
) -> tuple[Callable[[float], object], SimpleErrorMeasurer | ImprovedErrorMeasurer]:
    """Parameterized fixture providing error controllers with a measurer.

    Combines each error controller in the collection with the error measurer
    fixture. Used for testing adaptive time integration with different
    error control strategies.

    Parameters
    ----------
    request : _pytest.fixtures.FixtureRequest
        Pytest request object containing the parameter for this fixture.
    error_measurer : SimpleErrorMeasurer or ImprovedErrorMeasurer
        The error measurer from the error_measurer fixture.

    Returns
    -------
    tuple
        Tuple containing:
        - Callable[[float], ErrorController]: Factory function creating error controller
          with tolerance 1e-3 and lower bound 1e-4.
        - SimpleErrorMeasurer or ImprovedErrorMeasurer: The error measurer.
    """
    return (
        lambda p: request.param(
            p, config=ErrorControllerConfig(tol=1e-3, lower_bound=1e-4)
        ),
        error_measurer,
    )
