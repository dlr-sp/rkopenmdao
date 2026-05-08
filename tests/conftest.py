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
    time_discretization: TimeDiscretizationSchemeInterface
    order: float


@dataclass
class ODEWithReferenceStatesAndSolutions:
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
def discretization_order_pair(request) -> DiscretizationOrderPair:
    print(request.param.time_discretization.butcher_tableau)
    return request.param


@pytest.fixture(params=embedded_rk_tableaus)
def adaptive_discretization_order_pair(request) -> DiscretizationOrderPair:
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
    return request.param


@pytest.fixture(params=suitable_odes_for_any_time_integration)
def ode_with_reference_state_and_solution_for_adaptive(
    request,
) -> ODEWithReferenceStatesAndSolutions:
    return request.param


##### Error control #####


@pytest.fixture(params=[SimpleErrorMeasurer(), ImprovedErrorMeasurer()])
def error_measurer(request):
    return request.param


@pytest.fixture(params=error_controller_collection)
def adaptive_error_controller_and_measurer(request, error_measurer):
    return (
        lambda p: request.param(
            p, config=ErrorControllerConfig(tol=1e-3, lower_bound=1e-4)
        ),
        error_measurer,
    )
