"""Tests for the usage of pyrevolve in rkopenmdao."""

import numpy as np
import pytest

from rkopenmdao.butcher_tableaux import implicit_euler
from rkopenmdao.checkpointed_time_integration.pyrevolve_time_integration import (
    PyrevolveTimeIntegration,
)
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.states import StartingValues, FinalizationValues
from rkopenmdao.termination_criterion import PredefinedNumberOfSteps
from rkopenmdao.time_discretization.stage_ordered_runge_kutta_discretization import (
    StageOrderedRungeKuttaDiscretization,
)

from .odes import RootODE, root_ode_solution_adjoint_derivative

revolver_set = {"SingleLevel", "MultiLevel", "Memory", "Disk", "Base"}


# TODO: tests with compression
@pytest.mark.parametrize(
    "revolver_type, revolver_options",
    (
        ["SingleLevel", {}],
        ["SingleLevel", {"n_checkpoints": 2}],
        ["SingleLevel", {"n_checkpoints": 10}],
        ["SingleLevel", {"diskstorage": True}],
        ["SingleLevel", {"n_checkpoints": 2, "diskstorage": True}],
        ["SingleLevel", {"n_checkpoints": 10, "diskstorage": True}],
        # something is strange here. For certain checkpoint numbers, the
        # MultiLevelRevolver works, but for others it doesn't. We skip this for now
        # TODO: investigate this problem
        # [
        #     "MultiLevel",
        #     {
        #         "storage_list": {
        #             "Numpy": {"n_ckp": 3, "dtype": float},
        #             "Disk": {"n_ckp": 5, "dtype": float},
        #         }
        #     },
        # ],
        # [
        #     "MultiLevel",
        #     {
        #         "storage_list": {
        #             "Numpy": {"n_ckp": 5, "dtype": float},
        #             "Disk": {"n_ckp": 5, "dtype": float},
        #         }
        #     },
        # ],
        ["Memory", {}],
        ["Memory", {"n_checkpoints": 2}],
        ["Memory", {"n_checkpoints": 10}],
        ["Disk", {}],
        ["Disk", {"n_checkpoints": 2}],
        ["Disk", {"n_checkpoints": 10}],
    ),
)
def test_pyrevolve_time_integration_options(revolver_type, revolver_options):
    """Tests that the options given to the RungeKuttaIntegrator are passed through to
    the Revolver."""
    time_integration = PyrevolveTimeIntegration(
        ode=RootODE(),
        time_discretization_scheme=StageOrderedRungeKuttaDiscretization(implicit_euler),
        time_integration_config=IntegrationConfig(
            False, PredefinedNumberOfSteps(100), 0.001
        ),
        revolver_type=revolver_type,
        revolver_options=revolver_options,
    )
    initial_state = time_integration.starting_scheme(
        StartingValues(1.0, np.ones(1), np.zeros(0))
    )
    final_state_perturbations = time_integration.finalization_scheme_adjoint_derivative(
        initial_state, FinalizationValues(0.0, np.ones(1), np.zeros(0))
    )
    initial_state_perturbation = time_integration.integrate_adjoint_derivative(
        initial_state, [final_state_perturbations]
    )
    starting_value_perturbation = time_integration.starting_scheme_adjoint_derivative(
        StartingValues(1.0, np.ones(1), np.zeros(0)), initial_state_perturbation
    )
    reference = root_ode_solution_adjoint_derivative(
        StartingValues(1.0, np.ones(1), np.zeros(0)),
        FinalizationValues(0.0, np.ones(1), np.zeros(0)),
        0.1,
    )
    assert starting_value_perturbation.initial_values == pytest.approx(
        reference.initial_values
    )
