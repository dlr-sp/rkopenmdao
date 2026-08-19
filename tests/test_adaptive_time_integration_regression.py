import numpy as np
import pytest

from .odes import IdentityODE
from .utils.callback import TimeStepsLog, read_data

from rkopenmdao.butcher_tableaux import embedded_heun_euler
from rkopenmdao.states import StartingValues
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.termination_criterion import PredefinedFinalTime
from rkopenmdao.time_discretization.stage_ordered_runge_kutta_discretization import (
    StageOrderedEmbeddedRungeKuttaDiscretization,
)
from rkopenmdao.checkpointed_time_integration.no_checkpoint_time_integration import (
    NoCheckpointTimeIntegration,
)

from rkopenmdao.error_controllers import integral
from rkopenmdao.error_measurer import SimpleErrorMeasurer


def test_adaptive_step_size_regression():
    time_step_log = TimeStepsLog()
    time_integration = NoCheckpointTimeIntegration(
        ode=IdentityODE(),
        time_discretization_scheme=StageOrderedEmbeddedRungeKuttaDiscretization(
            embedded_heun_euler
        ),
        time_integration_config=IntegrationConfig(
            use_adaptive_time_stepping=True,
            termination_criterion=PredefinedFinalTime(0.01),
            initial_step_size=0.01,
        ),
        integrate_callbacks=[time_step_log],
        error_controller=integral(embedded_heun_euler.min_p_order()),
        error_measurer=SimpleErrorMeasurer(),
    )

    starting_values = StartingValues(
        initial_time=0.0, initial_values=np.ones(1), independent_inputs=np.zeros(0)
    )

    initial_state = time_integration.starting_scheme(starting_values)
    time_integration.integrate(initial_state)
    new_steps = np.array(time_step_log.time_steps)
    old_steps = np.array(read_data(f"tests/data/time_step_{0}.txt"))

    assert new_steps == pytest.approx(old_steps)
