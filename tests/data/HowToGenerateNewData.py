"""
Reference script that generates `time_step_0.txt` for a regression test, configured
with the following parameters:
- `Test case`: `IdentityODE`
- `Initial time`: `0.0`
- `End time`: `0.01`,
- `Initial step size`: `0.01`
- `initial value`: `1.0`
- `Butcher tableau`: `embedded_heun_euler`
- `Error controller`: `integral`
- `Error controller Tolerance`: `1e-6`
- `Error measurer`: `SimpleErrorMeasurer`
"""

from pathlib import Path

import numpy as np

from rkopenmdao.butcher_tableaux import embedded_heun_euler as heun_euler
from rkopenmdao.checkpointed_time_integration.no_checkpoint_time_integration import (
    NoCheckpointTimeIntegration,
)
from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import integral
from rkopenmdao.error_measurer import SimpleErrorMeasurer
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.states import StartingValues
from rkopenmdao.termination_criterion import PredefinedFinalTime
from rkopenmdao.time_discretization.stage_ordered_runge_kutta_discretization import (
    StageOrderedEmbeddedRungeKuttaDiscretization,
)

from ...src.rkopenmdao.callback import TimeStepsLog
from ..odes import IdentityODE
from ..utils.callback import save_data


def integration_cfg():
    """Integration configuration factory"""
    return IntegrationConfig(
        use_adaptive_time_stepping=True,
        termination_criterion=PredefinedFinalTime(0.01),
        initial_step_size=0.01,
    )


time_step_log = TimeStepsLog()
time_integration = NoCheckpointTimeIntegration(
    ode=IdentityODE(),
    time_discretization_scheme=StageOrderedEmbeddedRungeKuttaDiscretization(
        heun_euler
    ),
    time_integration_config=integration_cfg(),
    integrate_callbacks=[time_step_log],
    error_controller=integral(
        heun_euler.min_p_order(), config=ErrorControllerConfig(tol=1e-6)
    ),
    error_measurer=SimpleErrorMeasurer(),
)

starting_values = StartingValues(
    initial_time=0.0, initial_values=np.ones(1), independent_inputs=np.zeros(0)
)
initial_state = time_integration.starting_scheme(starting_values)
time_integration.integrate(initial_state)
save_data(time_step_log, write_file=str(Path(__file__).with_name("time_step_0.txt")))
