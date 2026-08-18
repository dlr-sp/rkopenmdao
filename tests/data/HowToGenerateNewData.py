"""
Reference script that generates `time_step_0.txt` for a regression test, configured
with the following parameters:
- `Test case`: `TestComp1`
- `Initial time`: `0.0`
- `End time`: `0.01`,
- `Initial step size`: `0.01`
- `initial value`: `1.0`
- `Butcher tableau`: `embedded_heun_euler`
- `Error controller`: `Integral`
- `Error controller Tolerance`: `1e-6`
- `Error measurer`: `SimpleErrorMeasurer`
"""

import openmdao.api as om

from rkopenmdao.butcher_tableaux import embedded_heun_euler as heun_euler
from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import integral
from rkopenmdao.error_measurer import SimpleErrorMeasurer
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.termination_criterion import PredefinedFinalTime

from ..utils.callback import TimeStepsLog, save_data
from ..test_components import TestComp1


def integration_cfg():
    """Integration configuration factory"""
    return IntegrationConfig(
        use_adaptive_time_stepping=True,
        termination_criterion=PredefinedFinalTime(0.01),
        initial_step_size=0.01,
    )


def time_stage_problem():
    """Time problem factory"""
    prob = om.Problem()
    prob.model.add_subsystem("test_comp", TestComp1())
    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    prob.model.linear_solver = om.ScipyKrylov()
    return prob


callbacks = [TimeStepsLog()]
#  Build outer RK problem
rk = om.Problem()
rk.model.add_subsystem(
    "rk_integrator",
    RungeKuttaIntegrator(
        time_stage_problem=time_stage_problem(),
        butcher_tableau=heun_euler,
        integration_config=integration_cfg(),
        time_integration_quantities=["x"],
        error_controller=[integral],
        error_controller_options={"config": ErrorControllerConfig(tol=1e-6)},
        error_measurer=SimpleErrorMeasurer(),
        compute_callbacks=callbacks or [],
    ),
    promotes=["*"],
)
rk.setup()
rk["x_initial"] = 1.0
rk.run_model()
save_data(callbacks[0], write_file="time_step_0.txt")
