import openmdao.api as om

from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import pid, integral, h0_211
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from dataclasses import dataclass
from typing import Callable
import pathlib


@dataclass
class IntegrationConfig:
    integration_control: TimeTerminationIntegrationControl
    error_controller: list[Callable[[], None]]
    error_measurer: Callable[[], ImprovedErrorMeasurer | SimpleErrorMeasurer]
    write_file: pathlib.Path
    config: dict


def rk_setup(
    problem,
    butcher_tableau,
    integration_config: IntegrationConfig,
):
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp",
        problem.problem(integration_config.integration_control, problem.stiffness_coef),
    )
    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_config.integration_control,
            time_integration_quantities=problem.quantity,
            error_controller=integration_config.error_controller,
            error_controller_options=integration_config.config,
            error_measurer=integration_config.error_measurer,
            write_out_distance=1,
        ),
        promote=["*"],
    )
    runge_kutta_prob.setup()
    for index, quantity in enumerate(problem.quantity):
        runge_kutta_prob[quantity + "_initial"].fill(
            problem.problem.get_initial_values[index]
        )
    runge_kutta_prob.run_model()
