from typing import Callable
from dataclasses import dataclass, field
import openmdao.api as om
import os
import pathlib

from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import pid, integral, h0_211
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator


@dataclass
class IntegrationConfig:
    integration_control: TimeTerminationIntegrationControl
    error_controller: list[Callable[[], None]]
    error_measurer: Callable
    write_file: pathlib.Path = field(
        default_factory=lambda: pathlib.Path.cwd() / "output.h5"
    )
    options: dict = field(default_factory=dict)


def generate_path(path):
    if path[-3::] == ".h5" or path[-4::] == ".png":
        idx = path.rfind("/")
        directory_path = path[: idx + 1]
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print("Path has been created")
            print(directory_path)
    return path


def run_rk_problem(
    problem,
    butcher_tableau,
    integration_config: IntegrationConfig,
):
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp",
        problem.problem(
            integration_control=integration_config.integration_control,
            **problem.stiffness_coef
        ),
    )
    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_config.integration_control,
            time_integration_quantities=problem.quantities,
            adaptive_time_stepping=True,
            error_controller=integration_config.error_controller,
            error_controller_options=integration_config.config,
            error_measurer=integration_config.error_measurer,
            write_file=generate_path(str(integration_config.write_file)),
            write_out_distance=1,
        ),
        promotes=["*"],
    )
    runge_kutta_prob.setup()
    for index, quantity in enumerate(problem.quantities):
        runge_kutta_prob[quantity + "_initial"].fill(
            problem.problem.get_initial_values()[index]
        )
    runge_kutta_prob.run_model()
