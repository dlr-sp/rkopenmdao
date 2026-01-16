import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
import os
import pathlib
from typing import Tuple

import numpy as np
from openmdao.core.system import System
import openmdao.api as om

from ..butcher_tableau import ButcherTableau
from ..error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from ..file_writer import read_last_local_error
from ..integration_control import TimeTerminationIntegrationControl
from ..runge_kutta_integrator import RungeKuttaIntegrator
from ..odes.kaps import kaps_solution, KapsGroup
from ..odes.prothero_robinson_ode import ProtheroRobinson


def generate_path(path: str):
    """Generate a path for output files and create the directory automatically if it doesn't exist"""
    if path[-3::] == ".h5" or path[-4::] == ".png":
        idx = path.rfind("/")
        directory_path = path[: idx + 1]
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"The path to {directory_path} has been paved.")
    return path


@dataclass
class IntegrationConfig:
    """A class to hold the integration configuration parameters."""

    integration_control: TimeTerminationIntegrationControl
    error_controller: list[Callable[[], None]]
    error_measurer: SimpleErrorMeasurer | ImprovedErrorMeasurer
    write_file: pathlib.Path = field(
        default_factory=lambda: pathlib.Path.cwd() / "output.h5"
    )
    options: dict = field(default_factory=dict)


@dataclass
class Problem:
    """A class to hold the problem parameters and execute the integration."""

    step_sizes: np.ndarray
    quantities: list[str]
    time_objective: float
    stiffness_coef: dict
    folder_path: pathlib.Path
    problem: System
    solution: Callable[[float], float | np.ndarray]

    def get_file_path(
        self, butcher_name: str, _type: str | float
    ) -> Tuple[str, pathlib.Path]:
        """Get the file's path"""
        if isinstance(_type, str):
            if _type != "adaptive" and _type != "avg_homogeneous":
                raise ValueError(
                    "_type (str) must be either 'adaptive' or 'avg_homogeneous'"
                )
            name = butcher_name.replace(" ", "_").replace(",", "").lower()
            return name, self.folder_path / _type / f"{name}.h5"
        elif isinstance(_type, float):
            name = (
                f"data_{_type:.0E}_{butcher_name}".replace(" ", "_")
                .replace(",", "")
                .lower()
            )
            return name, self.folder_path / "homogeneous" / f"{name}.h5"
        else:
            raise ValueError("_type must be of type str or float")

    def compute_tolerance(self, butcher_name: str) -> np.floating:
        """compute the tolerance for the given Runge-Kutta scheme"""
        try:
            error = np.zeros_like(self.step_sizes, dtype=np.float64)
            for idx, step_size in enumerate(self.step_sizes):
                _, file_path = self.get_file_path(butcher_name, step_size)
                error[idx] = read_last_local_error(
                    file_path, self.time_objective, step_size
                )
            return np.average(error)
        except FileNotFoundError:
            print(f"No homogeneous data found for {butcher_name}.")
            return 1.0e-6

    def execute(
        self,
        butcher_tableau: ButcherTableau,
        integration_config: IntegrationConfig,
    ) -> None:
        """execute the RK-integration for a given problem, a Butcher tableau and integration configuration."""
        # initialize the OpenMDAO problem for the time integration model
        time_integration_prob = om.Problem()
        # add the `Problem` subsystem for the time integration model
        time_integration_prob.model.add_subsystem(
            "test_comp",
            self.problem(
                integration_control=integration_config.integration_control,
                **self.stiffness_coef,
            ),
        )
        # initialize the OpenMDAO problem for the RK integration
        runge_kutta_prob = om.Problem()
        # add the `RungeKuttaIntegrator` subsystem for the RK integration and connect the time integration model
        runge_kutta_prob.model.add_subsystem(
            "rk_integration",
            RungeKuttaIntegrator(
                time_stage_problem=time_integration_prob,
                butcher_tableau=butcher_tableau,
                integration_control=integration_config.integration_control,
                time_integration_quantities=self.quantities,
                adaptive_time_stepping=True,
                error_controller=integration_config.error_controller,
                error_controller_options=integration_config.options,
                error_measurer=integration_config.error_measurer,
                write_file=generate_path(str(integration_config.write_file)),
                write_out_distance=1,
            ),
            promotes=["*"],
        )
        # set up the OpenMDAO problem, fill the initial values for each quantity and run the RK integration
        runge_kutta_prob.setup()
        for index, quantity in enumerate(self.quantities):
            runge_kutta_prob[quantity + "_initial"].fill(
                self.problem.get_initial_values()[index]
            )
        runge_kutta_prob.run_model()


def parse_problem():
    """Parse the problem and its' options from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stiffness", type=float, default=None, help="stiffness of problem"
    )
    parser.add_argument(
        "problem",
        type=str,
        default="prothero_robinson",
        help="prothero_robinson or kaps",
    )
    parsed_args = parser.parse_args()
    if parsed_args.problem == "prothero_robinson":
        problem = (
            prothero_robinson_problem()
            if (parsed_args.stiffness is None)
            else prothero_robinson_problem(parsed_args.stiffness)
        )
    elif parsed_args.problem == "kaps":
        problem = (
            kaps_problem()
            if (parsed_args.stiffness is None)
            else kaps_problem(parsed_args.stiffness)
        )
    else:
        raise ValueError("Problem must be prothero_robinson or kaps")
    return problem


### ---------------
### Problem builders
### ---------------


def prothero_robinson_problem(_lambda: float = -1e2):
    """initialize the Prothero-Robinson problem"""

    # step sizes for the homogeneous integration
    delta_t_list = np.array(
        [
            1.0e-3,
            2.0e-3,
            4.0e-3,
            5.0e-3,
            1.0e-2,
        ],
        dtype=float,
    )
    quantities = ["x"]
    time_objective = 10.0
    stiffness_coef = {"lambda_": _lambda}
    folder_path = pathlib.Path(__file__).parent.parent / "data" / "prothero_robinson"
    # "data/prothero_robinson" should contain the "adaptive" folder for the data of adaptive .h5 runs
    # (adaptive.py), and the "homogeneous" folder for the homogeneous for the data .h5 runs wrt. the adaptive's
    # average delta_t (run_non_adaptive_wrt_adaptive.py)
    solution = ProtheroRobinson.solution
    return Problem(
        delta_t_list,
        quantities,
        time_objective,
        stiffness_coef,
        folder_path,
        ProtheroRobinson,
        ProtheroRobinson.solution,
    )


def kaps_problem(epsilon=1.0):
    """initialize the Kaps' problem"""
    delta_t_list = np.array(
        [
            1.0e-2,
            2.0e-2,
            4.0e-2,
            5.0e-2,
            1.0e-1,
        ],
        dtype=float,
    )
    quantities = ["y_1", "y_2"]
    stiffness_coef = {"epsilon": epsilon}
    time_objective = 1.0
    folder_path = pathlib.Path(__file__).parent.parent / "data" / "kaps"
    # "data/kaps" should contain the "adaptive" folder for the data of adaptive .h5 runs
    # (adaptive.py), and the "homogeneous" folder for the homogeneous for the data .h5 runs (homogeneous.py) wrt. the adaptive's
    # average delta_t (avg_homogeneous.py)
    return Problem(
        delta_t_list,
        quantities,
        time_objective,
        stiffness_coef,
        folder_path,
        KapsGroup,
        kaps_solution,
    )
