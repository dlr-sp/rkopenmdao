import argparse
from collections.abc import Callable
from dataclasses import dataclass
import pathlib

import numpy as np
from rkopenmdao.utils.convergence_test_components import kaps_solution, KapsGroup

from ..utils.prothero_robinson_ode import ProtheroRobinson


def parse_problem():
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


@dataclass
class Problem:
    step_sizes: np.ndarray
    quantities: list[str]
    time_objective: float  # in seconds
    stiffness_coef: dict
    folder_path: pathlib.Path
    problem: Callable
    solution: Callable[[float], float | np.ndarray]

    def get_file_path(self, butcher_name, _type):
        """Get the file's path"""
        name = f"{_type}_{butcher_name}".replace(" ", "_").replace(",", "").lower()
        return name, self.folder_path / _type / f"{name}.h5"


def prothero_robinson_problem(_lambda=-1e2):
    delta_t_list = np.array(
        [
            1e-3,
            2e-3,
            4e-3,
            5e-3,
            1e-2,
        ]
    )
    quantities = ["x"]
    time_objective = 10.0  # in seconds
    stiffness_coef = {"lambda_": _lambda}
    folder_path = pathlib.Path(__file__).parent.parent / "data" / "prothero_robinson"
    # "data/prothero_robinson" should contain the "adaptive" folder for the data of adaptive .h5 runs
    # (run_adaptive_problem.py), and the "homogeneous" folder for the homogeneous for the data .h5 runs wrt. the adaptive's
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
    delta_t_list = np.array(
        [
            1e-2,
            2e-2,
            4e-2,
            5e-2,
            1e-1,
        ]
    )
    quantities = ["y_1", "y_2"]
    stiffness_coef = {"epsilon": epsilon}
    time_objective = 1.0  # in seconds
    folder_path = pathlib.Path(__file__).parent.parent / "data" / "kaps"
    return Problem(
        delta_t_list,
        quantities,
        time_objective,
        stiffness_coef,
        folder_path,
        KapsGroup,
        kaps_solution,
    )
