from typing import Optional

from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import integral
from rkopenmdao.error_measurer import ErrorMeasurer, SimpleErrorMeasurer
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.termination_criterion import PredefinedFinalTime
from rkopenmdao.utils.problems import Problem, ProblemConfig

from .constants import PROBLEM, BUTCHER_TABLEAUX


def adaptive_simulation(
    problem: Problem,
    butcher_tableaux: dict,
    error_estimator: Optional[list] = None,
    error_measurer: ErrorMeasurer = SimpleErrorMeasurer(),
) -> None:
    """Execute adaptive integration for each Butcher tableau."""
    if error_estimator is None:
        error_estimator = [integral]
    problem_config = ProblemConfig(
        IntegrationConfig(True, PredefinedFinalTime(problem.time_objective), 0.1),
        error_estimator,
        error_measurer,
    )
    for butcher_tableau in butcher_tableaux.values():
        # modify the config in each iteration, such that corresponding tolerances are used to the Runge-Kutta schemes.
        problem_config.options = {
            "config": ErrorControllerConfig(
                problem.compute_tolerance(butcher_tableau.name)
            )
        }
        # update the file path of the output file
        problem_config.write_file = problem.get_file_path(
            butcher_tableau.name, "adaptive"
        )[1]
        problem.execute(butcher_tableau, problem_config)


if __name__ == "__main__":
    adaptive_simulation(PROBLEM, BUTCHER_TABLEAUX)
