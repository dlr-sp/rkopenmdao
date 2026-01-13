from typing import Type

from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import pid, integral, h0_211
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from ..utils.constants import PROBLEM, BUTCHER_TABLEAUX
from ..utils.problems import Problem, IntegrationConfig


def adaptive_simulation(problem: Problem, butcher_tableaux: dict) -> None:
    """Execute adaptive integration for each Butcher tableau."""
    integration_config = IntegrationConfig(
        TimeTerminationIntegrationControl(0.1, problem.time_objective, 0.0),
        [integral],  # amendable
        SimpleErrorMeasurer(),  # changeable to ImprovedErrorMeasurer
    )
    for butcher_tableau in butcher_tableaux.values():
        # modify the config in each iteration, such that corresponding tolerances are used to the Runge-Kutta schemes.
        integration_config.options = {
            "config": ErrorControllerConfig(
                problem.compute_tolerance(butcher_tableau.name)
            )
        }
        # update the file path of the output file
        integration_config.write_file = problem.get_file_path(
            butcher_tableau.name, "adaptive"
        )[1]
        problem.execute(butcher_tableau, integration_config)


if __name__ == "__main__":
    adaptive_simulation(PROBLEM, BUTCHER_TABLEAUX)
