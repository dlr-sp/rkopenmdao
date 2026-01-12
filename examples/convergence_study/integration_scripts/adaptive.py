from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import pid, integral, h0_211
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from ..utils.constants import PROBLEM, BUTCHER_TABLEAUX
from ..utils.run_rk_problem import IntegrationConfig, run_rk_problem


def adaptive_simulation(problem, butcher_tableaux):
    integration_config = IntegrationConfig(
        TimeTerminationIntegrationControl(0.1, problem.time_objective, 0.0),
        [integral],
        SimpleErrorMeasurer(),
    )
    for butcher_tableau in butcher_tableaux.values():
        integration_config.config = {
            "config": ErrorControllerConfig(
                problem.compute_tolerance(butcher_tableau.name)
            )
        }
        integration_config.write_file = problem.get_file_path(
            butcher_tableau.name, "adaptive"
        )[1]
        run_rk_problem(problem, butcher_tableau, integration_config)


if __name__ == "__main__":
    adaptive_simulation(PROBLEM, BUTCHER_TABLEAUX)
