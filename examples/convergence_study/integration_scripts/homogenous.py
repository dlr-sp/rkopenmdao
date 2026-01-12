from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import pseudo
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from ..utils.constants import PROBLEM, BUTCHER_TABLEAUX
from ..utils.run_rk_problem import IntegrationConfig, run_rk_problem


def homogeneous_simulation(problem, butcher_tableaux):
    integration_config = IntegrationConfig(
        TimeTerminationIntegrationControl(0, problem.time_objective, 0.0),
        [pseudo],
        SimpleErrorMeasurer(),
    )

    for butcher_tableau in butcher_tableaux.values():
        for step_size in problem.step_sizes:
            integration_config.integration_control = TimeTerminationIntegrationControl(
                step_size, problem.time_objective, 0.0
            )
            integration_config.write_file = problem.get_file_path(
                butcher_tableau.name, step_size
            )[1]
            run_rk_problem(problem, butcher_tableau, integration_config)


if __name__ == "__main__":
    homogeneous_simulation(PROBLEM, BUTCHER_TABLEAUX)
