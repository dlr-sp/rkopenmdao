from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import pseudo
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from ..utils.constants import PROBLEM, BUTCHER_TABLEAUX
from ..utils.problems import Problem, IntegrationConfig


def homogeneous_simulation(problem: Problem, butcher_tableaux: dict):
    """Execute homogenous integration for each Butcher tableau."""
    integration_config = IntegrationConfig(
        TimeTerminationIntegrationControl(0, problem.time_objective, 0.0),
        [pseudo],
        SimpleErrorMeasurer(),
    )
    # run each Runge-Kutta scheme for each step size
    for butcher_tableau in butcher_tableaux.values():
        for step_size in problem.step_sizes:
            print(f"{butcher_tableau.name}: {step_size}")
            # update the integration control with the new step size
            integration_config.integration_control = TimeTerminationIntegrationControl(
                step_size, problem.time_objective, 0.0
            )
            # update the file path of the output file
            integration_config.write_file = problem.get_file_path(
                butcher_tableau.name, step_size
            )[1]
            problem.execute(butcher_tableau, integration_config)


if __name__ == "__main__":
    homogeneous_simulation(PROBLEM, BUTCHER_TABLEAUX)
