import h5py

from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import pseudo
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from ..utils.constants import PROBLEM, BUTCHER_TABLEAUX
from ..utils.run_rk_problem import IntegrationConfig, run_rk_problem


def count_keys(path):
    with h5py.File(path, "r") as f:
        group_name = list(f.keys())[0]
        return len(f[group_name].keys())


def avg_homogeneous_simulation(problem, butcher_tableaux):
    integration_config = IntegrationConfig(
        TimeTerminationIntegrationControl(0, PROBLEM.time_objective, 0.0),
        [pseudo],
        SimpleErrorMeasurer(),
    )
    for butcher_tableau in butcher_tableaux.values():
        steps = count_keys(
            str(problem.get_file_path(butcher_tableau.name, "adaptive")[1])
        )
        step_size = problem.time_objective / steps
        print(f"{butcher_tableau.name}: {step_size}")
        integration_config.integration_control = TimeTerminationIntegrationControl(
            step_size, problem.time_objective, 0.0
        )
        integration_config.write_file = problem.get_file_path(
            butcher_tableau.name, "homogeneous"
        )[1]
        run_rk_problem(problem, butcher_tableau, integration_config)


if __name__ == "__main__":
    avg_homogeneous_simulation(PROBLEM, BUTCHER_TABLEAUX)
