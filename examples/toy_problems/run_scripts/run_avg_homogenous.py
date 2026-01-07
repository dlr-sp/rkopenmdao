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


if __name__ == "__main__":
    integration_config = IntegrationConfig(
        TimeTerminationIntegrationControl(0, PROBLEM.time_objective, 0.0),
        [pseudo],
        SimpleErrorMeasurer(),
        "",
        {},
    )
    for butcher_tableau in BUTCHER_TABLEAUX.values():
        steps = count_keys(
            str(PROBLEM.get_file_path(butcher_tableau.name, "adaptive")[1])
        )
        step_size = PROBLEM.time_objective / steps
        print(f"{butcher_tableau.name}: {step_size}")
        integration_config.integration_control = TimeTerminationIntegrationControl(
            step_size, PROBLEM.time_objective, 0.0
        )
        integration_config.write_file = PROBLEM.get_file_path(
            butcher_tableau.name, "homogeneous"
        )[1]
        run_rk_problem(PROBLEM, butcher_tableau, integration_config)
