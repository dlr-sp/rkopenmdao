import h5py

from .constants import PROBLEM, BUTCHER_TABLEAUX
from rkopenmdao.error_controller import ErrorControllerConfig
from rkopenmdao.error_controllers import pseudo
from rkopenmdao.error_measurer import SimpleErrorMeasurer, ImprovedErrorMeasurer
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.utils.problems import Problem, IntegrationConfig


def count_keys(path: str):
    """Count the number of keys in a HDF5 file; this is equal to the number of steps."""
    with h5py.File(path, "r") as f:
        group_name = list(f.keys())[0]
        return len(f[group_name].keys())


def avg_homogeneous_simulation(problem: Problem, butcher_tableaux: dict):
    """Execute homogeneous integration for each Butcher tableau with the average step size of the adaptive runs."""
    integration_config = IntegrationConfig(
        TimeTerminationIntegrationControl(0, PROBLEM.time_objective, 0.0),
        [pseudo],
        SimpleErrorMeasurer(),
    )
    for butcher_tableau in butcher_tableaux.values():
        # count the number of steps
        steps = count_keys(
            str(problem.get_file_path(butcher_tableau.name, "adaptive")[1])
        )
        # compute the step size by dividing the total time by the number of steps
        step_size = problem.time_objective / steps
        print(f"{butcher_tableau.name}: {step_size}")
        # update the integration control with the new step size
        integration_config.integration_control = TimeTerminationIntegrationControl(
            step_size, problem.time_objective, 0.0
        )
        # update the file path of the output file
        integration_config.write_file = problem.get_file_path(
            butcher_tableau.name, "avg_homogeneous"
        )[1]
        problem.execute(butcher_tableau, integration_config)


if __name__ == "__main__":
    avg_homogeneous_simulation(PROBLEM, BUTCHER_TABLEAUX)
