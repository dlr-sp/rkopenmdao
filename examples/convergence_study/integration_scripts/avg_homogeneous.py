import h5py

from .constants import PROBLEM, BUTCHER_TABLEAUX
from rkopenmdao.error_controllers import pseudo
from rkopenmdao.error_measurer import SimpleErrorMeasurer
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.termination_criterion import PredefinedFinalTime
from rkopenmdao.utils.problems import Problem, ProblemConfig


def count_keys(path: str):
    """Count the number of keys in a HDF5 file; this is equal to the number of steps."""
    with h5py.File(path, "r") as f:
        group_name = list(f.keys())[0]
        return len(f[group_name].keys())


def avg_homogeneous_simulation(problem: Problem, butcher_tableaux: dict):
    """Execute homogeneous integration for each Butcher tableau with the average step size of the adaptive runs."""
    problem_config = ProblemConfig(
        IntegrationConfig(True, PredefinedFinalTime(problem.time_objective), 0.0),
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
        problem_config.integration_config.initial_step_size = step_size
        # update the file path of the output file
        problem_config.write_file = problem.get_file_path(
            butcher_tableau.name, "avg_homogeneous"
        )[1]
        problem.execute(butcher_tableau, problem_config)


if __name__ == "__main__":
    avg_homogeneous_simulation(PROBLEM, BUTCHER_TABLEAUX)
