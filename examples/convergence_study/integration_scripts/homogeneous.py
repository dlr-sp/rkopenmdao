from .constants import PROBLEM, BUTCHER_TABLEAUX
from rkopenmdao.error_controllers import pseudo
from rkopenmdao.error_measurer import SimpleErrorMeasurer
from rkopenmdao.integration_config import IntegrationConfig
from rkopenmdao.termination_criterion import PredefinedFinalTime
from rkopenmdao.utils.problems import Problem, ProblemConfig


def homogeneous_simulation(problem: Problem, butcher_tableaux: dict):
    """Execute homogenous integration for each Butcher tableau."""
    problem_config = ProblemConfig(
        IntegrationConfig(True, PredefinedFinalTime(problem.time_objective), 0.0),
        [pseudo],
        SimpleErrorMeasurer(),
    )
    # run each Runge-Kutta scheme for each step size
    for butcher_tableau in butcher_tableaux.values():
        for step_size in problem.step_sizes:
            print(f"{butcher_tableau.name}: {step_size}")
            # update the integration control with the new step size
            problem_config.integration_config.initial_step_size = step_size
            # update the file path of the output file
            problem_config.write_file = problem.get_file_path(
                butcher_tableau.name, step_size
            )[1]
            problem.execute(butcher_tableau, problem_config)


if __name__ == "__main__":
    homogeneous_simulation(PROBLEM, BUTCHER_TABLEAUX)
