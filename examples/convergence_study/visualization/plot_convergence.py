"""
1. Plots the convergence rate of each scheme.

This requires running the "./integration_scripts/homogenous.py" file."
"""

import matplotlib.pyplot as plt
from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.file_writer import read_last_local_error
from .constants import (
    PROBLEM,
    BUTCHER_TABLEAUX,
    MARKER,
    COLORS,
)
from rkopenmdao.src.rkopenmdao.utils.problems import Problem, generate_path


def extract_local_error_data(butcher_tableau: ButcherTableau, problem: Problem) -> dict:
    """Extract the local error data from a hdf5 file."""
    # initialize dictionary to store local error data
    butcher_error_data = {}
    # iterate over the step sizes
    for step_size in problem.step_sizes:
        file_name, file_path = problem.get_file_path(butcher_tableau.name, step_size)
        butcher_error_data[str(step_size)] = read_last_local_error(
            file_path, problem.time_objective, step_size
        )
    return butcher_error_data


def generate_convergence_graph(
    local_error_data: dict, butcher_tableaux: dict, problem: Problem
):
    """Plot local error data for each scheme."""
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"Local error measure $\epsilon^l$")
    plt.grid(True)
    for i, scheme in enumerate(butcher_tableaux.values()):
        ax.loglog(
            problem.step_sizes,
            local_error_data[scheme.name].values(),
            MARKER[i],
            fillstyle="full",
            lw=2,
            color=COLORS[i],
            label=f"{list(butcher_tableaux.keys())[i]}",
        )
    ax.set_xlim(problem.step_sizes[0], problem.step_sizes[-1])
    ax.legend()

    save_file = generate_path(
        str(problem.folder_path / "plots" / "local_error_time_plot.png")
    )
    fig.savefig(str(save_file))


if __name__ == "__main__":
    local_error_data = {}
    for butcher_tableau in BUTCHER_TABLEAUX.values():
        local_error_data[butcher_tableau.name] = extract_local_error_data(
            butcher_tableau, PROBLEM
        )
    generate_convergence_graph(local_error_data, BUTCHER_TABLEAUX, PROBLEM)
