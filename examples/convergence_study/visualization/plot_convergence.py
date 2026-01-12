"""
1. Plots the convergence rate of each scheme.

This requires running the "run_homogenous_problem" file."
"""

import h5py
import matplotlib.pyplot as plt

from ..utils.constants import (
    PROBLEM,
    BUTCHER_TABLEAUX,
    MARKER,
    COLORS,
)
from ..utils.run_rk_problem import generate_path
from rkopenmdao.file_writer import read_last_local_error


def plot_convergence(butcher_tableaux, problem):
    local_error_data = {}
    for butcher_tableau in butcher_tableaux.values():
        local_error_data[f"{butcher_tableau.name}"] = {}
        for step_size in problem.step_sizes:
            file_name, file_path = problem.get_file_path(
                butcher_tableau.name, step_size
            )
            local_error_data[f"{butcher_tableau.name}"][str(step_size)] = (
                read_last_local_error(file_path, problem.time_objective, step_size)
            )

    # PLOT LOCAL ERROR DATA

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"Local error measure $\epsilon^l$")
    plt.grid(True)
    for i, scheme in enumerate(butcher_tableaux.values()):
        p = scheme.p
        ax.loglog(
            problem.step_sizes,
            local_error_data[scheme.name].values(),
            MARKER[i],
            fillstyle="full",
            lw=2,
            color=COLORS[i],
            label=f"{butcher_tableaux.keys()[i]}",
        )
    ax.set_xlim(problem.step_sizes[0], problem.step_sizes[-1])
    ax.legend()

    save_file = generate_path(
        str(problem.folder_path / "plots" / "local_error_time_plot.png")
    )
    fig.savefig(str(save_file))


if __name__ == "__main__":
    plot_convergence(BUTCHER_TABLEAUX, PROBLEM)
