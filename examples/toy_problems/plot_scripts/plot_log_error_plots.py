"""
1. Plots the local error wrt. to analytical solution over delta time,
2. Plots the solution graphs for each scheme.

This requires running each of the "run_*" files."
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from ..utils.constants import PROBLEM, BUTCHER_TABLEAUX
from ..utils.run_rk_problem import generate_path
from rkopenmdao.file_writer import read_hdf5_file


@dataclass
class ResultData:
    error: dict
    result: dict
    time: dict
    delta_t: list = field(default_factory=list)


def sort_dicts(time, error, result, quantities):
    time = dict(sorted(time.items()))
    # Takes the time dictionary, sorts its items (by key),
    # and converts the result back into a dictionary
    dt = [0.0] * len(time)
    for i in range(
        len(time) - 1
    ):  # Calculates the difference between consecutive time points.
        dt[i] = time[i + 1] - time[i]
    dt[-1] = dt[-2]  # match the second-to-last element to maintain the list length
    for q in quantities:
        result[q] = dict(sorted(result[q].items()))
        error[q] = dict(sorted(error[q].items()))
    return (
        time,
        error,
        result,
        dt,
    )


def extract_solution_per_butcher_table(butcher_tableau):
    # ------------------
    # Adaptive
    data_name, file_path = PROBLEM.get_file_path(butcher_tableau.name, "adaptive")
    time_adaptive, error_data_adaptive, results_adaptive = read_hdf5_file(
        file_path, PROBLEM.quantities, PROBLEM.solution
    )

    # -----------------
    # Homogeneous
    _, file_path = PROBLEM.get_file_path(butcher_tableau.name, "homogeneous")
    time_homogeneous, error_data_homogeneous, results_homogenous = read_hdf5_file(
        file_path, PROBLEM.quantities, PROBLEM.solution
    )

    # -----------------
    # Sort
    time_adaptive, error_data_adaptive, results_adaptive, delta_t = sort_dicts(
        time_adaptive, error_data_adaptive, results_adaptive, PROBLEM.quantities
    )
    time_homogeneous, error_data_homogeneous, results_homogenous, _ = sort_dicts(
        time_homogeneous,
        error_data_homogeneous,
        results_homogenous,
        PROBLEM.quantities,
    )

    # -----------------
    # Create Dicts
    adaptive_data = ResultData(
        error_data_adaptive, results_adaptive, time_adaptive, delta_t
    )
    homogeneous_data = ResultData(
        error_data_homogeneous, results_homogenous, time_homogeneous
    )
    return data_name, adaptive_data, homogeneous_data


def decorate_last_time_box(axs, adaptive_data):
    axs[-1].set(xlabel=f"$t$", ylabel=f"$\Delta t$")
    axs[-1].grid(True)
    axs[-1].plot(adaptive_data.time.values(), adaptive_data.delta_t, "-")
    axs[-1].plot(
        [0, PROBLEM.time_objective],
        [np.average(adaptive_data.delta_t), np.average(adaptive_data.delta_t)],
        "k--",
        lw=1,
    )
    axs[-1].text(
        PROBLEM.time_objective * 0.95,
        np.average(adaptive_data.delta_t) * 1.01,
        r"$\Delta \bar{t}$",
    )
    axs[-1].set_xlim(0, PROBLEM.time_objective)


def generate_solution_figure(
    butcher_tableau, adaptive_data: ResultData, data_name="sol_data"
):
    """Generate solution figure for a given scheme."""
    fig, axs = plt.subplots(len(PROBLEM.quantities) + 1)
    plt.suptitle(f"{butcher_tableau.name} analytical solution")

    # For each quantity, plot the computed solution
    for index, quantity in enumerate(PROBLEM.quantities):
        axs[index].set(ylabel=f"${quantity}$")
        axs[index].grid(True)
        axs[index].plot(
            adaptive_data.time.values(), adaptive_data.result[quantity].values(), "-"
        )
        axs[index].set_xlim(0, PROBLEM.time_objective)

    # Plot in the last box the development of time step size over time
    decorate_last_time_box(axs, adaptive_data)

    # Save figure
    save_file = generate_path(
        str(PROBLEM.folder_path / "plots" / f"analytical_solution_{data_name}.png")
    )

    fig.savefig(str(save_file))


def generate_global_error_figure(
    butcher_tableau,
    adaptive_data: ResultData,
    homogeneous_data: ResultData,
    data_name="err_data",
):
    fig, axs = plt.subplots(len(PROBLEM.quantities) + 1)
    plt.suptitle(f"{butcher_tableau.name} global error")

    formatter = ticker.ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    for index, quantity in enumerate(PROBLEM.quantities):
        axs[index].set(ylabel=f"Global error $\epsilon^g_{quantity.replace('_','')}$")
        axs[index].grid(True)
        axs[index].plot(
            adaptive_data.time.values(),
            adaptive_data.error[quantity].values(),
            "-",
            label="Adaptive Method",
        )
        axs[index].plot(
            homogeneous_data.time.values(),
            homogeneous_data.error[quantity].values(),
            "--",
            label="Homogeneous Method",
        )
        axs[index].set_xlim(0, PROBLEM.time_objective)
        axs[index].set_xticklabels([])

        # Plot in the last box the development of time step size over time
        decorate_last_time_box(axs, adaptive_data)

        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.96),
        )
        save_file = generate_path(
            str(PROBLEM.folder_path / "plots" / f"global_error_{data_name}.png")
        )
        fig.savefig(str(save_file))


if __name__ == "__main__":
    for butcher_tableau in BUTCHER_TABLEAUX.values():
        data_name, adaptive_data, homogeneous_data = extract_solution_per_butcher_table(
            butcher_tableau
        )
        # ----------------------------------------------
        # Generate Solution Figure
        generate_solution_figure(butcher_tableau, adaptive_data, data_name)
        # ----------------------------------------------
        # Generate Global Error Figure
        generate_global_error_figure(
            butcher_tableau, adaptive_data, homogeneous_data, data_name
        )
