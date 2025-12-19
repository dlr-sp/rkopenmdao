"""
1. Plots the local error wrt. to analytical solution over delta time,
2. Plots the solution graphs for each scheme.

This requires running each of the "run_*" files and store the .h5 files as stated in "File locations" below."
"""

import numpy as np

import h5py
import matplotlib.pyplot as plt
from matplotlib import ticker

from ..utils.constants import PROBLEM, BUTCHER_TABLEAUX
from ..utils.rk_setup import generate_path


def h5py_read(file, quantities, solution):
    # Initialize dictionaries
    time = {}
    error_data = {}
    result = {}
    # Open the HDF5 file in read-only mode
    with h5py.File(
        file,
        mode="r",
    ) as f:
        # coefficient values
        group = f[quantities[0]]
        # Extract time metadata
        for key in group.keys():
            time[int(key)] = group[key].attrs["time"]
        # Extract solution and compute Error wrt. analytical solution
        for i, quan in enumerate(quantities):
            group = f[quan]
            error_data[quan] = {}  # initialize error data for each quantity
            result[quan] = {}
            for key in group.keys():
                result[quan][int(key)] = group[key][0]
                if len(quantities) > 1:
                    error_data[quan][int(key)] = np.abs(
                        solution(time[int(key)])[i] - result[quan][int(key)]
                    )
                else:
                    error_data[quan][int(key)] = np.abs(
                        solution(time[int(key)]) - result[quan][int(key)]
                    )
    return time, error_data, result


def sort_dicts(time, error, result, quantities):
    time = dict(sorted(time.items()))
    # Takes the time dictionary, sorts its items (by key),
    # and converts the result back into a dictionary
    dt = [0] * len(time)
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


if __name__ == "__main__":
    for butcher_tableau in BUTCHER_TABLEAUX:
        # ------------------
        # Adaptive
        data_name, file_path = PROBLEM.get_file_path(butcher_tableau.name, "adaptive")
        time_adaptive, error_data_adaptive, results_adaptive = h5py_read(
            file_path, PROBLEM.quantity, PROBLEM.solution
        )
        time_adaptive, error_data_adaptive, results_adaptive, delta_t = sort_dicts(
            time_adaptive, error_data_adaptive, results_adaptive, PROBLEM.quantity
        )

        # -----------------
        # Homogeneous

        _, file_path = PROBLEM.get_file_path(butcher_tableau.name, "homogeneous")
        time_homogeneous, error_data_homogeneous, results_homogenous = h5py_read(
            file_path, PROBLEM.quantity, PROBLEM.solution
        )

        # Sort
        time_homogeneous, error_data_homogeneous, results_homogenous, _ = sort_dicts(
            time_homogeneous,
            error_data_homogeneous,
            results_homogenous,
            PROBLEM.quantity,
        )

        # ----------------------------------------------
        # Generate Solution Figure
        fig, axs = plt.subplots(len(PROBLEM.quantity) + 1)
        plt.suptitle(f"{butcher_tableau.name} analytical solution")

        for index, quantity in enumerate(PROBLEM.quantity):
            axs[index].set(ylabel=f"${quantity}$")
            axs[index].grid(True)
            axs[index].plot(
                time_adaptive.values(), results_adaptive[quantity].values(), "-"
            )
            axs[index].set_xlim(0, PROBLEM.time_objective)

        axs[-1].set(xlabel=f"$t$", ylabel=f"$\Delta t$")
        axs[-1].grid(True)
        axs[-1].plot(time_adaptive.values(), delta_t, "-")
        axs[-1].plot(
            [0, PROBLEM.time_objective],
            [np.average(delta_t), np.average(delta_t)],
            "k--",
            lw=1,
        )
        axs[-1].text(
            PROBLEM.time_objective * 0.95,
            np.average(delta_t) * 1.01,
            r"$\Delta \bar{t}$",
        )
        axs[-1].set_xlim(0, PROBLEM.time_objective)
        save_file = generate_path(
            str(PROBLEM.folder_path / "plots" / f"analytical_solution_{data_name}.png")
        )

        fig.savefig(str(save_file))

        # ----------------------------------------------

        # Generate Global Error Figure
        fig, axs = plt.subplots(len(PROBLEM.quantity) + 1)
        plt.suptitle(f"{butcher_tableau.name} global error")

        formatter = ticker.ScalarFormatter(useMathText=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        for index, quantity in enumerate(PROBLEM.quantity):
            print(error_data_adaptive[quantity].values())
            axs[index].set(
                ylabel=f"Global error $\epsilon^g_{quantity.replace('_','')}$"
            )
            axs[index].grid(True)
            axs[index].plot(
                time_adaptive.values(),
                error_data_adaptive[quantity].values(),
                "-",
                label="Adaptive Method",
            )
            axs[index].plot(
                time_homogeneous.values(),
                error_data_homogeneous[quantity].values(),
                "--",
                label="Homogeneous Method",
            )
            axs[index].set_xlim(0, PROBLEM.time_objective)
            axs[index].set_xticklabels([])

        axs[-1].set(xlabel=f"$t$", ylabel=f"$\Delta t$")
        axs[-1].grid(True)
        axs[-1].plot(time_adaptive.values(), delta_t, "-")
        axs[-1].plot(
            [0, PROBLEM.time_objective],
            [np.average(delta_t), np.average(delta_t)],
            "--",
        )
        axs[-1].text(
            PROBLEM.time_objective * 0.95,
            np.average(delta_t) * 1.01,
            r"$\Delta \bar{t}$",
        )

        axs[-1].set_xlim(0, PROBLEM.time_objective)
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
        fig.savefig((save_file))
