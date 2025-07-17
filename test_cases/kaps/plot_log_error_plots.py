"""
1. Plots the local error wrt. to analytical solution over delta time,
2. Plots the solution graphs for each scheme.
3. Plots the global error (Adaptive Vs. Homogeneous) of each scheme.

This Requires to run each of the "run_*" files and store the .h5 files as stated in "File locations" below."
"""

import argparse
import pathlib
import numpy as np

import h5py
import matplotlib.pyplot as plt
from matplotlib import ticker
from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as second_order_sdirk,
    embedded_second_order_three_stage_esdirk as second_order_esdirk,
    embedded_third_order_three_stage_sdirk as third_order_sdirk,
    embedded_third_order_four_stage_esdirk as third_order_esdirk,
    embedded_fourth_order_four_stage_sdirk as fourth_order_sdirk,
    embedded_fourth_order_five_stage_esdirk as fourth_order_esdirk,
)
from rkopenmdao.utils.convergence_test_components import kaps_solution

parser = argparse.ArgumentParser()
parser.add_argument("--adaptive", action="store_true")
parsed_args = parser.parse_args()

# Constants
MARKER = ["-o", "-X", "-P", "-D", "-v", "-H"]
COLORS = ["indigo", "indianred", "seagreen", "slategray", "orange", "lightskyblue"]
BUTCHERNAMES = ["SDIRK2", "ESDIRK2", "SDIRK3", "ESDIRK3", "SDIRK4", "ESDIRK4"]
BUTCHER_TABLEAUX = [
    second_order_sdirk,
    second_order_esdirk,
    third_order_sdirk,
    third_order_esdirk,
    fourth_order_sdirk,
    fourth_order_esdirk,
]
DELTA_T_LIST = np.array(
    [
        1e-2,
        2e-2,
        4e-2,
        5e-2,
        1e-1,
    ]
)

OBJECTIVE_TIME = 1.0
QUANTITY = ["y_1", "y_2"]

# Files location
# ---------------
FOLDER = "eps1"  # e.g. eps1 or eps1e-3
FOLDER_ADAPT = "l2_norm" # example for names: l2_norm, inf_norm, y_1_excluded or y_2_excluded
# Path to the homogeneous .h5 files for constructing the local error [eps^l] plots
FOLDER_PATH = pathlib.Path(__file__).parent / "data" / FOLDER
# FOLDER_ADAPT should contain "adaptive" folder for the data of adaptive .h5 runs (run_adaptive_problem.py),
# and "homogeneous" folder for the homogeneous for the data .h5 runs wrt. the adaptive's average delta_t
# (run_non_adaptive_wrt_adaptive.py)
FOLDER_PATH_ADAPT = pathlib.Path(__file__).parent / "data" / FOLDER / FOLDER_ADAPT
# ---------------

if __name__ == "__main__":

    local_error_data = {}
    # ------------------
    # Homogeneous
    for butcher_tableau in BUTCHER_TABLEAUX:
        local_error_data[f"{butcher_tableau.name}"] = {}
        for dt in DELTA_T_LIST:
            local_error_data[f"{butcher_tableau.name}"][str(dt)] = []

            file_name = f"data_{dt:.0E}_{butcher_tableau.name}"
            file_name = file_name.replace(" ", "_")
            file_name = file_name.replace(",", "")
            file_name = file_name.lower()
            file_path = FOLDER_PATH / f"{file_name}.h5"
            with h5py.File(
                file_path,
                mode="r",
            ) as f:
                last_step = int(OBJECTIVE_TIME / dt)
                local_error_data[butcher_tableau.name][str(dt)].append(
                    f["Norm"][str(last_step)][0]
                )
        # ------------------
        # Adaptive
        if parsed_args.adaptive:
            adaptive_error_data = {}

            file_name = f"adaptive_{butcher_tableau.name}"
            file_name = file_name.replace(" ", "_")
            file_name = file_name.replace(",", "")
            file_name = file_name.lower()
            file_path = FOLDER_PATH_ADAPT / "adaptive" /f"{file_name}.h5"
            time = {}
            solutions = {}

            with h5py.File(
                file_path,
                mode="r",
            ) as f:
                # coefficient values
                group = f[QUANTITY[0]]
                for key in group.keys():
                    time[int(key)] = group[key].attrs["time"]
                for i, quantity in enumerate(QUANTITY):
                    solutions[quantity] = {}
                    adaptive_error_data[quantity] = {}
                    group = f[quantity]
                    for key in group.keys():
                        solutions[quantity][int(key)] = group[key][0]
                        adaptive_error_data[quantity][int(key)] = np.abs(
                            kaps_solution(time[int(key)])[i]
                            - solutions[quantity][int(key)]
                        )

            time = dict(sorted(time.items()))
            for quantity in QUANTITY:
                solutions[quantity] = dict(sorted(solutions[quantity].items()))
                adaptive_error_data[quantity] = dict(
                    sorted(adaptive_error_data[quantity].items())
                )

            delta_t = [0] * len(time)
            for i in range(len(time) - 1):
                delta_t[i] = time[i + 1] - time[i]
            delta_t[i + 1] = delta_t[i]

            # -----------------
            # Homogeneous

            file_name2 = f"homogeneous_{butcher_tableau.name}"
            file_name2 = file_name2.replace(" ", "_")
            file_name2 = file_name2.replace(",", "")
            file_name2 = file_name2.lower()
            file_path = FOLDER_PATH_ADAPT / "homogeneous" /f"{file_name2}.h5"

            time_non_adaptive = {}
            non_adaptive_error_data = {}
            with h5py.File(
                file_path,
                mode="r",
            ) as f:
                # coefficient values
                group = f[QUANTITY[0]]
                for key in group.keys():
                    time_non_adaptive[int(key)] = group[key].attrs["time"]
                for i, quantity in enumerate(QUANTITY):
                    non_adaptive_error_data[quantity] = {}
                    group = f[quantity]
                    for key in group.keys():
                        non_adaptive_error_data[quantity][int(key)] = np.abs(
                            kaps_solution(time_non_adaptive[int(key)])[i]
                            - group[key][0]
                        )
            time_non_adaptive = dict(sorted(time_non_adaptive.items()))
            for quantity in QUANTITY:
                non_adaptive_error_data[quantity] = dict(
                    sorted(non_adaptive_error_data[quantity].items())
                )

            # ----------------------------------------------
            # Generate Solution Figure

            fig, axs = plt.subplots(3)
            plt.suptitle(butcher_tableau.name)

            axs[0].set(ylabel=f"$y_1$")
            axs[0].grid(True)
            axs[0].plot(time.values(), solutions[QUANTITY[0]].values(), "-")
            axs[1].set(ylabel=f"$y_2$")
            axs[1].grid(True)
            axs[1].plot(time.values(), solutions[QUANTITY[1]].values(), "-")
            axs[2].set(xlabel=f"$t$", ylabel=f"$\Delta t$")
            axs[2].grid(True)
            axs[2].plot(time.values(), delta_t, "-")
            axs[2].plot(
                [0, 10], [np.average(delta_t), np.average(delta_t)], "k--", lw=1
            )
            axs[2].text(0.01, np.average(delta_t) * (1.01), r"$\Delta \bar{t}$")

            for i in range(3):
                axs[i].set_xlim(0, OBJECTIVE_TIME)

            fig.savefig(f"{file_name}_analytical.png")

            # ----------------------------------------------
            # Generate Global Error Comparison Figure

            fig, axs = plt.subplots(3)
            plt.suptitle(butcher_tableau.name)
            formatter = ticker.ScalarFormatter(useMathText=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))
            axs[0].yaxis.set_major_formatter(formatter)
            axs[0].set(ylabel=r"Global error $\epsilon^g_{y_1}$")
            axs[0].grid(True)
            axs[0].plot(
                time.values(),
                adaptive_error_data[QUANTITY[0]].values(),
                "-",
                label="Adaptive",
            )

            axs[0].plot(
                time_non_adaptive.values(),
                non_adaptive_error_data[QUANTITY[0]].values(),
                "--",
                label="Avg. Homogeneous",
            )
            axs[0].set_xticklabels([])
            axs[1].set(ylabel=r"Global error $\epsilon^g_{y_2}$")
            axs[1].grid(True)
            axs[1].plot(
                time.values(),
                adaptive_error_data[QUANTITY[1]].values(),
                "-",
            )

            axs[1].plot(
                time_non_adaptive.values(),
                non_adaptive_error_data[QUANTITY[1]].values(),
                "--",
            )
            axs[1].set_xticklabels([])

            axs[2].set(xlabel=f"$t$", ylabel=f"$\Delta t$")
            axs[2].grid(True)
            axs[2].plot(time.values(), delta_t, "-")
            axs[2].plot([0, 10], [np.average(delta_t), np.average(delta_t)], "--")
            axs[2].text(0.01, np.average(delta_t) * (1.01), r"$\Delta \bar{t}$")

            for i in range(3):
                axs[i].set_xlim(0, OBJECTIVE_TIME)

            fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.96))
            fig.savefig(f"{file_name}_analytical_error.png")

    # PLOT LOCAL ERROR DATA
    fig, ax = plt.subplots()

    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$L_2$ norm local error $\epsilon^l$")

    plt.grid(True)
    for i, scheme in enumerate(BUTCHER_TABLEAUX):
        p = scheme.p

        ax.loglog(
            DELTA_T_LIST,
            local_error_data[scheme.name].values(),
            MARKER[i],
            fillstyle="full",
            lw=2,
            color=COLORS[i],
            label=f"{BUTCHERNAMES[i]}",
        )
       # Asymptote set local_error_data[scheme.name][j][%SET_HERE%] a string of the smallest time step size
       # plt.loglog(
       #     DELTA_T_LIST,
       #     (local_error_data[scheme.name]["0.01"] / DELTA_T_LIST[0] ** p)
       #     * (DELTA_T_LIST) ** p,
       #    "k--",
       #     lw=1,
       # )
    ax.set_xlim(DELTA_T_LIST[0], DELTA_T_LIST[-1])
    ax.legend()
    fig.savefig(f"local_error_time_plot.png")
