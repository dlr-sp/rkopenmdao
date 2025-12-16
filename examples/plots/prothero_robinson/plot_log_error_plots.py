"""
1. Plots the local error wrt. to analytical solution over delta time,
2. Plots the solution graphs for each scheme.

This Requires to run each of the "run_*" files and store the .h5 files as stated in "File locations" below."
"""
import argparse
import numpy as np
import pathlib

import h5py
import matplotlib.pyplot as plt
from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as second_order_sdirk,
    embedded_second_order_three_stage_esdirk as second_order_esdirk,
    embedded_third_order_three_stage_sdirk as third_order_sdirk,
    embedded_third_order_four_stage_esdirk as third_order_esdirk,
    embedded_fourth_order_four_stage_sdirk as fourth_order_sdirk,
    embedded_fourth_order_five_stage_esdirk as fourth_order_esdirk,
)
from .odes import ProtheroRobinson

parser = argparse.ArgumentParser()
parser.add_argument("--adaptive", action="store_true")
parsed_args = parser.parse_args()
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
        1e-3,
        2e-3,
        4e-3,
        5e-3,
        1e-2,
    ]
)
QUANTITY = "x"
OBJECTIVE_TIME = 10.0  # in seconds
LAMBDA=-1e2

FOLDER_PATH = pathlib.Path(__file__).parent / "data"
# "data" should contain "adaptive" folder for the data of adaptive .h5 runs (run_adaptive_problem.py),
# and "homogeneous" folder for the homogeneous for the data .h5 runs wrt. the adaptive's average delta_t
# (run_non_adaptive_wrt_adaptive.py)



if __name__ == "__main__":

    local_error_data = {}
    for butcher_tableau in BUTCHER_TABLEAUX:
        local_error_data[f"{butcher_tableau.name}"] = [0] * int(OBJECTIVE_TIME)
        for i in range(10):
            local_error_data[butcher_tableau.name][i] = {}
        for dt in DELTA_T_LIST:

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

                for i in range(10):
                    timestep = int((last_step / 10) * (1 + i))
                    local_error_data[butcher_tableau.name][i][str(dt)] = f["Norm"][
                        str(timestep)
                    ][0]

        # ------------------
        # Adaptive
        if parsed_args.adaptive:
            adaptive_error_data = {}

            file_name = f"adaptive_{butcher_tableau.name}"
            file_name = file_name.replace(" ", "_")
            file_name = file_name.replace(",", "")
            file_name = file_name.lower()
            file_path = FOLDER_PATH / "adaptive" /f"{file_name}.h5"
            time = {}
            x = {}

            with h5py.File(
                file_path,
                mode="r",
            ) as f:
                # coefficient values
                group = f["x"]
                for key in group.keys():
                    time.update({int(key): group[key].attrs["time"]})
                    x.update({int(key): group[key][0]})
                    adaptive_error_data[int(key)] = np.abs(
                        ProtheroRobinson.solution(time[int(key)])
                        - x[int(key)]
                    )
            time = dict(sorted(time.items()))
            x = dict(sorted(x.items()))
            adaptive_error_data = dict(sorted(adaptive_error_data.items()))
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
            file_path2 = FOLDER_PATH / "homogeneous" /f"{file_name2}.h5"

            time_nonadapt = {}
            error_nonadapt={}
            with h5py.File(
                    file_path2,
                    mode="r",
            ) as f:
                group = f["x"]

                for key in group.keys():
                    time_nonadapt.update({int(key): group[key].attrs["time"]})
                    error_nonadapt[int(key)] = np.abs(
                        ProtheroRobinson.solution(time_nonadapt[int(key)])
                        - group[key][0]
                    )

            time_nonadapt = dict(sorted(time_nonadapt.items()))
            error_nonadapt = dict(sorted(error_nonadapt.items()))

            # ----------------------------------------------
            # Generate Solution Figure
            fig, axs = plt.subplots(2)
            plt.suptitle(butcher_tableau.name)

            axs[0].set(ylabel=f"$x$")
            axs[0].grid(True)
            axs[0].plot(time.values(), x.values(), "-")

            axs[1].set(xlabel=f"$t$", ylabel=f"$\Delta t$")
            axs[1].grid(True)
            axs[1].plot(time.values(), delta_t, "-")
            axs[1].plot(
                [0, 10], [np.average(delta_t), np.average(delta_t)], "k--", lw=1
            )
            axs[1].text(9.5, np.average(delta_t) * (1.1), r"$\Delta \bar{t}$")
            axs[0].set_xlim(0, OBJECTIVE_TIME)
            axs[1].set_xlim(0, OBJECTIVE_TIME)

            fig.savefig(f"{file_name}_analytical.png")

            # ----------------------------------------------

            # Generate Global Error Figure
            fig, axs = plt.subplots(2)
            plt.suptitle(butcher_tableau.name)

            axs[0].set(ylabel=r"Global error $\epsilon^g$")
            axs[0].grid(True)
            axs[0].plot(
                time.values(),
                adaptive_error_data.values(),
                "-",
                label="Adaptive",
            )
            axs[0].plot(
                time_nonadapt.values(),
                error_nonadapt.values(),
                "--",
                label="Avg. Homogeneous",
            )
            axs[0].set_xticklabels([])
            axs[1].set(xlabel=f"$t$", ylabel=f"$\Delta t$")
            axs[1].grid(True)
            axs[1].plot(time.values(), delta_t, "-")
            axs[1].plot([0, 10], [np.average(delta_t), np.average(delta_t)], "--")
            axs[1].text(9.5, np.average(delta_t) * (1.1), r"$\Delta \bar{t}$")

            for i in range(2):
                axs[i].set_xlim(0, OBJECTIVE_TIME)
            fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.96))
            fig.savefig(f"error_{file_name}.png")

    # PLOT LOCAL ERROR DATA
    for j in range(len(local_error_data[f"{butcher_tableau.name}"])):
        fig, ax = plt.subplots()
        plt.rcParams["mathtext.fontset"] = "cm"
        ax.set_xlabel("$\Delta t$")
        ax.set_ylabel(f"Local error  $\epsilon^l$")

        plt.grid(True)
        for i, scheme in enumerate(BUTCHER_TABLEAUX):
            p = scheme.p

            ax.loglog(
                DELTA_T_LIST,
                local_error_data[scheme.name][j].values(),
                MARKER[i],
                fillstyle="full",
                lw=2,
                color=COLORS[i],
                label=f"{BUTCHERNAMES[i]}",
            )
            # Asymptote set local_error_data[scheme.name][j][%SET_HERE%] a string of the smallest time step size
            # plt.loglog(
            #     DELTA_T_LIST,
            #     (local_error_data[scheme.name][j]["0.0001"] / DELTA_T_LIST[0] ** p)
            #    * (DELTA_T_LIST) ** p,
            #     "k--",
            #     lw=1,
            # )

        ax.set_xlim(DELTA_T_LIST[0], DELTA_T_LIST[-1])
        ax.legend()
        fig.savefig(f"local_error_time_plot_{j+1}.png")