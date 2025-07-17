"""Plots the error wrt. to analytical solution over delta time"""

import argparse
import pathlib

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np

from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as second_order_sdirk,
    embedded_second_order_three_stage_esdirk as second_order_esdirk,
    embedded_third_order_three_stage_sdirk as third_order_sdirk,
    embedded_third_order_four_stage_esdirk as third_order_esdirk,
    embedded_fourth_order_four_stage_sdirk as fourth_order_sdirk,
    embedded_fourth_order_five_stage_esdirk as fourth_order_esdirk,
)
from rkopenmdao.utils.convergence_test_components import kaps_solution
import plot_environment as penv

plt.rcParams["mathtext.fontset"] = "cm"

parser = argparse.ArgumentParser()
parser.add_argument("--base_name", default="data", type=str)
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
        2e-2,
        4e-2,
        5e-2,
        1e-1,
        2e-1,
        4e-1,
        5e-1,
        1,
    ]
)

FOLDER = "eps1"
FOLDER_ADAPT = "l2norm"
FOLDER_PATH = pathlib.Path(__file__).parent / "data" / FOLDER
FOLDER_PATH_ADAPT = pathlib.Path(__file__).parent / "data" / FOLDER / FOLDER_ADAPT
OBJECTIVE_TIME = 1.0  # in seconds
QUANTITY = ["y_1", "y_2"]
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if __name__ == "__main__":

    error_data = {}
    local_error_data = {}
    adaptive_error_data = {}
    average_dt_data = {}

    for butcher_tableau in BUTCHER_TABLEAUX:
        error_data[f"{butcher_tableau.name}"] = {}
        local_error_data[f"{butcher_tableau.name}"] = {}
        adaptive_error_data[f"{butcher_tableau.name}"] = {}
        average_dt_data[f"{butcher_tableau.name}"] = {}
        for dt in DELTA_T_LIST:
            error_data[f"{butcher_tableau.name}"][str(dt)] = {}
            local_error_data[f"{butcher_tableau.name}"][str(dt)] = []

            file_name = f"{parsed_args.base_name}_{dt:.0E}_{butcher_tableau.name}"
            file_name = file_name.replace(" ", "_")
            file_name = file_name.replace(",", "")
            file_name = file_name.lower()
            file_path = FOLDER_PATH / f"{file_name}.h5"
            with h5py.File(
                file_path,
                mode="r",
                driver="mpio",
                comm=comm,
            ) as f:
                last_step = int(OBJECTIVE_TIME / dt)
                for i, quantity in enumerate(QUANTITY):
                    error_data[butcher_tableau.name][str(dt)][quantity] = (
                        kaps_solution(OBJECTIVE_TIME)[i]
                        - f[quantity][str(last_step)][0]
                    )

                local_error_data[butcher_tableau.name][str(dt)].append(
                    f["Norm"][str(last_step)][0]
                )
        # ------------------
        # Adaptive
        if parsed_args.adaptive:
            file_name = f"adaptive_test_{butcher_tableau.name}"
            file_name = file_name.replace(" ", "_")
            file_name = file_name.replace(",", "")
            file_name = file_name.lower()
            file_path = FOLDER_PATH_ADAPT / f"{file_name}.h5"
            time = {}
            solutions = {}

            with h5py.File(
                file_path,
                mode="r",
                driver="mpio",
                comm=comm,
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
            print(time)
            for quantity in QUANTITY:
                solutions[quantity] = dict(sorted(solutions[quantity].items()))
                adaptive_error_data[quantity] = dict(
                    sorted(adaptive_error_data[quantity].items())
                )

            delta_t = [0] * len(time)
            for i in range(len(time) - 1):
                delta_t[i] = time[i + 1] - time[i]
            delta_t[i + 1] = delta_t[i]
            average_dt_data[butcher_tableau.name] = np.average(delta_t)

            # -----------------
            if rank == 0:
                # Generate Figure

                fig, axs = plt.subplots(3)
                plt.suptitle(butcher_tableau.name)

                axs[0].set(ylabel=f"$y_1$")
                axs[0].grid(True)
                axs[0].plot(time.values(), solutions[QUANTITY[0]].values(), "-")
                axs[1].set(ylabel=f"$y_2$")
                axs[1].grid(True)
                axs[1].plot(time.values(), solutions[QUANTITY[1]].values(), "-")
                if butcher_tableau == third_order_esdirk:
                    axs[2].plot(0.99647, 0.003225, "or", fillstyle="full")
                axs[2].set(xlabel=f"$t$", ylabel=f"$\Delta t$")
                axs[2].grid(True)
                axs[2].plot(time.values(), delta_t, "-")
                axs[2].plot(
                    [0, 10], [np.average(delta_t), np.average(delta_t)], "k--", lw=1
                )
                axs[2].text(0.05, np.average(delta_t) * (1.01), r"$\Delta \bar{t}$")

                for i in range(3):
                    axs[i].set_xlim(0, OBJECTIVE_TIME)

                fig.savefig(f"{file_name}_analytical.png")

                # ----------------------------------------------

                fig, axs = plt.subplots(3)
                plt.suptitle(butcher_tableau.name)

                axs[0].set(ylabel=r"Global error $\epsilon^g_{y_1}$")
                axs[0].grid(True)
                axs[0].plot(
                    time.values(),
                    adaptive_error_data[QUANTITY[0]].values(),
                    "-",
                    label="Adaptive",
                )
                axs[0].set_xticklabels([])
                axs[1].set(ylabel=r"Global error $\epsilon^g_{y_2}$")
                axs[1].grid(True)
                axs[1].plot(
                    time.values(),
                    adaptive_error_data[QUANTITY[1]].values(),
                    "-",
                )
                axs[1].set_xticklabels([])

                axs[2].set(xlabel=f"$t$", ylabel=f"$\Delta t$")
                axs[2].grid(True)
                axs[2].plot(time.values(), delta_t, "-")
                axs[2].plot(
                    [0, 10], [np.average(delta_t), np.average(delta_t)], "--", lw=1
                )
                for i in range(3):
                    axs[i].set_xlim(0, OBJECTIVE_TIME)
                fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.96))
                fig.savefig(f"{file_name}_analytical_error.png")
    if rank == 0:
        # PLOT LOCAL ERROR DATA
        fig, ax = penv.gen_figure()

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
            if p == 2:
                plt.loglog(
                    DELTA_T_LIST,
                    (local_error_data[scheme.name]["0.001"] / DELTA_T_LIST[0] ** p)
                    * (DELTA_T_LIST) ** p,
                    "k--",
                    lw=1,
                )

            if (i + 1) % 2 != 0:
                if p == 2:
                    print(scheme.name)
                    err = local_error_data[f"{scheme.name}"]["0.002"]
                    print(local_error_data[f"{scheme.name}"])
                    point = np.asarray(
                        [
                            float(DELTA_T_LIST[1]),
                            float(err[0]),
                        ]
                    )
                    print(point)
                    penv.plot_left_upper_triangle(
                        ax,
                        point,
                        y_order=p,
                        x_order=1,
                        x_length=0.3,
                        x_off=0.4,
                        y_off=-1.2,
                    )
        ax.set_xlim(DELTA_T_LIST[0], DELTA_T_LIST[-1])
        ax.legend()
        fig.savefig(f"local_error_time_plot.png")
