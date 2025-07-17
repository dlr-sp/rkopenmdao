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
from .odes import ODE_CFD_REAL
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
        1e-4,
        2e-4,
        4e-4,
        5e-4,
        1e-3,
        2e-3,
        4e-3,
        5e-3,
        1e-2,
    ]
)
FOLDER_ADAPT = "integral"
FOLDER_PATH = pathlib.Path(__file__).parent / "data"
FOLDER_PATH_ADAPT = pathlib.Path(__file__).parent / "data" / FOLDER_ADAPT
OBJECTIVE_TIME = 10.0  # in seconds
QUANTITY = "x"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if __name__ == "__main__":

    error_data = {}
    local_error_data = {}
    adaptive_error_data = {}
    average_dt_data = {}

    for butcher_tableau in BUTCHER_TABLEAUX:
        error_data[f"{butcher_tableau.name}"] = {}
        local_error_data[f"{butcher_tableau.name}"] = [0] * int(OBJECTIVE_TIME)
        for i in range(10):
            local_error_data[butcher_tableau.name][i] = {}
        adaptive_error_data[f"{butcher_tableau.name}"] = {}
        average_dt_data[f"{butcher_tableau.name}"] = {}
        for dt in DELTA_T_LIST:
            error_data[f"{butcher_tableau.name}"][str(dt)] = []

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
                error_data[butcher_tableau.name][str(dt)].append(
                    np.abs(
                        ODE_CFD_REAL.solution(OBJECTIVE_TIME, -1e2)
                        - f[QUANTITY][str(last_step)][0]
                    )
                )
                group = f["Norm"]

                for i in range(10):
                    timestep = int((last_step / 10) * (1 + i))
                    local_error_data[butcher_tableau.name][i][str(dt)] = group[
                        str(timestep)
                    ][0]
            print(local_error_data[butcher_tableau.name])

        if parsed_args.adaptive:
            file_name = f"adaptive_test_{butcher_tableau.name}"
            file_name = file_name.replace(" ", "_")
            file_name = file_name.replace(",", "")
            file_name = file_name.lower()
            file_path = FOLDER_PATH_ADAPT / f"{file_name}.h5"
            time = {}
            x = {}

            with h5py.File(
                file_path,
                mode="r",
                driver="mpio",
                comm=comm,
            ) as f:
                # coefficient values
                group = f["x"]
                step = -1
                for key in group.keys():
                    time.update({int(key): group[key].attrs["time"]})
                    x.update({int(key): group[key][0]})
                    step += 1
                adaptive_error_data[butcher_tableau.name] = np.abs(
                    ODE_CFD_REAL.solution(OBJECTIVE_TIME, -1e2) - x[step]
                )

            time = dict(sorted(time.items()))
            x = dict(sorted(x.items()))

            delta_t = [0] * len(time)
            for i in range(len(time) - 1):
                delta_t[i] = time[i + 1] - time[i]
            delta_t[i + 1] = delta_t[i]
            average_dt_data[butcher_tableau.name] = np.average(delta_t)
            print(average_dt_data[butcher_tableau.name])

            if rank == 0:
                # Generate Figure

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

    if rank == 0:
        # PLOT LOCAL ERROR DATA
        for j in range(len(local_error_data[f"{butcher_tableau.name}"])):
            fig, ax = penv.gen_figure()
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

                plt.loglog(
                    DELTA_T_LIST,
                    (local_error_data[scheme.name][j]["0.0001"] / DELTA_T_LIST[0] ** p)
                    * (DELTA_T_LIST) ** p,
                    "k--",
                    lw=1,
                )

                if (i + 1) % 2 != 0:
                    print(scheme.name)
                    err = local_error_data[f"{scheme.name}"][j]["0.0002"]
                    print(local_error_data[f"{scheme.name}"])
                    point = np.asarray(
                        [
                            float(DELTA_T_LIST[1]),
                            float(err),
                        ]
                    )
                    print(point)
                    penv.plot_left_upper_triangle(
                        ax,
                        point,
                        y_order=p,
                        x_order=1,
                        x_length=0.15,
                        x_off=0.0,
                        y_off=-0.2,
                    )

            ax.set_xlim(DELTA_T_LIST[0], DELTA_T_LIST[-1])
            ax.legend()
            fig.savefig(f"local_error_time_plot_{j+1}.png")

        # PLOT GLOBAL ERROR DATA
        fig, ax = penv.gen_figure()
        # x axis
        ax.set_xlabel(f"$\Delta t$")
        # y axis
        ax.set_ylabel("Global error")
        ax.grid(True)
        for i, scheme in enumerate(BUTCHER_TABLEAUX):
            ax.loglog(
                DELTA_T_LIST,
                error_data[scheme.name].values(),
                MARKER[i],
                lw=2,
                fillstyle="full",
                color=COLORS[i],
                label=f"{BUTCHERNAMES[i]}",
            )

            p = scheme.p
            ax.loglog(
                DELTA_T_LIST,
                (error_data[scheme.name]["0.0001"] / DELTA_T_LIST[0] ** p)
                * (DELTA_T_LIST) ** p,
                "k--",
                lw=1,
            )
            if (i + 1) % 2 != 0:
                print(scheme.name)
                err = error_data[f"{scheme.name}"]["0.0002"]
                print(error_data[f"{scheme.name}"])
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
                    x_length=0.07,
                    x_off=0.0,
                    y_off=-0.2,
                )

        ax.set_xlim(DELTA_T_LIST[0], DELTA_T_LIST[-1])
        ax.legend()
        fig.savefig(f"analytical_error_time_plot.pdf")
