"""Plots the error wrt. to embedded schemes over delta time"""

import argparse
import pathlib

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np

from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as two_stage_sdirk,
    embedded_third_order_four_stage_esdirk as four_stage_esdirk,
    embedded_fourth_order_five_stage_esdirk as five_stage_esdirk,
)
import plot_environment as penv

MARKER = ["-o", "-X", "-P", "-D", "-v", "-H"]
COLORS = ["indigo", "indianred", "seagreen", "slategray", "peachpuff", "lightskyblue"]
BUTCHERNAMES = ["SDIRK2", "ESDIRK2", "SDIRK3", "ESDIRK3", "SDIRK4", "ESDIRK4"]

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_name", default="data", type=str)
    parsed_args = parser.parse_args()
    folder_path = pathlib.Path(__file__).parent / "data"
    butcher_tableaux = [
        two_stage_sdirk,
        four_stage_esdirk,
        five_stage_esdirk,
    ]

    delta_t_list = [
        1e-3,
        2e-3,
        4e-3,
        5e-3,
        1e-2,
        2e-2,
        4e-2,
        5e-2,
        1e-1,
    ]
    delta_t_list = np.array(delta_t_list)

    error_data = {}
    for butcher_tableau in butcher_tableaux:
        error_data[f"{butcher_tableau.name}"] = {}
        for dt in delta_t_list:
            error_data[f"{butcher_tableau.name}"][str(dt)] = []
            file_name = f"{parsed_args.base_name}_{dt:.0E}_{butcher_tableau.name}"
            file_name = file_name.replace(" ", "_")
            file_name = file_name.replace(",", "")
            file_name = file_name.replace(".", "f")
            file_name = file_name.lower()
            file_path = folder_path / f"{file_name}.h5"

            with h5py.File(
                file_path,
                mode="r",
                driver="mpio",
                comm=comm,
            ) as f:
                last_step = int(1.0 / dt)
                error_data[butcher_tableau.name][f"{dt}"].append(
                    f["Norm"][str(last_step)][0]
                )
    if rank == 0:
        fig, ax = penv.gen_figure()

        ax.set_xlabel("$\Delta t$")
        ax.set_ylabel("Local error")

        ax.grid(True)
        for i, scheme in enumerate(butcher_tableaux):
            p = scheme.p
            ax.loglog(
                delta_t_list,
                error_data[f"{scheme.name}"].values(),
                MARKER[i],
                fillstyle="full",
                color=COLORS[i],
                lw=2,
                label=f"{BUTCHERNAMES[i]}",
            )
            ax.plot(
                delta_t_list,
                (error_data[f"{scheme.name}"]["0.001"] / delta_t_list[0] ** p)
                * (delta_t_list) ** p,
                "k--",
                lw=1,
            )
            if (i + 1) % 2 != 0:
                print(scheme.name)
                err = error_data[f"{scheme.name}"]["0.005"]
                print(error_data[f"{scheme.name}"])
                point = np.asarray(
                    [
                        float(delta_t_list[2]),
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
                    y_off=-0.1,
                )

        ax.set_xlim(delta_t_list[0], delta_t_list[-1])
        ax.legend()
        fig.savefig("Analytical_norm_error_time_plot.pdf")
