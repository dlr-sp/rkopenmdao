"""Plots the error wrt. to analytical solution over delta time"""

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
from .odes import ODE_CFD

OBJECTIVE_TIME = 10.0  # in seconds
QUANTITY = "x"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if __name__ == "__main__":
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
        # 1.0e-4,
        # 0.5e-3,
        # 1e-3,
        # 0.5e-2,
        1e-2,
        0.5e-1,
        1e-1,
        0.5,
        1.0,
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
            file_name = file_name.lower()
            file_path = folder_path / f"{file_name}.h5"
            with h5py.File(
                file_path,
                mode="r",
                driver="mpio",
                comm=comm,
            ) as f:
                last_step = int(OBJECTIVE_TIME / dt)
                error_data[butcher_tableau.name][f"{dt}"].append(
                    np.abs(
                        ODE_CFD.solution(OBJECTIVE_TIME, -1e1)
                        - f[QUANTITY][str(last_step)][0]
                    )
                )
                print(
                    ODE_CFD.solution(OBJECTIVE_TIME, -1e1),
                    f[QUANTITY][str(last_step)][0],
                )
    if rank == 0:
        fig = plt.figure()
        # x axis
        plt.xlabel("Step size t [s] (log scale)")
        plt.xscale("log")
        # y axis
        plt.ylabel("Error E [-] (log scale)")
        plt.yscale("log")
        plt.grid(True)
        for scheme in butcher_tableaux:
            p = scheme.p
            plt.plot(
                delta_t_list,
                error_data[f"{scheme.name}"].values(),
                lw=2,
                label=f"{scheme.name}",
            )
            plt.plot(
                delta_t_list,
                (error_data[f"{scheme.name}"]["1.0"] / delta_t_list[-1] ** p)
                * (delta_t_list) ** p,
                "k--",
                lw=1,
            )

        plt.xlim(delta_t_list[0], delta_t_list[-1])
        plt.legend()
        plt.show()
        fig.savefig("analytical_error_time_plot.pdf")
