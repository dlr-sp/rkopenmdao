import argparse
import numpy as np
import pathlib

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpi4py import MPI

from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as two_stage_dirk,
    embedded_third_order_four_stage_esdirk as four_stage_dirk,
    embedded_fourth_order_five_stage_esdirk as five_stage_esdirk,
)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_name", type=str)
    parsed_args = parser.parse_args()
    folder_path = pathlib.Path(__file__).parent / "data"
    butcher_tableaux = [
        two_stage_dirk,
        four_stage_dirk,
        five_stage_esdirk,
    ]

    delta_t_list = [1e-4, 0.5e-3, 1e-3, 0.5e-2, 1e-2, 0.5e-1, 1e-1, 0.5, 1.0]
    delta_t_list = np.array(delta_t_list)

    error_data = {}
    for butcher_tableau in butcher_tableaux:
        error_data[f"{butcher_tableau.name}"] = {}
        for dt in delta_t_list:
            error_data[f"{butcher_tableau.name}"][str(dt)] = []
            file_path = (
                folder_path / f"{parsed_args.base_name}_{dt}_{butcher_tableau.name}.h5"
            )
            # print(file_path)

            with h5py.File(
                file_path,
                mode="r",
                driver="mpio",
                comm=comm,
            ) as f:
                last_step = int(10.0 / dt)
                error_data[butcher_tableau.name][f"{dt}"].append(
                    f["Norm"][str(last_step)][0]
                )
    print(error_data[butcher_tableau.name].values())
    if rank == 0:
        fig = plt.figure()
        # x axis
        plt.xlabel("Step size t [s] (log scale)")
        plt.xscale("log")
        # y axis
        plt.ylabel("Norm Error E [-] (log scale)")
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
        fig.savefig("norm_error_time_plot.pdf")
