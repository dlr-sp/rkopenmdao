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


WRITE_FILE = "adaptive_test"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if __name__ == "__main__":
    butcher_tableaux = [
        second_order_sdirk,
        second_order_esdirk,
        third_order_sdirk,
        third_order_esdirk,
        fourth_order_sdirk,
        fourth_order_esdirk,
    ]
    folder_path = pathlib.Path(__file__).parent / "data"
    for butcher_tableau in butcher_tableaux:
        file_name = f"{WRITE_FILE}_{butcher_tableau.name}"
        file_name = file_name.replace(" ", "_")
        file_name = file_name.replace(",", "")
        file_name = file_name.lower()
        file_path = folder_path / f"{file_name}.h5"
        time = {}
        x = {}
        with h5py.File(
            file_path,
            mode="r",
            driver="mpio",
            comm=comm,
        ) as f:
            steps = list(f.keys())[0]
            group = f["x"]
            print(group.attrs.keys())

            for key in group.keys():
                time.update({int(key): group[key].attrs["time"]})
                x.update({int(key): group[key][0]})
        time = dict(sorted(time.items()))
        x = dict(sorted(x.items()))
        delta_t = [0] * len(time)
        for i in range(len(time) - 1):
            delta_t[i] = time[i + 1] - time[i]
        delta_t[i + 1] = delta_t[i]
        print(len(delta_t))
        print(np.average(delta_t))
        if rank == 0:
            # Generate Figure

            fig, axs = plt.subplots(2)
            plt.suptitle(butcher_tableau.name)

            axs[0].set(ylabel=f"$x$")
            axs[0].grid(True)
            axs[0].plot(time.values(), x.values(), "-")

            axs[1].set(xlabel=f"$x$")
            axs[1].set(xlabel=f"$t$", ylabel=f"$\Delta t$")
            axs[1].grid(True)
            axs[1].plot(time.values(), delta_t, "-")
            axs[1].plot(
                [0, 10], [np.average(delta_t), np.average(delta_t)], "k--", lw=1
            )

            fig.savefig(f"{file_name}_analytical.pdf")
