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

WRITE_FILE = "adaptive"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if __name__ == "__main__":
    butcher_tableaux = [
        two_stage_sdirk,
        four_stage_esdirk,
        five_stage_esdirk,
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
        print(time)
        print(len(time))
        time = dict(sorted(time.items()))
        x = dict(sorted(x.items()))
        delta_t = [0] * len(time)
        for i in range(len(time) - 1):
            delta_t[i] = time[i + 1] - time[i]
        delta_t[i + 1] = delta_t[i]
        if rank == 0:
            # Generate Figure
            fig = plt.figure()
            plt.xlabel("Time t [s]")  # time axis (x axis)
            plt.ylabel("dTime t [s]")  # delta time axis (y axis)
            plt.grid(True)
            plt.title(butcher_tableau.name)
            plt.plot(time.values(), delta_t, "-")
            plt.show()
            fig.savefig(f"{file_name}.pdf")

            # figure 2
            fig = plt.figure()
            plt.xlabel("time t [s]")  # time axis (x axis)
            plt.ylabel("x")  # delta time axis (y axis)
            plt.grid(True)
            plt.title(butcher_tableau.name)
            plt.plot(time.values(), x.values(), "-")
            plt.show()
            fig.savefig(f"{file_name}_graph.pdf")
