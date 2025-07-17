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
NONADAPTIVE_FILE = "non_adaptive_test"
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

    folder_path = pathlib.Path(__file__).parent / "data" / "integral"
    for butcher_tableau in butcher_tableaux:
        # Adaptive_file
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
        error = [0] * (len(time))
        for inx in range(len(time)):
            error[inx] = np.abs(ODE_CFD_REAL.solution(time[inx], -1e2) - x[inx])

        # NON ADAPTIVE PART
        file_name = f"{NONADAPTIVE_FILE}_{butcher_tableau.name}"
        file_name = file_name.replace(" ", "_")
        file_name = file_name.replace(",", "")
        file_name = file_name.lower()
        file_path = folder_path / f"{file_name}.h5"
        time_nonadapt = {}
        x_nonadapt = {}
        with h5py.File(
            file_path,
            mode="r",
            driver="mpio",
            comm=comm,
        ) as f:
            group = f["x"]
            print(group.attrs.keys())

            for key in group.keys():
                time_nonadapt.update({int(key): group[key].attrs["time"]})
                x_nonadapt.update({int(key): group[key][0]})
        x_nonadapt = dict(sorted(x_nonadapt.items()))
        time_nonadapt = dict(sorted(time_nonadapt.items()))
        error_nonadapt = [0] * (len(x_nonadapt))
        for inx in range(len(time_nonadapt)):
            error_nonadapt[inx] = np.abs(
                ODE_CFD_REAL.solution(time_nonadapt[inx], -1e2) - x_nonadapt[inx]
            )
        if rank == 0:
            # figure 2
            fig = plt.figure()
            plt.xlabel(f"$t$")  # time axis (x axis)
            plt.ylabel("Error")  # delta time axis (y axis)
            plt.grid(True)
            plt.title(butcher_tableau.name)
            plt.plot(
                time.values(),
                error,
                "-",
                label="adaptive",
            )
            plt.plot(
                time_nonadapt.values(),
                error_nonadapt,
                "-",
                label="non_adaptive",
            )
            plt.legend()
            plt.xlim(time[0], 10.0)
            plt.show()
            fig.savefig(f"{file_name}_error_comparison.pdf")

            print(np.abs(ODE_CFD_REAL.solution(10, -1.0e2) - x[list(x)[-1]]))
