"""
Plots the files created by the van_der_pol_oscillator_computation.py script.
The argument --num_of_files should be set such that it is equal to the number of that
script. This file then creates a gif showing the history of that optimization.
"""

import argparse

import h5py
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--num_of_files", type=int)
parser.add_argument("--y1_optimization", action="store_true")
parser.add_argument("--y2_optimization", action="store_true")
parser.add_argument("--epsilon_parameters", type=int, default=1)
parser.add_argument("--delta_t", type=float, default=0.1)

parsed_args = parser.parse_args()
num_of_files: int = parsed_args.num_of_files
y1_optimization = bool(parsed_args.y1_optimization)
y2_optimization = bool(parsed_args.y2_optimization)
epsilon_parameters = parsed_args.epsilon_parameters
delta_t = parsed_args.delta_t


fig, ax = plt.subplots(1, 1)
fig.suptitle("Van der Pol oscillator optimization")
fig.subplots_adjust(top=0.8)


def animation_frame(iternum: int):
    """Function for a single from for FuncAnimation of matplotlib."""
    with h5py.File(
        f"data/vdp_{delta_t}_{epsilon_parameters}_y1_"
        f"{y1_optimization}_y2_{y2_optimization}_{iternum}.h5",
        "r",
    ) as f:
        y1_arr = np.zeros(len(f["y1"]))
        y2_arr = np.zeros_like(y1_arr)
        times_arr = np.zeros_like(y1_arr)
        for i, val in f["y1"].items():
            y1_arr[int(i)] = val[0]
        for i, val in f["y2"].items():
            y2_arr[int(i)] = val[0]
            times_arr[int(i)] = val.attrs["time"]
        j = f["J"][str(y1_arr.size - 1)][:]

    ax.clear()
    ax.set_xlim(-3.1, 3.1)
    ax.set_ylim(-3.1, 3.1)
    ax.set_xlabel("y_1")
    ax.set_ylabel("y_2")
    ax.plot(y1_arr, y2_arr, color="black")
    ax.plot(y1_arr[0], y2_arr[0], color="black", marker="x")
    ax.plot(
        np.cos(np.linspace(0, 2 * np.pi, num=20)),
        np.sin(np.linspace(0, 2 * np.pi, num=20)),
        color="black",
        linestyle="dashed",
    )
    ax.set_title(f"y_1 = {y1_arr[0]}, y_2 = {y2_arr[0]}\n J = {j}")


# ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat=False)
ani = animation.FuncAnimation(
    fig,
    animation_frame,
    frames=num_of_files,
    interval=500,
    blit=False,
    repeat_delay=2000,
)
ani.save(
    f"vdp_{delta_t}_{epsilon_parameters}_y1_{y1_optimization}_y2_{y2_optimization}.gif"
)
