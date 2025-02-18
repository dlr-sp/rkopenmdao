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

parsed_args = parser.parse_args()
num_of_files: int = parsed_args.num_of_files


fig, ax = plt.subplots(1, 1)
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-2.1, 2.1)
ims = []
for files in range(num_of_files):
    with h5py.File(f"vdp_{files}.h5", "r") as f:
        y1_arr = np.zeros(len(f["y1"]))
        y2_arr = np.zeros_like(y1_arr)
        times_arr = np.zeros_like(y1_arr)
        for i, val in f["y1"].items():
            y1_arr[int(i)] = val[0]
        for i, val in f["y2"].items():
            y2_arr[int(i)] = val[0]
            times_arr[int(i)] = val.attrs["time"]

    (im0,) = ax.plot(y1_arr, y2_arr, color="black")
    (im1,) = ax.plot(y1_arr[0], y2_arr[0], color="black", marker="x")
    ims.append([im0, im1])

ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True, repeat=False)
ani.save("vdp.gif")
