import numpy as np
import matplotlib.pyplot as plt

import h5py


errors_half_1 = np.zeros(101)
errors_half_2 = np.zeros(101)
errors_rel_half_1 = np.zeros(101)
errors_rel_half_2 = np.zeros(101)

with h5py.File("monolithic.h5", mode="r") as f_1, h5py.File(
    "inner_problem_stage.h5", mode="r"
) as f_2:
    for i in range(0, 101):
        heat_monolith = np.array(f_1["heat/" + str(10 * i)])
        heat_monolith = heat_monolith.reshape(11, 11)
        heat_nested_1 = np.array(f_2["heat_1/" + str(10 * i)])
        heat_nested_2 = np.array(f_2["heat_2/" + str(10 * i)])
        errors_half_1[i] = np.linalg.norm(
            heat_monolith[:, :6].reshape(66) - heat_nested_1
        )
        errors_half_2[i] = np.linalg.norm(
            heat_monolith[:, 5:].reshape(66) - heat_nested_2
        )
        errors_rel_half_1[i] = errors_half_1[i] / np.linalg.norm(heat_nested_1)
        errors_rel_half_2[i] = errors_half_2[i] / np.linalg.norm(heat_nested_2)
    times = np.linspace(0.0, 0.1, 101)

    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust()
    fig.set_size_inches(16, 12)
    fig.suptitle(
        r"Newton solver, solve_subsystems=True, $\Delta t = 10^{-4}$",
        fontsize="xx-large",
    )
    for i in range(2):
        for j in range(2):
            axs[i, j].set_xlabel("time", fontsize="large")
            axs[i, j].set_ylabel("error", fontsize="large")
            axs[i, j].set_xlim(xmin=0.0, xmax=0.1)
            if i == 0:
                axs[i, j].set_ylim(ymin=0.0, ymax=2e-15)
            else:
                axs[i, j].set_ylim(ymin=0.0, ymax=1e-13)
    axs[0, 0].plot(times, errors_half_1)
    print(errors_half_1)
    axs[0, 0].set_title(
        r"Abs. error left, $||\~{u}^{left side} - \~{u}_1||$",
        fontsize="x-large",
    )
    axs[0, 1].plot(times, errors_half_2)
    axs[0, 1].set_title(
        r"Abs. error right, $||\~{u}^{right side} - \~{u}_2||$",
        fontsize="x-large",
    )
    axs[1, 0].plot(times, errors_rel_half_1)
    axs[1, 0].set_title(
        r"Rel. error left,  $||\~{u}^{left side} - \~{u}_1|| / ||\~{u}^{left side}||$",
        fontsize="x-large",
    )
    axs[1, 1].plot(times, errors_rel_half_2)
    axs[1, 1].set_title(
        r"Rel. error right, $||\~{u}^{right side} - \~{u}_2||/||\~{u}^{right side}||$",
        fontsize="x-large",
    )
    fig.savefig("error.png")
