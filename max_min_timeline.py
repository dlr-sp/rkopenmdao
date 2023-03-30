import numpy as np
import matplotlib.pyplot as plt

import h5py


max_nested = np.zeros(101)
min_nested = np.zeros(101)
max_monolithic = np.zeros(101)
min_monolithic = np.zeros(101)
max_analytic_discretized = np.zeros(101)
min_analytic_discretized = np.zeros(101)
max_analytic = np.zeros(101)
min_analytic = np.zeros(101)

checkpoint_distance = 10
points_per_direction = 21
half_points = points_per_direction // 2

with h5py.File("monolithic.h5", mode="r") as f_1, h5py.File(
    "inner_problem_stage.h5", mode="r"
) as f_2, h5py.File("analytic_discretized.h5", mode="r") as f_3, h5py.File(
    "analytic.h5", mode="r"
) as f_4:
    for i in range(0, 101):
        heat_monolith = np.array(f_1["heat/" + str(checkpoint_distance * i)])
        heat_nested_1 = np.array(f_2["heat_1/" + str(checkpoint_distance * i)])
        heat_nested_2 = np.array(f_2["heat_2/" + str(checkpoint_distance * i)])
        heat_analytic_discretized = np.array(
            f_3["heat/" + str(checkpoint_distance * i)]
        )
        heat_analytic = np.array(f_4["heat/" + str(checkpoint_distance * i)])

        max_nested[i] = heat_nested_1[0]
        min_nested[i] = heat_nested_2[(half_points + 1) ** 2 - 1]

        max_monolithic[i] = heat_monolith[0]
        min_monolithic[i] = heat_monolith[points_per_direction * (half_points + 1) - 1]

        max_analytic_discretized[i] = heat_analytic_discretized[0]
        min_analytic_discretized[i] = heat_analytic_discretized[
            points_per_direction * (half_points + 1) - 1
        ]

        max_analytic[i] = heat_analytic[0]
        min_analytic[i] = heat_analytic[points_per_direction * (half_points + 1) - 1]

    times = np.linspace(0.0, 0.1, 101)

    fig, axs = plt.subplots(1, 2)
    fig.subplots_adjust()
    fig.set_size_inches(16, 12)
    fig.suptitle(
        r"Newton solver, solve_subsystems=True, $\Delta t = 10^{-4}$",
        fontsize="xx-large",
    )
    for i in range(2):
        axs[i].set_xlabel("time", fontsize="large")
        axs[i].set_ylabel("heat", fontsize="large")
        axs[i].set_xlim(xmin=0.0, xmax=0.1)
    axs[0].plot(times, max_nested, "ro")
    axs[0].plot(times, max_monolithic, "b<")
    axs[0].plot(times, max_analytic_discretized, "g>")
    axs[0].plot(times, max_analytic, "y-")
    axs[0].set_title(
        r"TODO",
        fontsize="x-large",
    )
    axs[0].set_ylim(ymin=1.0, ymax=2.0)
    axs[1].plot(times, min_nested, "ro")
    axs[1].plot(times, min_monolithic, "b<")
    axs[1].plot(times, min_analytic_discretized, "g>")
    axs[1].plot(times, min_analytic, "y-")
    axs[1].set_ylim(ymin=0.0, ymax=1.0)
    axs[1].set_title(
        r"TODO",
        fontsize="x-large",
    )

    fig.savefig("max_min_timeline.png")
