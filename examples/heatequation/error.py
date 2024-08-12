"""Computation and plotting of error given fitting HDF5 files."""

import numpy as np
import matplotlib.pyplot as plt

import h5py


errors_half_1 = np.zeros(101)
errors_half_2 = np.zeros(101)
errors_rel_half_1 = np.zeros(101)
errors_rel_half_2 = np.zeros(101)
errors_half_1_analytic = np.zeros(101)
errors_half_2_analytic = np.zeros(101)
errors_rel_half_1_analytic = np.zeros(101)
errors_rel_half_2_analytic = np.zeros(101)

CHECKPOINT_DISTANCE = 100
POINTS_PER_DIRECTION = 51
HALF_POINTS = POINTS_PER_DIRECTION // 2

with h5py.File("monolithic.h5", mode="r") as f_1, h5py.File(
    "inner_problem_stage.h5", mode="r"
) as f_2, h5py.File("analytic.h5", mode="r") as f_3:
    for i in range(0, 101):
        heat_monolith = np.array(f_1["heat/" + str(CHECKPOINT_DISTANCE * i)])
        heat_monolith = heat_monolith.reshape(
            POINTS_PER_DIRECTION, POINTS_PER_DIRECTION
        )
        heat_nested_1 = np.array(f_2["heat_1/" + str(CHECKPOINT_DISTANCE * i)])
        heat_nested_2 = np.array(f_2["heat_2/" + str(CHECKPOINT_DISTANCE * i)])
        heat_analytic = np.array(f_3["heat/" + str(CHECKPOINT_DISTANCE * i)])
        heat_analytic = heat_analytic.reshape(
            POINTS_PER_DIRECTION, POINTS_PER_DIRECTION
        )
        errors_half_1[i] = np.linalg.norm(
            heat_monolith[:, : HALF_POINTS + 1].reshape(
                (HALF_POINTS + 1) * POINTS_PER_DIRECTION
            )
            - heat_nested_1
        )
        errors_half_2[i] = np.linalg.norm(
            heat_monolith[:, HALF_POINTS:].reshape(
                (HALF_POINTS + 1) * POINTS_PER_DIRECTION
            )
            - heat_nested_2
        )
        errors_rel_half_1[i] = errors_half_1[i] / np.linalg.norm(
            heat_monolith[:, : HALF_POINTS + 1].reshape(
                (HALF_POINTS + 1) * POINTS_PER_DIRECTION
            )
        )
        errors_rel_half_2[i] = errors_half_2[i] / np.linalg.norm(
            heat_monolith[:, HALF_POINTS:].reshape(
                (HALF_POINTS + 1) * POINTS_PER_DIRECTION
            )
        )

        errors_half_1_analytic[i] = np.linalg.norm(
            heat_analytic[:, : HALF_POINTS + 1].reshape(
                (HALF_POINTS + 1) * POINTS_PER_DIRECTION
            )
            - heat_nested_1
        )
        errors_half_2_analytic[i] = np.linalg.norm(
            heat_analytic[:, HALF_POINTS:].reshape(
                (HALF_POINTS + 1) * POINTS_PER_DIRECTION
            )
            - heat_nested_2
        )

        errors_rel_half_1_analytic[i] = errors_half_1_analytic[i] / np.linalg.norm(
            heat_analytic[:, : HALF_POINTS + 1].reshape(
                (HALF_POINTS + 1) * POINTS_PER_DIRECTION
            )
        )
        errors_rel_half_2_analytic[i] = errors_half_2_analytic[i] / np.linalg.norm(
            heat_analytic[:, HALF_POINTS:].reshape(
                (HALF_POINTS + 1) * POINTS_PER_DIRECTION
            )
        )
    times = np.linspace(0.0, 0.1, 101)

    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust()
    fig.set_size_inches(16, 12)
    fig.suptitle(
        r"Newton solver, solve_subsystems=True, "
        r"$\Delta t = 10^{-5}$, $\Delta x = \Delta y = 0.02$",
        fontsize="xx-large",
    )

    axs[0, 0].set_title(
        r"Abs. error to analytic solution: $||{u} - \~{u}_{nested}||_2$",
        fontsize="x-large",
    )

    axs[0, 1].set_title(
        r"Abs. error to monolithic numerical solution:"
        r" $||\~{u}_{monol.} - \~{u}_{nested}||_2$",
        fontsize="x-large",
    )
    (handle_1,) = axs[0, 0].plot(times, errors_half_1_analytic, "go", label="left side")
    (handle_2,) = axs[0, 0].plot(
        times, errors_half_2_analytic, "b+", label="right side"
    )

    axs[0, 1].plot(times, errors_half_1, "go")
    axs[0, 1].plot(times, errors_half_2, "b+")

    axs[1, 0].set_title(
        r"Rel. error to analytic solution: $||{u} - \~{u}_{nested}||_2 / ||{u}||_2$",
        fontsize="x-large",
    )

    axs[1, 1].set_title(
        r"Rel. error to monolithic numerical solution:"
        r" $||\~{u}_{monol.} - \~{u}_{nested}||_2/||\~{u}_{monol.}||_2$ ",
        fontsize="x-large",
    )

    axs[1, 0].plot(times, errors_rel_half_1_analytic, "go")
    axs[1, 0].plot(times, errors_rel_half_2_analytic, "b+")

    axs[1, 1].plot(times, errors_rel_half_1, "go")
    axs[1, 1].plot(times, errors_rel_half_2, "b+")

    for i in range(2):
        for j in range(2):
            axs[i, j].set_xlabel("time", fontsize="large")
            axs[i, j].set_ylabel("error", fontsize="large")
            axs[i, j].set_xlim(left=0.0, right=0.1)
            axs[i, j].set_ylim(bottom=0.0)

    fig.legend(handles=[handle_1, handle_2], loc="lower center", fontsize="x-large")

    fig.savefig("error.png")
