import numpy as np
import matplotlib.pyplot as plt


errors_half_1 = np.zeros(1001)
errors_half_2 = np.zeros(1001)
errors_rel_half_1 = np.zeros(1001)
errors_rel_half_2 = np.zeros(1001)

with open("monolithic.txt", mode="r", encoding="utf-8") as f_1, open(
    "inner_problem_stage.txt", mode="r", encoding="utf-8"
) as f_2:
    f_1.readline()
    f_2.readline()
    for i in range(1001):
        line_1 = f_1.readline()
        line_2 = f_2.readline()
        _, heat_monolith_str = line_1.split(sep=",")
        _, heat_nested_1_str, heat_nested_2_str = line_2.split(sep=",")
        heat_monolith = np.fromstring(heat_monolith_str[1:-2], sep=" ")
        heat_monolith = heat_monolith.reshape(11, 11)
        heat_nested_1 = np.fromstring(heat_nested_1_str[1:-1], sep=" ")
        heat_nested_2 = np.fromstring(heat_nested_2_str[1:-2], sep=" ")
        errors_half_1[i] = np.linalg.norm(
            heat_monolith[:, :6].reshape(66) - heat_nested_1
        )
        errors_half_2[i] = np.linalg.norm(
            heat_monolith[:, 5:].reshape(66) - heat_nested_2
        )
        errors_rel_half_1[i] = errors_half_1[i] / np.linalg.norm(heat_nested_1)
        errors_rel_half_2[i] = errors_half_2[i] / np.linalg.norm(heat_nested_2)
    times = np.linspace(0.0, 0.1, 1001)

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
