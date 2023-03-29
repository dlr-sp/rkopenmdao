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
    axs[0, 0].plot(times, errors_half_1)
    axs[0, 1].plot(times, errors_half_2)
    axs[1, 0].plot(times, errors_rel_half_1)
    axs[1, 1].plot(times, errors_rel_half_2)
    plt.savefig("error.png")
