import numpy as np
import h5py
from numba import vectorize, float64


@vectorize([float64(float64, float64, float64)])
def analytic_solution_to_example_heat_equation(t, x, y):
    return (
        np.exp(-8 * np.pi**2 * t) * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + 1
    )


points_per_direction = 51
x = np.linspace(0.0, 1.0, points_per_direction)
y = np.linspace(0.0, 1.0, points_per_direction)

x, y = np.meshgrid(x, y)

checkpoint_distance = 100
with h5py.File("analytic.h5", mode="w") as f:
    for i in range(0, 101):
        time = i * 1e-3
        step_result = analytic_solution_to_example_heat_equation(time, x, y).reshape(
            points_per_direction * points_per_direction
        )
        f.create_dataset("heat/" + str(checkpoint_distance * i), data=step_result)
