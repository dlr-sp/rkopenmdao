"""A short script that calculates the analytic solution of the heat equation for
comparison with numerical results."""

import numpy as np
import h5py
from numba import vectorize, float64


@vectorize([float64(float64, float64, float64)])
def analytic_solution_to_example_heat_equation(t, x, y):
    """analytical solution to the heat equation
    partial_t u = Delta u
    u(0, x, y) = cos(2 pi x)cos(2 pi y) + 1
    partial_n u = 0
    on the unit square.
    """
    return np.exp(-8 * np.pi**2 * t) * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + 1


POINTS_PER_DIRECTION = 51
x_grid = np.linspace(0.0, 1.0, POINTS_PER_DIRECTION)
y_grid = np.linspace(0.0, 1.0, POINTS_PER_DIRECTION)

x_grid, y_grid = np.meshgrid(x_grid, y_grid)

WRITEOUT_DISTANCE = 1
with h5py.File("analytic.h5", mode="w") as f:
    for i in range(0, 101):
        time = i * 1e-3
        step_result = analytic_solution_to_example_heat_equation(
            time, x_grid, y_grid
        ).reshape(POINTS_PER_DIRECTION * POINTS_PER_DIRECTION)
        f.create_dataset("heat/" + str(WRITEOUT_DISTANCE * i), data=step_result)
