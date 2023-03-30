import numpy as np
import h5py

from scipy.sparse.linalg import LinearOperator, expm_multiply

from examples.heatequation.src.domain import Domain
from examples.heatequation.src.fdm_matrix import FdmMatrix

from numba import vectorize, float64


@vectorize([float64(float64, float64, float64)])
def analytic_solution_to_example_heat_equation(t, x, y):
    return (
        np.exp(-8 * np.pi**2 * t) * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + 1
    )


points_per_direction = 21
x = np.linspace(0.0, 1.0, points_per_direction)
y = np.linspace(0.0, 1.0, points_per_direction)

x, y = np.meshgrid(x, y)


example_domain = Domain(
    [0.0, 1.0], [0.0, 1.0], points_per_direction, points_per_direction
)
fdm_mat = FdmMatrix(example_domain, 1.0)
fdm_mat_lin_op = LinearOperator(
    shape=(
        points_per_direction * points_per_direction,
        points_per_direction * points_per_direction,
    ),
    matvec=fdm_mat.mat_vec_prod,
    rmatvec=fdm_mat.mat_vec_prod_transpose,
)
initial = analytic_solution_to_example_heat_equation(0.0, x, y).reshape(
    points_per_direction * points_per_direction
)

step_results = expm_multiply(
    fdm_mat_lin_op,
    initial,
    start=0.0,
    stop=0.1,
    num=101,
    endpoint=True,
    traceA=-4 * points_per_direction**2 * (points_per_direction - 1) ** 2,
)

checkpoint_distance = 10
with h5py.File("analytic_discretized.h5", mode="w") as f:
    for i in range(0, 101, 1):
        f.create_dataset("heat/" + str(checkpoint_distance * i), data=step_results[i])
