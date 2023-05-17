import numpy as np

from runge_kutta_openmdao.heatequation import heatequation
from runge_kutta_openmdao.heatequation.domain import Domain
from runge_kutta_openmdao.heatequation.boundary import BoundaryCondition

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau

from scipy.sparse.linalg import LinearOperator

points_per_direction = 51
delta_x = 1.0 / (points_per_direction - 1)
example_domain = Domain(
    [0.0, 1.0], [0.0, 1.0], points_per_direction, points_per_direction
)
example_boundary = BoundaryCondition(
    upper=lambda t, x, y: 0.0,
    lower=lambda t, x, y: 0.0,
    left=lambda t, x, y: 0.0,
    right=lambda t, x, y: 0.0,
)


def g(x: float):
    return np.cos(2 * np.pi * x)


gamma = (2.0 - np.sqrt(2.0)) / 2.0
butcher_tableau = ButcherTableau(
    np.array(
        [
            [gamma, 0.0],
            [1 - gamma, gamma],
        ]
    ),
    np.array([1 - gamma, gamma]),
    np.array([gamma, 1.0]),
)
heat_precon = LinearOperator(
    shape=(
        (points_per_direction * points_per_direction),
        (points_per_direction * points_per_direction),
    ),
    matvec=lambda x: delta_x**2 / -4 * x,
)

heatequation = heatequation.HeatEquation(
    example_domain,
    lambda t, x, y: 0.0,
    example_boundary,
    1.0,
    lambda x, y: g(x) * g(y) + 1,
    {"tol": 1e-15, "atol": "legacy", "M": heat_precon},
)

heatequation.solve_heat_equation(butcher_tableau, 1e-5, 10000, "monolithic.h5", 100)
