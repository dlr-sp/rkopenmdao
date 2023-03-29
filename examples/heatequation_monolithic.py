import numpy as np

from examples.heatequation import heat_equation
from examples.heatequation.src.domain import Domain
from examples.heatequation.src.boundary import BoundaryCondition

from src.runge_kutta_openmdao.runge_kutta.runge_kutta import ButcherTableau

example_domain = Domain([0.0, 1.0], [0.0, 1.0], 11, 11)
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

heatequation = heat_equation.HeatEquation(
    example_domain,
    lambda t, x, y: 0.0,
    example_boundary,
    1.0,
    lambda x, y: g(x) * g(y),
    {"tol": 1e-12, "atol": "legacy"},
)

heatequation.solve_heat_equation(butcher_tableau, 1e-4, 1000, "monolithic.txt")
