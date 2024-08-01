"""Directly solving the heat equation without OpenMDAO for comparison."""

import numpy as np

from rkopenmdao.butcher_tableaux import (
    third_order_third_weak_stage_order_four_stage_dirk,
)

from .heatequation import heatequation
from .heatequation.domain import Domain
from .heatequation.boundary import BoundaryCondition


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
    # pylint: disable=missing-function-docstring
    return np.cos(2 * np.pi * x)


butcher_tableau = third_order_third_weak_stage_order_four_stage_dirk

heatequation = heatequation.HeatEquation(
    example_domain,
    lambda t, x, y: 0.0,
    example_boundary,
    1.0,
    lambda x, y: g(x) * g(y) + 1,
    {"tol": 1e-10, "atol": 1e-10},
)

heatequation.solve_heat_equation(butcher_tableau, 1e-3, 100, "monolithic.h5", 1)
