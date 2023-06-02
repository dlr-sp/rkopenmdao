import numpy as np

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau

explicit_euler = ButcherTableau(np.array([[0.0]]), np.array([1.0]), np.array([0.0]))

implicit_euler = ButcherTableau(np.array([[1.0]]), np.array([1.0]), np.array([1.0]))

implicit_midpoint = ButcherTableau(np.array([[0.5]]), np.array([1.0]), np.array([0.5]))

gamma = (2.0 - np.sqrt(2.0)) / 2.0
two_stage_dirk = ButcherTableau(
    np.array([[gamma, 0.0], [1 - gamma, gamma]]),
    np.array([1 - gamma, gamma]),
    np.array([gamma, 1.0]),
)

runge_kutta_four = ButcherTableau(
    np.array(
        [0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]
    ),
    np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
    np.array([0.0, 0.5, 0.5, 1.0]),
)
