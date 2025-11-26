"""Tests the adaptive time integration with the various components of
test_components.py"""

# pylint: disable=duplicate-code

import pytest
import numpy as np

from rkopenmdao.error_controller import ErrorController, ErrorControllerStatus
from rkopenmdao.error_controllers import (
    integral,
    h0_211,
    pid,
)


@pytest.mark.parametrize(
    """controller, error_measure, delta_t, remaining_time, error_history, 
    step_size_history, expected_status""",
    [
        (
            integral(2),
            1.0e-6,
            0.1,
            1.0,
            np.full(2, 1.0e-6),
            np.zeros(2),
            ErrorControllerStatus(0.095, True),
        ),
        (
            integral(2),
            1.0e-5,
            0.1,
            1.0,
            np.full(2, 1.0e-6),
            np.zeros(2),
            ErrorControllerStatus(0.095 * 0.1 ** (1 / 3), False),
        ),
        (
            h0_211(2),
            1.0e-6,
            0.1,
            1.0,
            np.full(2, 1.0e-6),
            np.array([0.1, 0.0]),
            ErrorControllerStatus(0.095, True),
        ),
        (
            h0_211(2),
            1.0e-5,
            0.1,
            1.0,
            np.full(2, 1.0e-6),
            np.array([0.1, 0.0]),
            ErrorControllerStatus(0.095 * 0.1 ** (1 / 4), False),
        ),
        (
            pid(2),
            1.0e-6,
            0.1,
            1.0,
            np.full(2, 1.0e-6),
            np.zeros(2),
            ErrorControllerStatus(0.095, True),
        ),
        (
            pid(2),
            1.0e-5,
            0.1,
            1.0,
            np.array([1.0e-5, 1.0e-4]),
            np.zeros(2),
            ErrorControllerStatus(
                0.095 * 0.1 ** (1 / 36) * 10 ** (-1 / 18) * 0.01 ** (1 / 36), False
            ),
        ),
        # Tests lower bound being upheld
        (
            integral(2, lower_bound=0.01),
            0.5,
            0.1,
            1.0,
            np.full(2, 1.0e-6),
            np.zeros(2),
            ErrorControllerStatus(0.01, False),
        ),
        # Tests upper bound being upheld
        (
            integral(2, upper_bound=0.5),
            1.0e-9,
            0.1,
            1.0,
            np.full(2, 1.0e-6),
            np.zeros(2),
            ErrorControllerStatus(0.5, True),
        ),
        # Tests remaining time being upheld
        (
            integral(2),
            1.0e-12,
            0.1,
            1.0,
            np.full(2, 1.0e-6),
            np.zeros(2),
            ErrorControllerStatus(1.0, True),
        ),
    ],
)
def test_error_controller_step_size_estimation(
    controller: ErrorController,
    error_measure: float,
    delta_t: float,
    remaining_time: float,
    error_history: np.ndarray,
    step_size_history: np.ndarray,
    expected_status: ErrorControllerStatus,
):
    status = controller(
        error_measure, delta_t, remaining_time, error_history, step_size_history
    )
    assert status == pytest.approx(expected_status)
