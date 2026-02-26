"""
Test the ErrorMeasurer class.
"""

import numpy as np
import pytest

from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE
from rkopenmdao.error_measurer import (
    SimpleErrorMeasurer,
    ImprovedErrorMeasurer,
    ErrorMeasurer,
)
from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODEResultState
from .odes import RootODE


@pytest.mark.parametrize(
    "measurer, estimate, state, ode, expected_measure",
    [
        (SimpleErrorMeasurer(), np.ones(1), np.full(1, 2.0), RootODE(), 1.0),
        (ImprovedErrorMeasurer(), np.ones(1), np.full(1, 2.0), RootODE(), 1 / 3),
        (
            ImprovedErrorMeasurer(eta=1e-3),
            np.ones(1),
            np.full(1, 2.0),
            RootODE(),
            1 / 1002,
        ),
        (
            ImprovedErrorMeasurer(eps=1e-3),
            np.ones(1),
            np.full(1, 2.0),
            RootODE(),
            1000 / 2001,
        ),
    ],
)
def test_error_measurer(
    measurer: ErrorMeasurer,
    estimate: np.ndarray,
    state: np.ndarray,
    ode: DiscretizedODE,
    expected_measure: float,
):
    """
    Tests the correct calculation of the error measure based on an estimate and ODE.
    """
    result = measurer.get_measure(
        DiscretizedODEResultState(None, estimate, None),
        DiscretizedODEResultState(None, state, None),
        ode,
    )
    assert result == pytest.approx(expected_measure)
