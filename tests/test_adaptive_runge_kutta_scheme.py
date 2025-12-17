"""Tests the adaptive Runge-kutta scheme"""

# pylint: disable = c-extension-no-member

import numpy as np
import pytest
from mpi4py import MPI
from rkopenmdao.runge_kutta_scheme import RungeKuttaScheme
from rkopenmdao.butcher_tableau import EmbeddedButcherTableau
from rkopenmdao.butcher_tableaux import (
    embedded_heun_euler,
)
from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE
from rkopenmdao.error_measurer import (
    ErrorMeasurer,
    SimpleErrorMeasurer,
    ImprovedErrorMeasurer,
)
from rkopenmdao.metadata_extractor import (
    TimeIntegrationMetadata,
    TimeIntegrationTranslationMetadata,
    TimeIntegrationQuantity,
    TimeIndependentQuantity,
    ArrayMetadata,
)
from rkopenmdao.error_controllers import pid

from .test_runge_kutta_scheme import RootODE

comm = MPI.COMM_WORLD
array_metadata = ArrayMetadata()
translations_metadata = TimeIntegrationTranslationMetadata()
time_int_quantity = TimeIntegrationQuantity(
    name="x",
    type="time_integration",
    array_metadata=array_metadata,
    translation_metadata=translations_metadata,
)
time_ind_quantity = TimeIndependentQuantity(
    name="x",
    type="time_integration",
    array_metadata=array_metadata,
    translation_metadata=translations_metadata,
)
metadata = TimeIntegrationMetadata(
    time_integration_quantity_list=[time_int_quantity],
    time_independent_input_quantity_list=[time_ind_quantity],
)

error_controller = pid(embedded_heun_euler.min_p_order())


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize(
    """ode, embedded_tableau, delta_t, old_state, stage_field, error_measurer,
    expected_new_state, expected_error_measure""",
    (
        [
            RootODE(),
            embedded_heun_euler,
            0.1,
            np.array([1.0]),
            np.array([[1.0], [21 / 20]]),
            SimpleErrorMeasurer(),
            np.array([1.1025]),
            0.0025,
        ],
        [
            RootODE(),
            embedded_heun_euler,
            0.1,
            np.array([1.0]),
            np.array([[1.0], [21 / 20]]),
            ImprovedErrorMeasurer(),
            np.array([1.1025]),
            0.0025 / (1.1025 + 1),
        ],
    ),
)
def test_compute_step(
    ode: DiscretizedODE,
    embedded_tableau: EmbeddedButcherTableau,
    delta_t: float,
    old_state: np.ndarray,
    stage_field: np.ndarray,
    error_measurer: ErrorMeasurer,
    expected_new_state: np.ndarray,
    expected_error_measure: float,
):
    """Tests the compute_step function."""
    rk_scheme = RungeKuttaScheme(
        embedded_tableau,
        ode,
        True,
        error_controller,
        error_measurer,
    )
    assert rk_scheme.compute_step(
        delta_t,
        old_state,
        stage_field,
        delta_t,
        np.full(2, error_controller.config.tol),
        np.zeros(2),
    ) == pytest.approx(
        (np.array([expected_new_state]), 0.1, True, expected_error_measure)
    )
