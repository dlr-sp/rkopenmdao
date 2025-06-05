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
from rkopenmdao.error_estimator import (
    SimpleErrorEstimator,
    ImprovedErrorEstimator,
)
from rkopenmdao.metadata_extractor import (
    TimeIntegrationMetadata,
    TimeIntegrationTranslationMetadata,
    TimeIntegrationQuantity,
    PostprocessingQuantity,
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
post_proc_quantity = PostprocessingQuantity(
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
    postprocessing_quantity_list=[post_proc_quantity],
    time_independent_input_quantity_list=[time_ind_quantity],
)

simple_error_estimator = SimpleErrorEstimator(2, comm=comm, quantity_metadata=metadata)
improved_error_estimator = ImprovedErrorEstimator(
    2, comm=comm, quantity_metadata=metadata
)
error_controller = pid(
    embedded_heun_euler.min_p_order(), error_estimator=simple_error_estimator
)


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize(
    "ode, embedded_tableau, delta_t, old_state, stage_field, expected_new_state",
    (
        [
            RootODE,
            embedded_heun_euler,
            0.1,
            np.array([1.0]),
            np.array([[1.0], [21 / 20]]),
            np.array([1.1025]),
        ],
    ),
)
def test_compute_step(
    ode: DiscretizedODE,
    embedded_tableau: EmbeddedButcherTableau,
    delta_t: float,
    old_state: np.ndarray,
    stage_field: np.ndarray,
    expected_new_state: np.ndarray,
):
    """Tests the compute_step function."""
    rk_scheme = RungeKuttaScheme(embedded_tableau, ode, True, error_controller)
    assert rk_scheme.compute_step(delta_t, old_state, stage_field) == pytest.approx(
        (np.array([expected_new_state]), 0.1, True)
    )
