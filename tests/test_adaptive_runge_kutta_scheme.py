"""Tests the adaptive Runge-kutta scheme"""

# pylint: disable = c-extension-no-member

import numpy as np
import pytest
from mpi4py import MPI
from rkopenmdao.runge_kutta_scheme import RungeKuttaScheme
from rkopenmdao.butcher_tableaux import (
    embedded_heun_euler,
)
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
from tests.test_runge_kutta_scheme import (
    RkFunctionProvider,
    RootOdeJacvec,
    RootOdeJacvecTransposed,
)

root_ode_jacvec = RootOdeJacvec()
root_ode_jacvec_transposed = RootOdeJacvecTransposed()
root_ode_provider = RkFunctionProvider(
    lambda old_state, acc_stages, stage_time, delta_t, butcher_diagonal_element: delta_t
    * butcher_diagonal_element
    / 2
    + np.sqrt(
        delta_t**2 * butcher_diagonal_element**2 / 4 + old_state + delta_t * acc_stages
    ),
    root_ode_jacvec,
    root_ode_jacvec_transposed,
)
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
root_ode_embedded_heun = RungeKuttaScheme(
    embedded_heun_euler,
    root_ode_provider.stage_computation_functor,
    root_ode_provider.stage_computation_functor_jacvec,
    root_ode_provider.stage_computation_functor_transposed_jacvec,
    use_adaptive_time_stepping=True,
    error_controller=error_controller,
)


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize(
    "rk_scheme, delta_t, old_state, stage_field, expected_new_state",
    (
        [
            root_ode_embedded_heun,
            0.1,
            np.array([1.0]),
            np.array([[1.0], [21 / 20]]),
            np.array([1.1025]),
        ],
    ),
)
def test_compute_step(
    rk_scheme: RungeKuttaScheme,
    delta_t: float,
    old_state: np.ndarray,
    stage_field: np.ndarray,
    expected_new_state: np.ndarray,
):
    """Tests the compute_step function."""
    assert rk_scheme.compute_step(delta_t, old_state, stage_field) == pytest.approx(
        (np.array([expected_new_state]), 0.1, True)
    )
