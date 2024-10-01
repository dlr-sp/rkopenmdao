from typing import Callable, Tuple

import numpy as np
import pytest

from rkopenmdao.runge_kutta_scheme import RungeKuttaScheme
from rkopenmdao.butcher_tableaux import (
    embedded_heun_euler,
    embedded_runge_kutta_fehlberg,
)
from rkopenmdao.error_controllers import PID

class RkFunctionProvider:
    """Wraps functions and their (transposed) derivative into a common interface."""

    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        stage_computation_functor: Callable[
            [np.ndarray, np.ndarray, float, float, float], np.ndarray
        ],
        stage_computation_functor_jacvec: Callable[
            [np.ndarray, np.ndarray, float, float, float], np.ndarray
        ],
        stage_computation_functor_transposed_jacvec: Callable[
            [np.ndarray, float, float, float], Tuple[np.ndarray, np.ndarray]
        ],
    ):
        self.stage_computation_functor = stage_computation_functor
        self.stage_computation_functor_jacvec = stage_computation_functor_jacvec
        self.stage_computation_functor_transposed_jacvec = (
            stage_computation_functor_transposed_jacvec
        )


class RootOdeJacvec:
    """Functor for the jacvec-product of the root function."""

    def __init__(self):
        self.old_state = 0.0
        self.acc_stages = 0.0

    def __call__(
        self,
        old_state_perturb,
        acc_stage_perturb,
        stage_time,
        delta_t,
        butcher_diagonal_element,
    ):
        return (old_state_perturb + delta_t * acc_stage_perturb) / (
            2
            * np.sqrt(
                delta_t**2 * butcher_diagonal_element**2 / 4
                + (self.old_state + delta_t * self.acc_stages)
            )
        )

    def linearize(self, old_state, acc_stages):
        """Saves the linearization point for the jacobian."""
        self.old_state = old_state
        self.acc_stages = acc_stages


class RootOdeJacvecTransposed:
    """Functor for the transposed jacvec-product of the root function."""

    def __init__(self):
        self.old_state = 0.0
        self.acc_stages = 0.0

    def __call__(self, stage_perturb, stage_time, delta_t, butcher_diagonal_element):
        return (
            stage_perturb
            / (
                2
                * np.sqrt(
                    delta_t**2 * butcher_diagonal_element**2 / 4
                    + (self.old_state + delta_t * self.acc_stages)
                )
            ),
            delta_t
            * stage_perturb
            / (
                2
                * np.sqrt(
                    delta_t**2 * butcher_diagonal_element**2 / 4
                    + (self.old_state + delta_t * self.acc_stages)
                )
            ),
        )

    def linearize(self, old_state, acc_stages):
        """Saves the linearization point for the jacobian."""
        self.old_state = old_state
        self.acc_stages = acc_stages


root_ode_jacvec = RootOdeJacvec()
root_ode_jacvec_transposed = RootOdeJacvecTransposed()
root_ode_provider = RkFunctionProvider(
    lambda old_state, acc_stages, stage_time, delta_t, butcher_diagonal_element: delta_t
    * butcher_diagonal_element / 2 + np.sqrt(delta_t**2 * butcher_diagonal_element**2
                                             / 4 + old_state + delta_t * acc_stages),
    root_ode_jacvec,
    root_ode_jacvec_transposed,
)
error_controller = PID(embedded_heun_euler.min_p_order())
root_ode_embedded_heun = RungeKuttaScheme(embedded_heun_euler,root_ode_provider.stage_computation_functor,
                                          root_ode_provider.stage_computation_functor_jacvec,
                                          root_ode_provider.stage_computation_functor_transposed_jacvec,
                                          error_controller)

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
        (np.array([expected_new_state]),.09028504670200695,False)
    )
