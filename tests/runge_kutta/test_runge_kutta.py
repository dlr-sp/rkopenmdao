from typing import Tuple

import numpy as np
import pytest

from runge_kutta_openmdao.runge_kutta.runge_kutta_scheme import RungeKuttaScheme


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize(
    "rk_scheme, stage, delta_t, old_time, old_state, accumulated_stages, expected_stage", []
)
def test_compute_stage(
    rk_scheme: RungeKuttaScheme,
    stage: int,
    delta_t: float,
    old_time: float,
    old_state: np.ndarray,
    accumulated_stages: np.ndarray,
    expected_stage: np.ndarray,
):
    assert (
        rk_scheme.compute_stage(stage, delta_t, old_time, old_state, accumulated_stages)
        == expected_stage
    )


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize("rk_scheme, stage, stage_field, expected_accumulated_stages")
def test_compute_accumulated_stages(
    rk_scheme: RungeKuttaScheme,
    stage: int,
    stage_field: np.ndarray,
    expected_accumulated_stages: np.ndarray,
):
    assert rk_scheme.compute_accumulated_stages(stage, stage_field) == expected_accumulated_stages


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize("rk_scheme, delta_t, old_state, stage_field, expected_new_state")
def test_compute_step(
    rk_scheme: RungeKuttaScheme,
    delta_t: float,
    old_state: np.ndarray,
    stage_field: np.ndarray,
    expected_new_state: np.ndarray,
):
    assert rk_scheme.compute_step(delta_t, old_state, stage_field) == expected_new_state


def test_compute_stage_jacvec(
    rk_scheme: RungeKuttaScheme,
    stage: int,
    delta_t: float,
    old_time: float,
    old_state_perturbation: np.ndarray,
    accumulated_stages_perturbation: np.ndarray,
    linearization_args: dict,
    expected_jacvec_product: np.ndarray,
):
    assert (
        rk_scheme.compute_stage_jacvec(
            stage,
            delta_t,
            old_time,
            old_state_perturbation,
            accumulated_stages_perturbation,
            **linearization_args
        )
        == expected_jacvec_product
    )


def test_compute_stage_transposed_jacvec(
    rk_scheme: RungeKuttaScheme,
    stage: int,
    delta_t: float,
    old_time: float,
    joined_perturbation: np.ndarray,
    linearization_args: dict,
    expected_jacvec_product: Tuple[np.ndarray, np.ndarray],
):
    assert (
        rk_scheme.compute_stage_tranposed_jacvec(
            stage, delta_t, old_time, joined_perturbation, **linearization_args
        )
        == expected_jacvec_product
    )


def test_join_new_state_and_accumulated_stages_perturbations(
    rk_scheme: RungeKuttaScheme,
    stage: int,
    new_state_perturbation: np.ndarray,
    accumulated_stages_perturbation_field: np.ndarray,
    expected_joined_perturbation: np.ndarray,
):
    assert (
        rk_scheme.join_new_state_and_accumulated_stages_perturbations(
            stage, new_state_perturbation, accumulated_stages_perturbation_field
        )
        == expected_joined_perturbation
    )


def test_compute_step_transposed_jacvec(
    rk_scheme: RungeKuttaScheme,
    delta_t: float,
    new_state_perturbation: np.ndarray,
    stage_perturbation_field: np.ndarray,
    expected_old_state_perturbation: np.ndarray,
):
    assert (
        rk_scheme.compute_step_transposed_jacvec(
            delta_t, new_state_perturbation, stage_perturbation_field
        )
        == expected_old_state_perturbation
    )
