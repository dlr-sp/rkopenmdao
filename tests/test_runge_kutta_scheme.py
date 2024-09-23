"""Tests for the Runge-Kutta integration core."""

from __future__ import annotations
from collections.abc import Callable

import numpy as np
import pytest

from rkopenmdao.runge_kutta_scheme import RungeKuttaScheme
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    second_order_two_stage_sdirk as two_stage_dirk,
    runge_kutta_four,
)


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
            [np.ndarray, float, float, float], tuple[np.ndarray, np.ndarray]
        ],
    ):
        self.stage_computation_functor = stage_computation_functor
        self.stage_computation_functor_jacvec = stage_computation_functor_jacvec
        self.stage_computation_functor_transposed_jacvec = (
            stage_computation_functor_transposed_jacvec
        )


def identity_ode_function(
    old_state, acc_stages, stage_time, delta_t, butcher_diagonal_element
):
    """Solution for the stage update for the identity ode"""
    # pylint: disable=unused-argument
    return (old_state + delta_t * acc_stages) / (1 - delta_t * butcher_diagonal_element)


def identity_ode_jacvec(
    old_state_perturb, acc_stage_perturb, stage_time, delta_t, butcher_diagonal_element
):
    """Jacvec for the stage update for the identity ode"""
    # pylint: disable=unused-argument
    return (old_state_perturb + delta_t * acc_stage_perturb) / (
        1 - delta_t * butcher_diagonal_element
    )


def identity_ode_transposed_jacvec(
    stage_perturb, stage_time, delta_t, butcher_diagonal_element
):
    """Transposed jacvec for the stage update for the identity ode"""
    # pylint: disable=unused-argument
    return (
        stage_perturb / (1 - delta_t * butcher_diagonal_element),
        delta_t * stage_perturb / (1 - delta_t * butcher_diagonal_element),
    )


identity_ode_provider = RkFunctionProvider(
    identity_ode_function, identity_ode_jacvec, identity_ode_transposed_jacvec
)


def time_ode_solution(
    old_state, acc_stages, stage_time, delta_t, butcher_diagonal_element
):
    """Solution for the stage update for the time ode"""
    # pylint: disable=unused-argument
    return stage_time


def time_ode_jacvec(
    old_state_perturb, acc_stage_perturb, stage_time, delta_t, butcher_diagonal_element
):
    """Jacvec for the stage update for the time ode"""
    # pylint: disable=unused-argument
    return np.zeros(1)


def time_ode_tranposed_jacvec(
    stage_perturb, stage_time, delta_t, butcher_diagonal_element
):
    """Transposed jacvec for the stage update for the time ode"""
    # pylint: disable=unused-argument
    return np.zeros(1), np.zeros(1)


time_ode_provider = RkFunctionProvider(
    time_ode_solution, time_ode_jacvec, time_ode_tranposed_jacvec
)


def time_scaled_identity_ode_solution(
    old_state, acc_stages, stage_time, delta_t, butcher_diagonal_element
):
    """Solution for the stage update for the time scaled identity ode"""
    return (
        stage_time
        * (old_state + delta_t * acc_stages)
        / (1 - stage_time * delta_t * butcher_diagonal_element)
    )


def time_scaled_identity_ode_jacvec(
    old_state_perturb, acc_stage_perturb, stage_time, delta_t, butcher_diagonal_element
):
    """Jacvec for the stage update for the time scaled identity ode"""
    return (
        stage_time
        * (old_state_perturb + delta_t * acc_stage_perturb)
        / (1 - stage_time * delta_t * butcher_diagonal_element)
    )


def time_scaled_identity_ode_transposed_jacvec(
    stage_perturb, stage_time, delta_t, butcher_diagonal_element
):
    """Transposed jacvec for the stage update for the time scaled identity ode"""
    return (
        stage_time
        * stage_perturb
        / (1 - stage_time * delta_t * butcher_diagonal_element),
        stage_time
        * delta_t
        * stage_perturb
        / (1 - stage_time * delta_t * butcher_diagonal_element),
    )


time_scaled_identity_ode_provider = RkFunctionProvider(
    time_scaled_identity_ode_solution,
    time_scaled_identity_ode_jacvec,
    time_scaled_identity_ode_transposed_jacvec,
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


root_ode_jacvec = RootOdeJacvec()


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

identity_ode_rk4_scheme = RungeKuttaScheme(
    runge_kutta_four,
    identity_ode_provider.stage_computation_functor,
    identity_ode_provider.stage_computation_functor_jacvec,
    identity_ode_provider.stage_computation_functor_transposed_jacvec,
)
time_ode_implicit_euler_scheme = RungeKuttaScheme(
    implicit_euler,
    time_ode_provider.stage_computation_functor,
    time_ode_provider.stage_computation_functor_jacvec,
    time_ode_provider.stage_computation_functor_transposed_jacvec,
)

time_scaled_identity_ode_implicit_euler_scheme = RungeKuttaScheme(
    implicit_euler,
    time_scaled_identity_ode_provider.stage_computation_functor,
    time_scaled_identity_ode_provider.stage_computation_functor_jacvec,
    time_scaled_identity_ode_provider.stage_computation_functor_transposed_jacvec,
)

root_ode_two_stage_dirk_scheme = RungeKuttaScheme(
    two_stage_dirk,
    root_ode_provider.stage_computation_functor,
    root_ode_provider.stage_computation_functor_jacvec,
    root_ode_provider.stage_computation_functor_transposed_jacvec,
)


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize(
    """rk_scheme, stage, delta_t, old_time, old_state, accumulated_stages, 
    expected_stage""",
    (
        [identity_ode_rk4_scheme, 0, 0.1, 0.0, 1.0, 1.0, 1.1],
        [identity_ode_rk4_scheme, 3, 0.01, 0.0, 2.0, 10.0, 2.1],
        [time_ode_implicit_euler_scheme, 0, 0.1, 1.0, 1.0, 1.0, 1.1],
        [time_ode_implicit_euler_scheme, 0, 0.01, 10, 20.0, 111.1, 10.01],
        [
            time_scaled_identity_ode_implicit_euler_scheme,
            0,
            0.1,
            1.0,
            1.0,
            1.0,
            1.21 / 0.89,
        ],
        [
            time_scaled_identity_ode_implicit_euler_scheme,
            0,
            0.01,
            10,
            20.0,
            111.1,
            211.32111 / 0.8999,
        ],
        [
            root_ode_two_stage_dirk_scheme,
            0,
            0.1,
            1.0,
            0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            0.55 * ((2.0 - np.sqrt(2.0)) / 2.0),
        ],
        [
            root_ode_two_stage_dirk_scheme,
            0,
            0.01,
            1.0,
            0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            0.505 * ((2.0 - np.sqrt(2.0)) / 2.0),
        ],
    ),
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
    """Tests the compute_stage function."""
    assert rk_scheme.compute_stage(
        stage, delta_t, old_time, old_state, accumulated_stages
    ) == pytest.approx(expected_stage)


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize(
    "rk_scheme, stage, stage_field, expected_accumulated_stages",
    (
        [
            identity_ode_rk4_scheme,
            1,
            np.array([[1.0], [0.0], [0.0], [0.0]]),
            np.array([0.5]),
        ],
        [
            identity_ode_rk4_scheme,
            2,
            np.array([[1.0], [21 / 20], [0.0], [0.0]]),
            np.array([21 / 40]),
        ],
        [
            identity_ode_rk4_scheme,
            3,
            np.array([[1.0], [21 / 20], [421 / 400], [0.0]]),
            np.array([421 / 400]),
        ],
        [root_ode_two_stage_dirk_scheme, 1, np.array([[2.0]]), np.array([2**0.5])],
    ),
)
def test_compute_accumulated_stages(
    rk_scheme: RungeKuttaScheme,
    stage: int,
    stage_field: np.ndarray,
    expected_accumulated_stages: np.ndarray,
):
    """Tests the compute_accumulated_stages function."""
    assert rk_scheme.compute_accumulated_stages(stage, stage_field) == pytest.approx(
        expected_accumulated_stages
    )


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize(
    "rk_scheme, delta_t, old_state, stage_field, expected_new_state",
    (
        [
            identity_ode_rk4_scheme,
            0.1,
            np.array([1.0]),
            np.array([[1.0], [21 / 20], [421 / 400], [4421 / 4000]]),
            np.array([26524.1 / 24000]),
        ],
        [
            time_scaled_identity_ode_implicit_euler_scheme,
            0.01,
            np.array([1.0]),
            np.array([[10100 / 9899]]),
            np.array([10000 / 9899]),
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
        expected_new_state
    )


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize(
    """rk_scheme, stage, delta_t, old_time, old_state_perturbation,
    accumulated_stages_perturbation, linearization_args, expected_jacvec_product""",
    (
        [identity_ode_rk4_scheme, 0, 0.1, 0.0, 1.0, 1.0, {}, 1.1],
        [identity_ode_rk4_scheme, 3, 0.01, 0.0, 2.0, 10.0, {}, 2.1],
        [time_ode_implicit_euler_scheme, 0, 0.1, 1.0, 1.0, 1.0, {}, 0.0],
        [time_ode_implicit_euler_scheme, 0, 0.01, 10, 20.0, 111.1, {}, 0.0],
        [
            time_scaled_identity_ode_implicit_euler_scheme,
            0,
            0.1,
            1.0,
            1.0,
            1.0,
            {},
            1.21 / 0.89,
        ],
        [
            time_scaled_identity_ode_implicit_euler_scheme,
            0,
            0.01,
            10,
            20.0,
            111.1,
            {},
            211.32111 / 0.8999,
        ],
        [
            root_ode_two_stage_dirk_scheme,
            0,
            0.1,
            1.0,
            1.0,
            1.0,
            {
                "old_state": 0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                "acc_stages": 0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            },
            1.1 * 2 / (2 - 2**0.5),
        ],
        [
            root_ode_two_stage_dirk_scheme,
            0,
            0.01,
            1.0,
            1.0,
            1.0,
            {
                "old_state": 0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                "acc_stages": 0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            },
            1.01 * 2 / (2 - 2**0.5),
        ],
    ),
)
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
    """Tests the compute_stage_jacvec function."""
    # pylint: disable=too-many-arguments
    assert rk_scheme.compute_stage_jacvec(
        stage,
        delta_t,
        old_time,
        old_state_perturbation,
        accumulated_stages_perturbation,
        **linearization_args,
    ) == pytest.approx(expected_jacvec_product)


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize(
    """rk_scheme, stage, delta_t, old_time, joined_perturbation, linearization_args,
    expected_jacvec_product""",
    (
        [identity_ode_rk4_scheme, 0, 0.1, 0.0, 1.0, {}, (1.0, 0.1)],
        [identity_ode_rk4_scheme, 3, 0.01, 0.0, 2.0, {}, (2.0, 0.02)],
        [time_ode_implicit_euler_scheme, 0, 0.1, 1.0, 1.0, {}, (0.0, 0.0)],
        [time_ode_implicit_euler_scheme, 0, 0.01, 10, 111.1, {}, (0.0, 0.0)],
        [
            time_scaled_identity_ode_implicit_euler_scheme,
            0,
            0.1,
            1.0,
            1.0,
            {},
            (1.1 / 0.89, 0.11 / 0.89),
        ],
        [
            time_scaled_identity_ode_implicit_euler_scheme,
            0,
            0.01,
            10,
            111.1,
            {},
            (1112.111 / 0.8999, 11.12111 / 0.8999),
        ],
        [
            root_ode_two_stage_dirk_scheme,
            0,
            0.1,
            1.0,
            1.0,
            {
                "old_state": 0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                "acc_stages": 0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            },
            (2 / (2 - 2**0.5), 0.1 * 2 / (2 - 2**0.5)),
        ],
        [
            root_ode_two_stage_dirk_scheme,
            0,
            0.01,
            1.0,
            1.0,
            {
                "old_state": 0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                "acc_stages": 0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            },
            (2 / (2 - 2**0.5), 0.01 * 2 / (2 - 2**0.5)),
        ],
    ),
)
def test_compute_stage_transposed_jacvec(
    rk_scheme: RungeKuttaScheme,
    stage: int,
    delta_t: float,
    old_time: float,
    joined_perturbation: np.ndarray,
    linearization_args: dict,
    expected_jacvec_product: tuple[np.ndarray, np.ndarray],
):
    """Tests the compute_stage_tranposed_jacvec function."""
    assert rk_scheme.compute_stage_transposed_jacvec(
        stage, delta_t, old_time, joined_perturbation, **linearization_args
    ) == pytest.approx(expected_jacvec_product)


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize(
    """rk_scheme, stage, new_state_perturbation, accumulated_stages_perturbation_field,
    expected_joined_perturbation""",
    (
        [
            identity_ode_rk4_scheme,
            2,
            np.array([2.0]),
            np.array([[0.0], [0.0], [0.0], [1.0]]),
            np.array([5 / 3]),
        ],
        [
            identity_ode_rk4_scheme,
            0,
            np.array([1.0]),
            np.array([[0.0], [0.25], [0.5], [1.0]]),
            np.array([7 / 24]),
        ],
        [
            root_ode_two_stage_dirk_scheme,
            0,
            np.array([2.0]),
            np.array([[0.0], [1.0]]),
            np.array([3 * 2**-0.5]),
        ],
    ),
)
def test_join_new_state_and_accumulated_stages_perturbations(
    rk_scheme: RungeKuttaScheme,
    stage: int,
    new_state_perturbation: np.ndarray,
    accumulated_stages_perturbation_field: np.ndarray,
    expected_joined_perturbation: np.ndarray,
):
    """Tests the join_new_state_and_accumulated_stages_perturbations function"""
    assert rk_scheme.join_perturbations(
        stage, new_state_perturbation, accumulated_stages_perturbation_field
    ) == pytest.approx(expected_joined_perturbation)


@pytest.mark.rk
@pytest.mark.rk_scheme
@pytest.mark.parametrize(
    """rk_scheme, delta_t, new_state_perturbation, stage_perturbation_field,
    expected_old_state_perturbation""",
    (
        [
            identity_ode_rk4_scheme,
            0.1,
            np.array([1.0]),
            np.array([[1.0], [21 / 20], [421 / 400], [4421 / 4000]]),
            np.array([56831 / 40000]),
        ],
        [
            time_scaled_identity_ode_implicit_euler_scheme,
            0.01,
            np.array([1.0]),
            np.array([[10100 / 9899]]),
            np.array([1000000 / 989900]),
        ],
    ),
)
def test_compute_step_transposed_jacvec(
    rk_scheme: RungeKuttaScheme,
    delta_t: float,
    new_state_perturbation: np.ndarray,
    stage_perturbation_field: np.ndarray,
    expected_old_state_perturbation: np.ndarray,
):
    """Tests the compute_step_tranposed_jacvec function."""
    assert rk_scheme.compute_step_transposed_jacvec(
        delta_t, new_state_perturbation, stage_perturbation_field
    ) == pytest.approx(expected_old_state_perturbation)
