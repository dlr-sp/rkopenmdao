"""Tests for the Runge-Kutta integration core."""

# pylint: disable=unused-argument
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE
from rkopenmdao.runge_kutta_scheme import RungeKuttaScheme
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    embedded_second_order_two_stage_sdirk as two_stage_dirk,
    runge_kutta_four,
)


class IdentityODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = x(t).
    """

    CacheType = None

    def compute_update(
        self,
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
        time: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stage_update = (step_input + step_size * stage_input) / (
            1.0 - step_size * stage_factor
        )
        stage_state = stage_update.copy()
        return stage_update, stage_state, np.zeros(0)

    # This is a linear ODE, there is no need for saving linearization data
    def export_linearization(self) -> CacheType:
        pass

    def import_linearization(self, cache: CacheType) -> CacheType:
        pass

    def compute_update_derivative(
        self,
        step_input_pert: np.ndarray,
        stage_input_pert: np.ndarray,
        independent_input_pert: np.ndarray,
        time_pert: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.compute_update(
            step_input_pert,
            stage_input_pert,
            independent_input_pert,
            0.0,
            step_size,
            stage_factor,
        )

    def compute_update_adjoint_derivative(
        self,
        stage_update_pert: np.ndarray,
        stage_output_pert: np.ndarray,
        independent_output_pert: np.ndarray,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        step_input_pert = (stage_update_pert + stage_output_pert) / (
            1 - step_size * stage_factor
        )
        stage_output_pert = step_size * step_input_pert
        return step_input_pert, stage_output_pert, np.zeros(0), 0.0


class TimeODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = t.
    """

    CacheType = None

    def compute_update(
        self,
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
        time: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stage_update = np.array([time])
        stage_state = step_input + step_size * (
            stage_input + stage_factor * stage_update
        )
        return stage_update, stage_state, np.zeros(0)

    def export_linearization(self) -> CacheType:
        pass

    def import_linearization(self, cache: CacheType) -> CacheType:
        pass

    def compute_update_derivative(
        self,
        step_input_pert: np.ndarray,
        stage_input_pert: np.ndarray,
        independent_input_pert: np.ndarray,
        time_pert: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.compute_update(
            step_input_pert,
            stage_input_pert,
            independent_input_pert,
            time_pert,
            step_size,
            stage_factor,
        )

    def compute_update_adjoint_derivative(
        self,
        stage_update_pert: np.ndarray,
        stage_output_pert: np.ndarray,
        independent_output_pert: np.ndarray,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        step_input_pert = stage_output_pert
        stage_input_pert = step_size * stage_output_pert
        time_pert = stage_update_pert * step_size * stage_factor * stage_output_pert
        return step_input_pert, stage_input_pert, np.zeros(0), time_pert


@dataclass
class TimeScaledIdentityODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = t*x(t).
    """

    CacheType = tuple[float, np.ndarray, np.ndarray]

    _cache: CacheType = (0.0, np.zeros(0), np.zeros(0))

    def compute_update(
        self,
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
        time: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._cache = (time, deepcopy(step_input), deepcopy(stage_input))
        stage_state = (step_input + step_size * stage_input) / (
            1 - time * step_size * stage_factor
        )
        stage_update = time * stage_state
        return stage_update, stage_state, np.zeros(0)

    def export_linearization(self) -> CacheType:
        return deepcopy(self._cache)

    def import_linearization(self, cache: CacheType) -> None:
        self._cache = cache

    def compute_update_derivative(
        self,
        step_input_pert: np.ndarray,
        stage_input_pert: np.ndarray,
        independent_input_pert: np.ndarray,
        time_pert: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time = self._cache[0]
        step_input = self._cache[1]
        stage_input = self._cache[2]

        inv_divisor = 1 / (1 - time * step_size * stage_factor)

        stage_update_pert = (
            time * (step_input_pert + step_size * stage_input_pert) * inv_divisor
        ) + (step_input + step_size * stage_input) * inv_divisor**2

        stage_state_pert = (
            step_input_pert + step_size * stage_input_pert
        ) * inv_divisor + (
            step_input + step_size * stage_input
        ) * step_size * stage_factor * inv_divisor**2

        return stage_update_pert, stage_state_pert, np.zeros(0)

    def compute_update_adjoint_derivative(
        self,
        stage_update_pert: np.ndarray,
        stage_output_pert: np.ndarray,
        independent_output_pert: np.ndarray,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        time = self._cache[0]
        step_input = self._cache[1]
        stage_input = self._cache[2]

        inv_divisor = 1 / (1 - time * step_size * stage_factor)

        step_input_pert = (time * stage_update_pert + stage_output_pert) * inv_divisor
        stage_input_pert = step_size * step_input_pert
        time_pert = (
            (step_input + step_size * stage_input)
            * inv_divisor**2
            * (stage_update_pert + step_size * stage_factor * stage_output_pert)
        )
        return step_input_pert, stage_input_pert, np.zeros(0), time_pert


class ParameterODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = b, with b a time independent
    parameter.
    """

    CacheType = None

    def compute_update(
        self,
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
        time: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stage_update = independent_input
        stage_output = step_input + step_size * (
            stage_input * stage_factor * stage_update
        )

        return stage_update, stage_output, np.zeros(0)

    def export_linearization(self) -> CacheType:
        pass

    def import_linearization(self, cache: CacheType) -> None:
        pass

    def compute_update_derivative(
        self,
        step_input_pert: np.ndarray,
        stage_input_pert: np.ndarray,
        independent_input_pert: np.ndarray,
        time_pert: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stage_update_pert = independent_input_pert
        stage_output_pert = step_input_pert + step_size * (
            stage_input_pert + stage_factor * stage_update_pert
        )
        return stage_update_pert, stage_output_pert, np.zeros(0)

    def compute_update_adjoint_derivative(
        self,
        stage_update_pert: np.ndarray,
        stage_output_pert: np.ndarray,
        independent_output_pert: np.ndarray,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        step_input_pert = stage_output_pert
        stage_input_pert = step_size * step_input_pert
        independent_input_pert = (
            stage_update_pert + step_size * stage_factor * stage_output_pert
        )

        return step_input_pert, stage_input_pert, independent_input_pert, 0.0


@dataclass
class RootODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = sqrt(x(t)).
    """

    CacheType = tuple[np.ndarray, np.ndarray]

    _cache = (np.zeros(0), np.zeros(0))

    def compute_update(
        self,
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
        time: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._cache = (deepcopy(step_input), deepcopy(stage_input))

        stage_update = 0.5 * step_size * stage_factor + np.sqrt(
            0.25 * step_size**2 * stage_factor**2 + step_input + step_size * stage_input
        )
        stage_output = step_input + step_size * (
            stage_input + stage_factor * stage_update
        )

        return stage_update, stage_output, np.zeros(0)

    def export_linearization(self) -> CacheType:
        return deepcopy(self._cache)

    def import_linearization(self, cache: CacheType) -> None:
        self._cache = cache

    def compute_update_derivative(
        self,
        step_input_pert: np.ndarray,
        stage_input_pert: np.ndarray,
        independent_input_pert: np.ndarray,
        time_pert: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        step_input = self._cache[0]
        stage_input = self._cache[1]

        inv_divisor = 0.5 / np.sqrt(
            0.25 * step_size**2 * stage_factor**2 + step_input + step_size * stage_input
        )

        stage_update_pert = inv_divisor * (
            step_input_pert + step_size * stage_input_pert
        )
        stage_output_pert = (1 + 0.5 * step_size * stage_factor * inv_divisor) * (
            step_input_pert + step_size * stage_input_pert
        )
        return stage_update_pert, stage_output_pert, np.zeros(0)

    def compute_update_adjoint_derivative(
        self,
        stage_update_pert: np.ndarray,
        stage_output_pert: np.ndarray,
        independent_output_pert: np.ndarray,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        step_input = self._cache[0]
        stage_input = self._cache[1]

        inv_divisor = 0.5 / np.sqrt(
            0.25 * step_size**2 * stage_factor**2 + step_input + step_size * stage_input
        )

        step_input_pert = stage_update_pert * inv_divisor + stage_output_pert * (
            1 + step_size * stage_factor * inv_divisor
        )
        stage_input_pert = step_size * step_input_pert

        return step_input_pert, stage_input_pert, np.zeros(0), 0.0


@pytest.mark.parametrize(
    """ode, butcher_tableau, stage, delta_t, old_time, old_state, accumulated_stages, 
    expected_stage""",
    (
        [IdentityODE, runge_kutta_four, 0, 0.1, 0.0, 1.0, 1.0, 1.1],
        [IdentityODE, runge_kutta_four, 3, 0.01, 0.0, 2.0, 10.0, 2.1],
        [TimeODE, implicit_euler, 0, 0.1, 1.0, 1.0, 1.0, 1.1],
        [TimeODE, implicit_euler, 0, 0.01, 10, 20.0, 111.1, 10.01],
        [
            TimeScaledIdentityODE,
            implicit_euler,
            0,
            0.1,
            1.0,
            1.0,
            1.0,
            1.21 / 0.89,
        ],
        [
            TimeScaledIdentityODE,
            implicit_euler,
            0,
            0.01,
            10,
            20.0,
            111.1,
            211.32111 / 0.8999,
        ],
        [
            RootODE,
            two_stage_dirk,
            0,
            0.1,
            1.0,
            0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            0.55 * ((2.0 - np.sqrt(2.0)) / 2.0),
        ],
        [
            RootODE,
            two_stage_dirk,
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
    ode: DiscretizedODE,
    butcher_tableau: ButcherTableau,
    stage: int,
    delta_t: float,
    old_time: float,
    old_state: np.ndarray,
    accumulated_stages: np.ndarray,
    expected_stage: np.ndarray,
):
    """Tests the compute_stage function."""
    rk_scheme = RungeKuttaScheme(butcher_tableau, ode())
    assert rk_scheme.compute_stage(
        stage, delta_t, old_time, old_state, accumulated_stages, 0.0
    ) == pytest.approx(expected_stage)


@pytest.mark.parametrize(
    """ode, butcher_tableau, param, expected_stage""",
    (
        [ParameterODE, implicit_euler, 1.0, 1.0],
        [ParameterODE, two_stage_dirk, 1.0, 1.0],
        [ParameterODE, implicit_euler, 2.0, 2.0],
        [ParameterODE, two_stage_dirk, 2.0, 2.0],
    ),
)
def test_compute_stage_with_param(
    ode: DiscretizedODE,
    butcher_tableau: ButcherTableau,
    param: float,
    expected_stage: np.ndarray,
):
    """Tests the compute_stage function."""
    rk_scheme = RungeKuttaScheme(butcher_tableau, ode())
    assert rk_scheme.compute_stage(0, 0.1, 1.0, 0.0, 0.0, param) == pytest.approx(
        expected_stage
    )


@pytest.mark.parametrize(
    "ode, butcher_tableau, stage, stage_field, expected_accumulated_stages",
    (
        [
            IdentityODE,
            runge_kutta_four,
            1,
            np.array([[1.0], [0.0], [0.0], [0.0]]),
            np.array([0.5]),
        ],
        [
            IdentityODE,
            runge_kutta_four,
            2,
            np.array([[1.0], [21 / 20], [0.0], [0.0]]),
            np.array([21 / 40]),
        ],
        [
            IdentityODE,
            runge_kutta_four,
            3,
            np.array([[1.0], [21 / 20], [421 / 400], [0.0]]),
            np.array([421 / 400]),
        ],
        [RootODE, two_stage_dirk, 1, np.array([[2.0]]), np.array([2**0.5])],
    ),
)
def test_compute_accumulated_stages(
    ode: DiscretizedODE,
    butcher_tableau: ButcherTableau,
    stage: int,
    stage_field: np.ndarray,
    expected_accumulated_stages: np.ndarray,
):
    """Tests the compute_accumulated_stages function."""
    rk_scheme = RungeKuttaScheme(butcher_tableau, ode())
    assert rk_scheme.compute_accumulated_stages(stage, stage_field) == pytest.approx(
        expected_accumulated_stages
    )


@pytest.mark.parametrize(
    "ode, butcher_tableau, delta_t, old_state, stage_field, expected_new_state",
    (
        [
            IdentityODE,
            runge_kutta_four,
            0.1,
            np.array([1.0]),
            np.array([[1.0], [21 / 20], [421 / 400], [4421 / 4000]]),
            np.array([26524.1 / 24000]),
        ],
        [
            TimeScaledIdentityODE,
            implicit_euler,
            0.01,
            np.array([1.0]),
            np.array([[10100 / 9899]]),
            np.array([10000 / 9899]),
        ],
    ),
)
def test_compute_step(
    ode: DiscretizedODE,
    butcher_tableau: ButcherTableau,
    delta_t: float,
    old_state: np.ndarray,
    stage_field: np.ndarray,
    expected_new_state: np.ndarray,
):
    """Tests the compute_step function."""
    rk_scheme = RungeKuttaScheme(butcher_tableau, ode())
    assert rk_scheme.compute_step(
        delta_t, old_state, stage_field, remaining_time=delta_t
    )[0] == pytest.approx(expected_new_state)


@pytest.mark.parametrize(
    """ode, butcher_tableau, stage, delta_t, old_time, old_state_perturbation,
    accumulated_stages_perturbation, linearization_cache, expected_jacvec_product""",
    (
        [IdentityODE, runge_kutta_four, 0, 0.1, 0.0, 1.0, 1.0, None, 1.1],
        [IdentityODE, runge_kutta_four, 3, 0.01, 0.0, 2.0, 10.0, None, 2.1],
        [TimeODE, implicit_euler, 0, 0.1, 1.0, 1.0, 1.0, None, 0.0],
        [TimeODE, implicit_euler, 0, 0.01, 10, 20.0, 111.1, None, 0.0],
        [
            TimeScaledIdentityODE,
            implicit_euler,
            0,
            0.1,
            1.0,
            1.0,
            1.0,
            (1.1, 0.0, 0.0),
            1.21 / 0.89,
        ],
        [
            TimeScaledIdentityODE,
            implicit_euler,
            0,
            0.01,
            10,
            20.0,
            111.1,
            (10.01, 0.0, 0.0),
            211.32111 / 0.8999,
        ],
        [
            RootODE,
            two_stage_dirk,
            0,
            0.1,
            1.0,
            1.0,
            1.0,
            (
                0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            ),
            1.1 * 2 / (2 - 2**0.5),
        ],
        [
            RootODE,
            two_stage_dirk,
            0,
            0.01,
            1.0,
            1.0,
            1.0,
            (
                0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            ),
            1.01 * 2 / (2 - 2**0.5),
        ],
    ),
)
def test_compute_stage_jacvec(
    ode: DiscretizedODE,
    butcher_tableau: ButcherTableau,
    stage: int,
    delta_t: float,
    old_time: float,
    old_state_perturbation: np.ndarray,
    accumulated_stages_perturbation: np.ndarray,
    linearization_cache,
    expected_jacvec_product: np.ndarray,
):
    """Tests the compute_stage_jacvec function."""
    # pylint: disable=too-many-arguments
    rk_scheme = RungeKuttaScheme(butcher_tableau, ode())
    assert rk_scheme.compute_stage_jacvec(
        stage,
        delta_t,
        old_state_perturbation,
        accumulated_stages_perturbation,
        0.0,
        linearization_cache,
    ) == pytest.approx(expected_jacvec_product)


@pytest.mark.parametrize(
    """ode, butcher_tableau, param_perturbation, expected_jacvec_product""",
    (
        [ParameterODE, implicit_euler, 1.0, 1.0],
        [ParameterODE, two_stage_dirk, 1.0, 1.0],
        [ParameterODE, implicit_euler, 2.0, 2.0],
        [ParameterODE, two_stage_dirk, 2.0, 2.0],
    ),
)
def test_compute_stage_jacvec_with_param(
    ode: DiscretizedODE,
    butcher_tableau: ButcherTableau,
    param_perturbation: float,
    expected_jacvec_product: np.ndarray,
):
    """Tests the compute_stage function."""
    rk_scheme = RungeKuttaScheme(butcher_tableau, ode())
    assert rk_scheme.compute_stage_jacvec(
        0, 0.1, 1.0, 0.0, param_perturbation
    ) == pytest.approx(expected_jacvec_product)


@pytest.mark.parametrize(
    """ode, butcher_tableau, stage, delta_t, old_time, joined_perturbation,
    linearization_cache, expected_jacvec_product""",
    (
        [IdentityODE, runge_kutta_four, 0, 0.1, 0.0, 1.0, None, (1.0, 0.1)],
        [IdentityODE, runge_kutta_four, 3, 0.01, 0.0, 2.0, None, (2.0, 0.02)],
        [TimeODE, implicit_euler, 0, 0.1, 1.0, 1.0, None, (0.0, 0.0)],
        [TimeODE, implicit_euler, 0, 0.01, 10, 111.1, None, (0.0, 0.0)],
        [
            TimeScaledIdentityODE,
            implicit_euler,
            0,
            0.1,
            1.0,
            1.0,
            (1.1, 0.0, 0.0),
            (1.1 / 0.89, 0.11 / 0.89),
        ],
        [
            TimeScaledIdentityODE,
            implicit_euler,
            0,
            0.01,
            10,
            111.1,
            (10.01, 0.0, 0.0),
            (1112.111 / 0.8999, 11.12111 / 0.8999),
        ],
        [
            RootODE,
            two_stage_dirk,
            0,
            0.1,
            1.0,
            1.0,
            (
                0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            ),
            (2 / (2 - 2**0.5), 0.1 * 2 / (2 - 2**0.5)),
        ],
        [
            RootODE,
            two_stage_dirk,
            0,
            0.01,
            1.0,
            1.0,
            (
                0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
            ),
            (2 / (2 - 2**0.5), 0.01 * 2 / (2 - 2**0.5)),
        ],
    ),
)
def test_compute_stage_transposed_jacvec(
    ode: DiscretizedODE,
    butcher_tableau: ButcherTableau,
    stage: int,
    delta_t: float,
    old_time: float,
    joined_perturbation: np.ndarray,
    linearization_cache,
    expected_jacvec_product: tuple[np.ndarray, np.ndarray],
):
    """Tests the compute_stage_transposed_jacvec function."""
    rk_scheme = RungeKuttaScheme(butcher_tableau, ode())
    assert rk_scheme.compute_stage_transposed_jacvec(
        stage, delta_t, joined_perturbation, linearization_cache
    )[0:2] == pytest.approx(expected_jacvec_product)


@pytest.mark.parametrize(
    """ode, butcher_tableau, joined_perturbation, expected_param_jacvec_product""",
    (
        [ParameterODE, implicit_euler, 1.0, 1.0],
        [ParameterODE, two_stage_dirk, 1.0, 1.0],
        [ParameterODE, implicit_euler, 2.0, 2.0],
        [ParameterODE, two_stage_dirk, 2.0, 2.0],
    ),
)
def test_compute_stage_transposed_jacvec_with_param(
    ode: DiscretizedODE,
    butcher_tableau: ButcherTableau,
    joined_perturbation: np.ndarray,
    expected_param_jacvec_product: np.ndarray,
):
    """Tests the compute_stage_tranposed_jacvec function."""
    rk_scheme = RungeKuttaScheme(butcher_tableau, ode())
    assert rk_scheme.compute_stage_transposed_jacvec(
        0,
        0.1,
        joined_perturbation,
    )[
        2
    ] == pytest.approx(expected_param_jacvec_product)


@pytest.mark.parametrize(
    """ode, butcher_tableau, stage, new_state_perturbation,
    accumulated_stages_perturbation_field, expected_joined_perturbation""",
    (
        [
            IdentityODE,
            runge_kutta_four,
            2,
            np.array([2.0]),
            np.array([[0.0], [0.0], [0.0], [1.0]]),
            np.array([5 / 3]),
        ],
        [
            IdentityODE,
            runge_kutta_four,
            0,
            np.array([1.0]),
            np.array([[0.0], [0.25], [0.5], [1.0]]),
            np.array([7 / 24]),
        ],
        [
            RootODE,
            two_stage_dirk,
            0,
            np.array([2.0]),
            np.array([[0.0], [1.0]]),
            np.array([3 * 2**-0.5]),
        ],
    ),
)
def test_join_new_state_and_accumulated_stages_perturbations(
    ode: DiscretizedODE,
    butcher_tableau: ButcherTableau,
    stage: int,
    new_state_perturbation: np.ndarray,
    accumulated_stages_perturbation_field: np.ndarray,
    expected_joined_perturbation: np.ndarray,
):
    """Tests the join_new_state_and_accumulated_stages_perturbations function"""
    rk_scheme = RungeKuttaScheme(butcher_tableau, ode())
    assert rk_scheme.join_perturbations(
        stage, new_state_perturbation, accumulated_stages_perturbation_field
    ) == pytest.approx(expected_joined_perturbation)


@pytest.mark.parametrize(
    """ode, butcher_tableau, delta_t, new_state_perturbation, stage_perturbation_field,
    expected_old_state_perturbation""",
    (
        [
            IdentityODE,
            runge_kutta_four,
            0.1,
            np.array([1.0]),
            np.array([[1.0], [21 / 20], [421 / 400], [4421 / 4000]]),
            np.array([56831 / 40000]),
        ],
        [
            TimeScaledIdentityODE,
            implicit_euler,
            0.01,
            np.array([1.0]),
            np.array([[10100 / 9899]]),
            np.array([1000000 / 989900]),
        ],
    ),
)
def test_compute_step_transposed_jacvec(
    ode: DiscretizedODE,
    butcher_tableau: ButcherTableau,
    delta_t: float,
    new_state_perturbation: np.ndarray,
    stage_perturbation_field: np.ndarray,
    expected_old_state_perturbation: np.ndarray,
):
    """Tests the compute_step_tranposed_jacvec function."""
    rk_scheme = RungeKuttaScheme(butcher_tableau, ode())
    assert rk_scheme.compute_step_transposed_jacvec(
        delta_t, new_state_perturbation, stage_perturbation_field
    ) == pytest.approx(expected_old_state_perturbation)
