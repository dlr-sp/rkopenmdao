"""Tests for the Runge-Kutta integration core."""

# pylint: disable=unused-argument
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pytest

from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.discretized_ode.discretized_ode import (
    DiscretizedODE,
    DiscretizedODELinearizationPoint,
    DiscretizedODEInputState,
    DiscretizedODEResultState,
)
from rkopenmdao.runge_kutta_scheme import RungeKuttaScheme
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    embedded_second_order_two_stage_sdirk as two_stage_dirk,
    runge_kutta_four,
)


@dataclass
class SimpleODELinearizationPoint(DiscretizedODELinearizationPoint):
    """
    Minimal linearization point implementation for the ODEs used in the tests of this
    file.

    Parameters
    ----------
    time: float
        Time of the simulation at the caching point.
    ode_input: np.ndarray
        Parts of the ODE input that need to be cached for linearization.
    """

    time: float
    ode_input: np.ndarray

    def to_numpy_array(self) -> np.ndarray:
        array = np.zeros(1 + self.ode_input.size)
        array[0] = self.time
        array[1:] = self.ode_input
        return array

    def from_numpy_array(self, array: np.ndarray) -> None:
        self.time = array[0]
        self.ode_input = array[1:]


class IdentityODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = x(t).
    """

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        stage_update = (ode_input.step_input + step_size * ode_input.stage_input) / (
            1.0 - step_size * stage_factor
        )
        stage_state = stage_update.copy()
        return DiscretizedODEResultState(stage_update, stage_state, np.zeros(0))

    # This is a linear ODE, there is no need for saving linearization data
    def get_linearization_point(self) -> DiscretizedODELinearizationPoint:
        pass

    def set_linearization_point(
        self, linearization_state: DiscretizedODELinearizationPoint
    ) -> None:
        pass

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        return self.compute_update(
            ode_input_perturbation,
            step_size,
            stage_factor,
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        step_input_pert = (
            ode_result_perturbation.stage_update + ode_result_perturbation.stage_state
        ) / (1 - step_size * stage_factor)
        stage_output_pert = step_size * step_input_pert
        return DiscretizedODEInputState(
            step_input_pert, stage_output_pert, np.zeros(0), 0.0
        )

    def compute_state_norm(self, state: DiscretizedODEResultState):
        return np.abs(state.stage_state)


class TimeODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = t.
    """

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        stage_update = np.array([ode_input.time])
        stage_state = ode_input.step_input + step_size * (
            ode_input.stage_input + stage_factor * stage_update
        )
        return DiscretizedODEResultState(stage_update, stage_state, 0.0)

    def get_linearization_point(self) -> DiscretizedODELinearizationPoint:
        pass

    def set_linearization_point(
        self, linearization_state: DiscretizedODELinearizationPoint
    ) -> None:
        pass

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        return self.compute_update(
            ode_input_perturbation,
            step_size,
            stage_factor,
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        step_input_pert = ode_result_perturbation.stage_state
        stage_input_pert = step_size * ode_result_perturbation.stage_state
        time_pert = (
            ode_result_perturbation.stage_update
            * step_size
            * stage_factor
            * ode_result_perturbation.stage_state
        )
        return DiscretizedODEInputState(
            step_input_pert, stage_input_pert, np.zeros(0), time_pert
        )

    def compute_state_norm(self, state: DiscretizedODEResultState):
        return np.abs(state.stage_state)


@dataclass
class TimeScaledIdentityODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = t*x(t).
    """

    _cached_linearization: SimpleODELinearizationPoint = field(
        default_factory=lambda: SimpleODELinearizationPoint(0.0, np.zeros(2))
    )

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        cached_input = np.zeros(2)
        cached_input[0] = ode_input.step_input
        cached_input[1] = ode_input.stage_input
        self._cached_linearization = SimpleODELinearizationPoint(
            ode_input.time, cached_input
        )
        stage_state = (ode_input.step_input + step_size * ode_input.stage_input) / (
            1 - ode_input.time * step_size * stage_factor
        )
        stage_update = ode_input.time * stage_state
        return DiscretizedODEResultState(stage_update, stage_state, np.zeros(0))

    def get_linearization_point(self) -> SimpleODELinearizationPoint:
        return self._cached_linearization

    def set_linearization_point(
        self, linearization_state: SimpleODELinearizationPoint
    ) -> None:
        self._cached_linearization = linearization_state

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        time = self._cached_linearization.time
        step_input = self._cached_linearization.ode_input[0]
        stage_input = self._cached_linearization.ode_input[1]

        inv_divisor = 1 / (1 - time * step_size * stage_factor)

        stage_update_pert = (
            time
            * (
                ode_input_perturbation.step_input
                + step_size * ode_input_perturbation.stage_input
            )
            * inv_divisor
        ) + (step_input + step_size * stage_input) * inv_divisor**2

        stage_state_pert = (
            ode_input_perturbation.step_input
            + step_size * ode_input_perturbation.stage_input
        ) * inv_divisor + (
            step_input + step_size * stage_input
        ) * step_size * stage_factor * inv_divisor**2

        return DiscretizedODEResultState(
            stage_update_pert, stage_state_pert, np.zeros(0)
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        time = self._cached_linearization.time
        step_input = self._cached_linearization.ode_input[0]
        stage_input = self._cached_linearization.ode_input[1]

        inv_divisor = 1 / (1 - time * step_size * stage_factor)

        step_input_pert = (
            time * ode_result_perturbation.stage_update
            + ode_result_perturbation.stage_state
        ) * inv_divisor
        stage_input_pert = step_size * step_input_pert
        time_pert = (
            (step_input + step_size * stage_input)
            * inv_divisor**2
            * (
                ode_result_perturbation.stage_update
                + step_size * stage_factor * ode_result_perturbation.stage_state
            )
        )
        return DiscretizedODEInputState(
            step_input_pert, stage_input_pert, np.zeros(0), time_pert
        )

    def compute_state_norm(self, state: DiscretizedODEResultState):
        return np.abs(state.stage_state)


class ParameterODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = b, with b a time independent
    parameter.
    """

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        stage_update = ode_input.independent_input
        stage_output = ode_input.step_input + step_size * (
            ode_input.stage_input * stage_factor * stage_update
        )

        return DiscretizedODEResultState(stage_update, stage_output, np.zeros(0))

    def get_linearization_point(self) -> DiscretizedODELinearizationPoint:
        pass

    def set_linearization_point(
        self, linearization_state: DiscretizedODELinearizationPoint
    ) -> None:
        pass

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        stage_update_pert = ode_input_perturbation.independent_input
        stage_output_pert = ode_input_perturbation.step_input + step_size * (
            ode_input_perturbation.stage_input + stage_factor * stage_update_pert
        )
        return DiscretizedODEResultState(
            stage_update_pert, stage_output_pert, np.zeros(0)
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        step_input_pert = ode_result_perturbation.stage_state
        stage_input_pert = step_size * step_input_pert
        independent_input_pert = (
            ode_result_perturbation.stage_update
            + step_size * stage_factor * ode_result_perturbation.stage_state
        )

        return DiscretizedODEInputState(
            step_input_pert, stage_input_pert, independent_input_pert, 0.0
        )

    def compute_state_norm(self, state: DiscretizedODEResultState):
        return np.abs(state.stage_state)


@dataclass
class RootODE(DiscretizedODE):
    """
    Discretized ODE implementation for the ODE x'(t) = sqrt(x(t)).
    """

    _cached_linearization: SimpleODELinearizationPoint = field(
        default_factory=lambda: SimpleODELinearizationPoint(0.0, np.zeros(2))
    )

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        cached_input = np.zeros(2)
        cached_input[0] = ode_input.step_input
        cached_input[1] = ode_input.stage_input
        self._cached_linearization = SimpleODELinearizationPoint(0.0, cached_input)

        stage_update = 0.5 * step_size * stage_factor + np.sqrt(
            0.25 * step_size**2 * stage_factor**2
            + ode_input.step_input
            + step_size * ode_input.stage_input
        )
        stage_output = ode_input.step_input + step_size * (
            ode_input.stage_input + stage_factor * stage_update
        )

        return DiscretizedODEResultState(stage_update, stage_output, np.zeros(0))

    def get_linearization_point(self) -> SimpleODELinearizationPoint:
        return self._cached_linearization

    def set_linearization_point(
        self, linearization_state: SimpleODELinearizationPoint
    ) -> None:
        self._cached_linearization = linearization_state

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        step_input = self._cached_linearization.ode_input[0]
        stage_input = self._cached_linearization.ode_input[1]

        inv_divisor = 0.5 / np.sqrt(
            0.25 * step_size**2 * stage_factor**2 + step_input + step_size * stage_input
        )

        stage_update_pert = inv_divisor * (
            ode_input_perturbation.step_input
            + step_size * ode_input_perturbation.stage_input
        )
        stage_output_pert = (1 + 0.5 * step_size * stage_factor * inv_divisor) * (
            ode_input_perturbation.step_input
            + step_size * ode_input_perturbation.stage_input
        )
        return DiscretizedODEResultState(
            stage_update_pert, stage_output_pert, np.zeros(0)
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        step_input = self._cached_linearization.ode_input[0]
        stage_input = self._cached_linearization.ode_input[1]

        inv_divisor = 0.5 / np.sqrt(
            0.25 * step_size**2 * stage_factor**2 + step_input + step_size * stage_input
        )

        step_input_pert = (
            ode_result_perturbation.stage_update * inv_divisor
            + ode_result_perturbation.stage_state
            * (1 + step_size * stage_factor * inv_divisor)
        )
        stage_input_pert = step_size * step_input_pert

        return DiscretizedODEInputState(
            step_input_pert, stage_input_pert, np.zeros(0), 0.0
        )

    def compute_state_norm(self, state: DiscretizedODEResultState):
        return np.abs(state.stage_state)


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
    result = rk_scheme.compute_step(
        delta_t, old_state, stage_field, remaining_time=delta_t
    )
    assert result[0] == pytest.approx(expected_new_state)
    assert result[1] == delta_t
    assert result[2]
    assert np.isnan(result[3])


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
            SimpleODELinearizationPoint(1.1, np.array([0.0, 0.0])),
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
            SimpleODELinearizationPoint(10.01, np.array([0.0, 0.0])),
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
            SimpleODELinearizationPoint(
                0.0,
                np.array(
                    [
                        0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                        0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                    ]
                ),
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
            SimpleODELinearizationPoint(
                0.0,
                np.array(
                    [
                        0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                        0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                    ]
                ),
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
            SimpleODELinearizationPoint(1.1, np.array([0.0, 0.0])),
            (1.1 / 0.89, 0.11 / 0.89),
        ],
        [
            TimeScaledIdentityODE,
            implicit_euler,
            0,
            0.01,
            10,
            111.1,
            SimpleODELinearizationPoint(10.01, np.array([0.0, 0.0])),
            (1112.111 / 0.8999, 11.12111 / 0.8999),
        ],
        [
            RootODE,
            two_stage_dirk,
            0,
            0.1,
            1.0,
            1.0,
            SimpleODELinearizationPoint(
                0.0,
                np.array(
                    [
                        0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                        0.9 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                    ]
                ),
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
            SimpleODELinearizationPoint(
                0.0,
                np.array(
                    [
                        0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                        0.99 * ((2.0 - np.sqrt(2.0)) / 2.0) ** 2 / 4,
                    ]
                ),
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
