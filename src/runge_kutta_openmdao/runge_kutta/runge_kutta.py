from typing import Callable
import numpy as np
from scipy.optimize import root

from .butcher_tableau import ButcherTableau

# from scipy.optimize import newton_krylov


class RungeKuttaStage:
    def __init__(
        self,
        butcher_tableau: ButcherTableau,
        func: Callable[[float, np.ndarray], np.ndarray],
        delta_t: float,
    ):
        self.butcher_tableau = butcher_tableau
        self.func = func
        self.delta_t = delta_t

    def stage_residual(
        self,
        stage_num: int,
        old_value: np.ndarray,
        old_time: float,
        current_stage: np.ndarray,
        **other_stages,
    ):
        time = old_time + self.butcher_tableau.butcher_time_stages[stage_num]
        value = old_value.copy()
        iteration_range = (
            range(self.butcher_tableau.number_of_stages() - 1)
            if self.butcher_tableau.is_implicit()
            else range(stage_num - 1)
        )
        for i in iteration_range:
            if i == stage_num:
                pass
            else:
                value += (
                    self.delta_t
                    * self.butcher_tableau.butcher_matrix[stage_num, i]
                    * other_stages[str(i)]
                )
        value += (
            self.delta_t
            * self.butcher_tableau.butcher_matrix[stage_num, stage_num]
            * current_stage
        )
        return current_stage - self.func(time, value)

    def solve_stage_residual(
        self,
        stage_num: int,
        old_value: np.ndarray,
        old_time: float,
        initial_guess: np.ndarray,
        **other_stages,
    ):
        if self.butcher_tableau.is_explicit():
            return initial_guess - self.stage_residual(
                stage_num, old_value, old_time, initial_guess, **other_stages
            )
        else:
            result = root(
                lambda stage_val: self.stage_residual(
                    stage_num, old_value, old_time, stage_val, **other_stages
                ),
                initial_guess,
                method="krylov",
                options={
                    "ftol:": 1e-10,
                    "fatol": 1e-10,
                    "jac_options": {
                        "method": "gmres",
                        "inner_maxiter": 1000,
                        "inner_atol": "legacy",
                        "inner_tol": 1e-12,
                    },
                },
            )
            return result.x


class RungeKuttaStep:
    def __init__(
        self,
        butcher_tableau: ButcherTableau,
        func: Callable[[float, np.ndarray], np.ndarray],
        delta_t: float,
    ):
        self.butcher_tableau = butcher_tableau
        self.delta_t = delta_t
        self.runge_kutta_stage = RungeKuttaStage(self.butcher_tableau, func, delta_t)

    def compute_step(self, old_value: np.ndarray, old_time: float):
        new_value = old_value
        stage_values = {}
        if self.butcher_tableau.is_implicit():
            raise NotImplementedError(
                "General implicit schemes are currently not implemented"
            )
        else:
            for i in range(self.butcher_tableau.number_of_stages()):
                stage_value = self.runge_kutta_stage.solve_stage_residual(
                    i, old_value, old_time, old_value.copy(), **stage_values
                )
                stage_values[str(i)] = stage_value

        for i in range(self.butcher_tableau.number_of_stages()):
            new_value += (
                self.delta_t
                * self.butcher_tableau.butcher_weight_vector[i]
                * stage_values[str(i)]
            )

        return new_value


class RungeKuttaScheme:
    def __init__(
        self,
        butcher_tableau: ButcherTableau,
        func: Callable[[float, np.ndarray], np.ndarray],
        initial_value: np.ndarray,
        delta_t: float,
        start_time=0.0,
    ):
        self.runge_kutta_step = RungeKuttaStep(butcher_tableau, func, delta_t)
        self.current_value = initial_value
        self.current_time = start_time

    def stage_residual(
        self,
        current_stage: int,
        input_value: np.ndarray,
        **other_stages,
    ) -> np.ndarray:
        return self.runge_kutta_step.runge_kutta_stage.stage_residual(
            current_stage,
            self.current_value,
            self.current_time,
            input_value,
            **other_stages,
        )

    def compute_stage(
        self,
        current_stage: int,
        initial_guess: np.ndarray,
        **other_stages,
    ) -> np.ndarray:
        return self.runge_kutta_step.runge_kutta_stage.solve_stage_residual(
            current_stage,
            self.current_value,
            self.current_time,
            initial_guess,
            **other_stages,
        )

    # TODO: Add function that can solve for general implicit RK-Schemes

    def compute_step(self) -> np.ndarray:
        return self.runge_kutta_step.compute_step(self.current_value, self.current_time)

    def perform_step(self) -> np.ndarray:
        self.current_value = self.compute_step()
        self.current_time += self.runge_kutta_step.delta_t
        return (self.current_time, self.current_value)


if __name__ == "__main__":
    # TODO: Maybe predefine certain common RK-schemes for other to use
    explicit_euler_tableau = ButcherTableau(
        np.array([[0.0]]), np.array([1.0]), np.array([0.0])
    )

    explicit_euler = RungeKuttaScheme(
        explicit_euler_tableau,
        lambda t, x: np.array([np.exp(-0.1 * t)]),
        np.array([-10.0]),
        0.1,
    )

    explicit_midpoint_tableau = ButcherTableau(
        np.array([[0.0, 0.0], [0.5, 0.0]]), np.array([0.0, 1.0]), np.array([0.0, 0.5])
    )

    explicit_midpoint = RungeKuttaScheme(
        explicit_midpoint_tableau,
        lambda t, x: np.array([np.exp(-0.1 * t)]),
        np.array([-10.0]),
        0.1,
    )

    implicit_midpoint_tableau = ButcherTableau(
        np.array([[0.5]]), np.array([1.0]), np.array([0.5])
    )
    implicit_midpoint = RungeKuttaScheme(
        implicit_midpoint_tableau,
        lambda t, x: np.array([np.exp(-0.1 * t)]),
        np.array([-10.0]),
        0.1,
    )

    implicit_four_stage_tableau = ButcherTableau(
        np.array(
            [
                [0.5, 0.0, 0.0, 0.0],
                [0.167, 0.5, 0.0, 0.0],
                [-0.5, 0.5, 0.5, 0.0],
                [1.5, -1.5, 0.5, 0.5],
            ]
        ),
        np.array([1.5, -1.5, 0.5, 0.5]),
        np.array([0.5, 0.667, 0.5, 1.0]),
    )
    implicit_four_stage = RungeKuttaScheme(
        implicit_four_stage_tableau,
        lambda t, x: np.array([np.exp(-0.1 * t)]),
        np.array([-10.0]),
        0.1,
    )
    for step in range(2000):
        print(
            explicit_euler.perform_step()[1],
            explicit_midpoint.perform_step()[1],
            implicit_midpoint.perform_step()[1],
            implicit_four_stage.perform_step()[1],
        )
