from math import isclose
import numpy as np
import pytest


from runge_kutta_openmdao.runge_kutta import runge_kutta
from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau


@pytest.mark.rk
@pytest.mark.parametrize(
    "tableau, stages, kind",
    [
        (
            runge_kutta.ButcherTableau(
                np.array([[0.0]]), np.array([1.0]), np.array([0.0])
            ),
            1,
            "explicit",
        ),
        (
            runge_kutta.ButcherTableau(
                np.array([[0.25, -0.25], [0.25, 0.417]]),
                np.array([0.25, 0.75]),
                np.array([0.0, 0.667]),
            ),
            2,
            "implicit",
        ),
        (
            runge_kutta.ButcherTableau(
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
            ),
            4,
            "diagonally implicit",
        ),
    ],
)
def test_butcher_tableau(tableau: runge_kutta.ButcherTableau, stages, kind):
    assert tableau.number_of_stages() == stages
    if kind == "explicit":
        assert tableau.is_explicit()
    elif kind == "diagonally implicit":
        assert tableau.is_diagonally_implicit()
    else:
        assert tableau.is_implicit()


@pytest.mark.rk
@pytest.mark.parametrize(
    "res_value, stage_class, stage_num, prev_step, time, guess, prev_stages",
    [
        (
            1.0,
            runge_kutta.RungeKuttaStage(
                runge_kutta.ButcherTableau(
                    np.array([[1.0]]), np.array([1.0]), np.array([1.0])
                ),
                lambda t, x: -2 * x[0],
                0.5,
            ),
            0,
            np.array([1.0]),
            0.0,
            np.array([-0.5]),
            {},
        ),
        (
            0.0,
            runge_kutta.RungeKuttaStage(
                runge_kutta.ButcherTableau(
                    np.array([[1.0]]), np.array([1.0]), np.array([1.0])
                ),
                lambda t, x: -2 * x[0],
                0.5,
            ),
            0,
            np.array([1.0]),
            0.0,
            np.array([-1.0]),
            {},
        ),
        (
            -1.0,
            runge_kutta.RungeKuttaStage(
                runge_kutta.ButcherTableau(
                    np.array([[1.0]]), np.array([1.0]), np.array([1.0])
                ),
                lambda t, x: -2 * x[0],
                0.5,
            ),
            0,
            np.array([1.0]),
            0.0,
            np.array([-1.5]),
            {},
        ),
    ],
)
def test_stage_residual(
    res_value, stage_class, stage_num, prev_step, time, guess, prev_stages
):
    assert stage_class.stage_residual(
        stage_num, prev_step, time, guess, **prev_stages
    ) == pytest.approx(res_value)


@pytest.mark.rk
@pytest.mark.parametrize(
    "solution, stage_class, stage_num, prev_step, time, initial_guess, prev_stages",
    [
        (
            -1.0,
            runge_kutta.RungeKuttaStage(
                runge_kutta.ButcherTableau(
                    np.array([[1.0]]), np.array([1.0]), np.array([1.0])
                ),
                lambda t, x: -2 * x[0],
                0.5,
            ),
            0,
            np.array([1.0]),
            0.0,
            np.array([-0.5]),
            {},
        ),
        (
            -1.0,
            runge_kutta.RungeKuttaStage(
                runge_kutta.ButcherTableau(
                    np.array([[1.0]]), np.array([1.0]), np.array([1.0])
                ),
                lambda t, x: -2 * x[0],
                0.5,
            ),
            0,
            np.array([1.0]),
            0.0,
            np.array([-1.0]),
            {},
        ),
        (
            -1.0,
            runge_kutta.RungeKuttaStage(
                runge_kutta.ButcherTableau(
                    np.array([[1.0]]), np.array([1.0]), np.array([1.0])
                ),
                lambda t, x: -2 * x[0],
                0.5,
            ),
            0,
            np.array([1.0]),
            0.0,
            np.array([-1.5]),
            {},
        ),
    ],
)
def test_stage_residual_solution(
    solution, stage_class, stage_num, prev_step, time, initial_guess, prev_stages
):
    assert stage_class.solve_stage_residual(
        stage_num, prev_step, time, initial_guess, **prev_stages
    ) == pytest.approx(solution)
    assert stage_class.stage_residual(
        stage_num, prev_step, time, solution, **prev_stages
    ) == pytest.approx(0.0)


@pytest.mark.rk
@pytest.mark.parametrize(
    "step_class, prev_step, time, result",
    [
        (
            runge_kutta.RungeKuttaStep(
                runge_kutta.ButcherTableau(
                    np.array([[1.0]]), np.array([1.0]), np.array([1.0])
                ),
                lambda t, x: -2 * x[0],
                0.5,
            ),
            np.array([1.0]),
            0.0,
            np.array([0.5]),
        ),
        (
            runge_kutta.RungeKuttaStep(
                runge_kutta.ButcherTableau(
                    np.array([[1.0]]), np.array([1.0]), np.array([1.0])
                ),
                lambda t, x: -2 * x[0],
                0.25,
            ),
            np.array([1.0]),
            0.0,
            np.array([2.0 / 3.0]),
        ),
        (
            runge_kutta.RungeKuttaStep(
                runge_kutta.ButcherTableau(
                    np.array([[1.0]]), np.array([1.0]), np.array([1.0])
                ),
                lambda t, x: -2 * x[0],
                0.25,
            ),
            np.array([2.0 / 3.0]),
            0.0,
            np.array([4.0 / 9.0]),
        ),
    ],
)
def test_step_value(step_class, prev_step, time, result):
    assert step_class.compute_step(prev_step, time) == pytest.approx(result)


##TODO reconsider how to test convergence order


# @pytest.mark.rk
# @pytest.mark.parametrize(
#     "rk_scheme, expected_order, step_num, solution_func",
#     [
#         (
#             runge_kutta.RungeKuttaScheme(
#                 runge_kutta.ButcherTableau(
#                     np.array([[1.0]]), np.array([1.0]), np.array([1.0])
#                 ),
#                 lambda t, x: -2 * x[0],
#                 np.array([1.0]),
#                 0.5,
#             ),
#             1,
#             10,
#             lambda t: np.exp(-2 * t),
#         ),
#         (
#             runge_kutta.RungeKuttaScheme(
#                 runge_kutta.ButcherTableau(
#                     np.array([[0.5]]),
#                     np.array([1.0]),
#                     np.array([0.5]),
#                 ),
#                 lambda t, x: -2 * x[0],
#                 np.array([1.0]),
#                 0.5,
#             ),
#             2,
#             10,
#             lambda t: np.exp(-2 * t),
#         ),
#     ],
# )
# def test_runge_kutta_convergence(rk_scheme, expected_order, step_num, solution_func):
#     init_value = rk_scheme.current_value.copy()
#     init_delta_t = rk_scheme.runge_kutta_step.delta_t
#     init_time = rk_scheme.current_time

#     first_residual = 0.0
#     for i in range(step_num):
#         if i == step_num - 1:
#             first_residual = np.abs(
#                 solution_func(init_time + init_delta_t * (step_num))
#                 - rk_scheme.perform_step()[1][0]
#             )
#         else:
#             rk_scheme.perform_step()

#     rk_scheme.current_value = init_value.copy()
#     rk_scheme.runge_kutta_step.delta_t /= 2
#     rk_scheme.runge_kutta_step.runge_kutta_stage.delta_t /= 2
#     rk_scheme.curent_time = init_time

#     second_residual = 0.0
#     for i in range(2 * step_num):
#         if i == 2 * step_num - 1:
#             second_residual = abs(
#                 solution_func(init_time + init_delta_t * (step_num))
#                 - rk_scheme.perform_step()[1][0]
#             )
#         else:
#             rk_scheme.perform_step()
#     assert np.log(second_residual / first_residual) / np.log(
#         0.5
#     ) > expected_order or np.log(second_residual / first_residual) / np.log(
#         0.5
#     ) == pytest.approx(
#         expected_order
#     )

#     rk_scheme.current_value = init_value.copy()
#     rk_scheme.runge_kutta_step.delta_t /= 2
#     rk_scheme.runge_kutta_step.runge_kutta_stage.delta_t /= 2
#     rk_scheme.curent_time = init_time

#     third_residual = 0.0
#     for i in range(4 * step_num):
#         if i == 4 * step_num - 1:
#             third_residual = abs(
#                 solution_func(init_time + init_delta_t * (step_num))
#                 - rk_scheme.perform_step()[1][0]
#             )
#         else:
#             rk_scheme.perform_step()
#     assert np.log(third_residual / second_residual) / np.log(
#         0.5
#     ) > expected_order or np.log(third_residual / second_residual) / np.log(
#         0.5
#     ) == pytest.approx(
#         expected_order
#     )
#     assert np.log(third_residual / first_residual) / np.log(
#         0.25
#     ) > expected_order or np.log(third_residual / first_residual) / np.log(
#         0.25
#     ) == pytest.approx(
#         expected_order
#     )
