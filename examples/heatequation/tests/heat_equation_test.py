import pytest
import numpy as np

from heatequation import heat_equation

# TODO: Design test cases


def test_set_time(time, example_heat_equation, expected_pde_inhom):
    example_heat_equation.set_time(time)
    assert example_heat_equation.time == time
    assert np.array_equal(
        expected_pde_inhom, example_heat_equation.inhomogenity_vector.pde_inhomogenity
    )


def test_evaluate_stationary(coord, example_heat_equation, expected_result):
    assert np.array_equal(expected_result, example_heat_equation.stationary_func(coord))


def test_evaluate_d_stationary(coord, example_heat_equation, expected_result):
    assert np.array_equal(
        expected_result, example_heat_equation.stationary_d_func(coord)
    )


def test_time_stage_residual(
    example_heat_equation,
    expected_residual_value,
    time,
    delta_t,
    butcher_tableau,
    stage_num,
    old_value,
    current_guess,
    previous_stages,
):
    assert np.array_equal(
        expected_residual_value,
        example_heat_equation.heat_equation_time_stage_residual(
            time,
            delta_t,
            butcher_tableau,
            stage_num,
            old_value,
            current_guess,
            **previous_stages
        ),
    )


def test_time_stage_d_residual_fwd(
    example_heat_equation,
    expected_d_residual_value,
    delta_t,
    butcher_tableau,
    stage_num,
    current_d_guess,
):
    assert np.array_equal(
        expected_d_residual_value,
        example_heat_equation.heat_equation_time_stage_d_residual_fwd(
            delta_t, butcher_tableau, stage_num, current_d_guess
        ),
    )


def test_time_stage_d_residual_rev(
    example_heat_equation,
    expected_d_residual_value,
    delta_t,
    butcher_tableau,
    stage_num,
    current_d_guess,
):
    assert np.array_equal(
        expected_d_residual_value,
        example_heat_equation.heat_equation_time_stage_d_residual_rev(
            delta_t, butcher_tableau, stage_num, current_d_guess
        ),
    )


def test_assemble_stage_heat(
    example_heat_equation,
    expected_value,
    stage_num,
    butcher_tableau,
    old_value,
    stage_values,
):
    assert np.array_equal(
        expected_value,
        example_heat_equation.assemble_stage_heat(
            stage_num, butcher_tableau, old_value, **stage_values
        ),
    )


def test_solve_time_stage_residual(
    example_heat_equation,
    expected_result,
    time,
    delta_t,
    butcher_tableau,
    stage_num,
    old_value,
    current_guess,
    previous_stages,
):
    assert expected_result == pytest.approx(
        example_heat_equation.solve_heat_equation_time_stage_residual(
            time,
            delta_t,
            butcher_tableau,
            stage_num,
            old_value,
            current_guess,
            **previous_stages
        )
    )


def test_solve_time_stage_d_residual_fwd(
    example_heat_equation,
    expected_result,
    delta_t,
    butcher_tableau,
    stage_num,
    rhs,
    previous_stages,
):
    assert expected_result == pytest.approx(
        example_heat_equation.solve_heat_equation_time_stage_residual_fwd(
            delta_t, butcher_tableau, stage_num, rhs, **previous_stages
        )
    )


def test_solve_time_stage_d_residual_rev(
    example_heat_equation,
    expected_result,
    delta_t,
    butcher_tableau,
    stage_num,
    rhs,
    previous_stages,
):
    assert expected_result == pytest.approx(
        example_heat_equation.solve_heat_equation_time_stage_residual_rev(
            delta_t, butcher_tableau, stage_num, rhs, **previous_stages
        )
    )


def test_heat_equation_time_step(
    example_heat_equation,
    expected_result,
    time,
    delta_t,
    old_value,
    butcher_tableau,
):
    assert expected_result == pytest.approx(
        example_heat_equation.solve_heat_equation(
            time, delta_t, old_value, butcher_tableau
        )
    )


def test_solve_heat_equation(
    example_heat_equation,
    solution,
    butcher_tableau,
    delta_t,
    number_of_steps,
):
    num_solution = example_heat_equation.solve_heat_equation(
        butcher_tableau, delta_t, number_of_steps
    )
    for time, val in num_solution.items():
        assert solution(time) == pytest.approx(val)
