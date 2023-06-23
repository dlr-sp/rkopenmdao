from typing import Callable
import importlib
import pytest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_check_totals

from runge_kutta_openmdao.heatequation.boundary import BoundaryCondition
from runge_kutta_openmdao.heatequation.split_heat_group import create_split_heat_group
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl

zero_boundary_left = BoundaryCondition(
    upper=lambda t, x, y: 0.0,
    lower=lambda t, x, y: 0.0,
    left=lambda t, x, y: 0.0,
)

zeros_boundary_right = BoundaryCondition(
    upper=lambda t, x, y: 0.0,
    lower=lambda t, x, y: 0.0,
    right=lambda t, x, y: 0.0,
)

integration_control_1 = IntegrationControl(0.0, 1, 0, 1e-4)


gmres_args_without_precon = {
    "tol": 1e-15,
    "atol": "legacy",
}


@pytest.mark.heatequ
@pytest.mark.heatequ_split_heat_group
@pytest.mark.parametrize(
    "points_per_direction, boundary_condition_1, boundary_condition_2, inhomogeneity, diffusivity, initial_condition, integration_control, delta_t, butcher_diagonal_element, gmres_args",
    (
        [
            11,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.01,
            0.0,
            gmres_args_without_precon,
        ],
        [
            11,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.01,
            (2 - 2**0.5) / 2,
            gmres_args_without_precon,
        ],
        [
            11,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.01,
            1.0,
            gmres_args_without_precon,
        ],
        [
            21,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0025,
            0.0,
            gmres_args_without_precon,
        ],
        [
            21,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0025,
            (2 - 2**0.5) / 2,
            gmres_args_without_precon,
        ],
        [
            21,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0025,
            1.0,
            gmres_args_without_precon,
        ],
        [
            51,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0004,
            0.0,
            gmres_args_without_precon,
        ],
        [
            51,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0004,
            (2 - 2**0.5) / 2,
            gmres_args_without_precon,
        ],
        [
            51,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0004,
            1.0,
            gmres_args_without_precon,
        ],
    ),
)
def test_split_heat_group_partials(
    points_per_direction: int,
    boundary_condition_1: BoundaryCondition,
    boundary_condition_2: BoundaryCondition,
    inhomogeneity: Callable[[float, float, float], float],
    diffusivity: float,
    initial_condition: Callable[[float, float], float],
    integration_control: IntegrationControl,
    delta_t: float,
    butcher_diagonal_element: float,
    gmres_args: dict,
):
    integration_control.delta_t = delta_t
    integration_control_1.butcher_diagonal_element = butcher_diagonal_element
    split_heat_group = create_split_heat_group(
        points_per_direction,
        boundary_condition_1,
        boundary_condition_2,
        inhomogeneity,
        inhomogeneity,
        diffusivity,
        diffusivity,
        initial_condition,
        initial_condition,
        integration_control,
        gmres_args,
        nonlinear_solver=om.NewtonSolver(
            solve_subsystems=True
        ),  # Need to specify the solver here again for some reason, using the default solvers doesn't work after the first test
        linear_solver=om.PETScKrylov(),
    )
    example_problem = om.Problem(split_heat_group)
    example_problem.setup()
    example_problem.run_model()
    data = example_problem.check_partials(step=1e-1)
    assert_check_partials(data)


# TODO: look into this further
@pytest.mark.heatequ
@pytest.mark.heatequ_split_heat_group
@pytest.mark.parametrize(
    "points_per_direction, boundary_condition_1, boundary_condition_2, inhomogeneity, diffusivity, initial_condition, integration_control, delta_t, butcher_diagonal_element, gmres_args",
    (
        [
            11,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.01,
            0.0,
            gmres_args_without_precon,
        ],
        [
            11,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.01,
            (2 - 2**0.5) / 2,
            gmres_args_without_precon,
        ],
        [
            11,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.01,
            1.0,
            gmres_args_without_precon,
        ],
        [
            21,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0025,
            0.0,
            gmres_args_without_precon,
        ],
        [
            21,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0025,
            (2 - 2**0.5) / 2,
            gmres_args_without_precon,
        ],
        [
            21,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0025,
            1.0,
            gmres_args_without_precon,
        ],
        [
            51,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0004,
            0.0,
            gmres_args_without_precon,
        ],
        [
            51,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0004,
            (2 - 2**0.5) / 2,
            gmres_args_without_precon,
        ],
        [
            51,
            zero_boundary_left,
            zeros_boundary_right,
            lambda t, x, y: 0,
            1.0,
            lambda x, y: 0.0,
            integration_control_1,
            0.0004,
            1.0,
            gmres_args_without_precon,
        ],
    ),
)
def test_split_heat_group_totals(
    points_per_direction: int,
    boundary_condition_1: BoundaryCondition,
    boundary_condition_2: BoundaryCondition,
    inhomogeneity: Callable[[float, float, float], float],
    diffusivity: float,
    initial_condition: Callable[[float, float], float],
    integration_control: IntegrationControl,
    delta_t: float,
    butcher_diagonal_element: float,
    gmres_args: dict,
):
    integration_control.delta_t = delta_t
    integration_control_1.butcher_diagonal_element = butcher_diagonal_element
    split_heat_group = create_split_heat_group(
        points_per_direction,
        boundary_condition_1,
        boundary_condition_2,
        inhomogeneity,
        inhomogeneity,
        diffusivity,
        diffusivity,
        initial_condition,
        initial_condition,
        integration_control,
        gmres_args,
        nonlinear_solver=om.NewtonSolver(
            iprint=2, solve_subsystems=True, maxiter=30, atol=1e-10, rtol=1e-10
        ),
        linear_solver=om.PETScKrylov(iprint=0, atol=1e-12, rtol=1e-12),
    )
    example_problem = om.Problem(split_heat_group)
    example_problem.setup()
    example_problem.run_model()
    output_list = example_problem.model.list_outputs()
    data = example_problem.check_totals(
        of=[
            "stage_heat_1",
            "stage_slope_1",
            "stage_heat_2",
            "stage_slope_2",
        ],
        wrt=[
            "heat_1",
            "accumulated_stages_1",
            "heat_2",
            "accumulated_stages_2",
        ],
        step=1e-1,
        form="central",
    )
    assert_check_totals(data)
