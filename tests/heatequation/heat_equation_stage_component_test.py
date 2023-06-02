import numpy as np
import pytest
import itertools

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from scipy.sparse.linalg import LinearOperator, eigs

from runge_kutta_openmdao.heatequation.heatequation_stage_component import (
    HeatEquationStageComponent,
)
from runge_kutta_openmdao.heatequation.heatequation import HeatEquation
from runge_kutta_openmdao.heatequation.boundary import BoundaryCondition
from runge_kutta_openmdao.heatequation.domain import Domain

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl


implicit_euler = ButcherTableau(np.array([[0.5]]), np.array([1.0]), np.array([0.5]))

gamma = (2.0 - np.sqrt(2.0)) / 2.0
two_stage_dirk = ButcherTableau(
    np.array(
        [
            [gamma, 0.0],
            [1 - gamma, gamma],
        ]
    ),
    np.array([1 - gamma, gamma]),
    np.array([gamma, 1.0]),
)

runge_kutta_four = ButcherTableau(
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


@pytest.mark.heatequ
@pytest.mark.heatequ_stage_component
@pytest.mark.parametrize(
    "points_per_direction, butcher_tableau, delta_t",
    itertools.product(
        [11, 21, 51], [implicit_euler, two_stage_dirk, runge_kutta_four], [1e-2, 1e-3, 1e-4]
    ),
)
def test_heat_equation_stage_component_zero_vector(
    points_per_direction, butcher_tableau: ButcherTableau, delta_t
):
    domain = Domain([0.0, 1.0], [0.0, 1.0], points_per_direction, points_per_direction)

    boundary_condition = BoundaryCondition(
        upper=lambda t, x, y: 0.0,
        lower=lambda t, x, y: 0.0,
        left=lambda t, x, y: 0.0,
        right=lambda t, x, y: 0.0,
    )

    heat_equation = HeatEquation(
        domain,
        lambda t, x, y: 0.0,
        boundary_condition,
        1.0,
        lambda x, y: 1,
        {"tol": 1e-10, "atol": "legacy"},
    )

    integration_control = IntegrationControl(0.0, 1, 1, delta_t)

    test_prob = om.Problem()
    test_prob.model.add_subsystem(
        "heat_equation_stage_component",
        HeatEquationStageComponent(
            heat_equation=heat_equation, integration_control=integration_control
        ),
    )
    test_prob.setup()
    test_prob.set_val("heat_equation_stage_component.heat", np.zeros(points_per_direction**2))
    test_prob.set_val(
        "heat_equation_stage_component.accumulated_stages", np.zeros(points_per_direction**2)
    )
    for stage in range(butcher_tableau.number_of_stages()):
        integration_control.butcher_diagonal_element = butcher_tableau.butcher_matrix[stage, stage]
        test_prob.run_model()
        assert test_prob.get_val(
            "heat_equation_stage_component.result_stage_slope"
        ) == pytest.approx(np.zeros(points_per_direction**2))


@pytest.mark.heatequ
@pytest.mark.heatequ_stage_component
@pytest.mark.parametrize(
    "points_per_direction, butcher_tableau, delta_t, boundary_segment",
    itertools.product(
        [11, 21, 51],
        [implicit_euler, two_stage_dirk, runge_kutta_four],
        [1e-2, 1e-3, 1e-4],
        ["upper", "lower", "left", "right"],
    ),
)
def test_heat_equation_stage_component_zero_vector_with_one_boundary(
    points_per_direction, butcher_tableau: ButcherTableau, delta_t, boundary_segment
):
    domain = Domain([0.0, 1.0], [0.0, 1.0], points_per_direction, points_per_direction)

    if boundary_segment == "upper":
        boundary_condition = BoundaryCondition(
            lower=lambda t, x, y: 0.0, left=lambda t, x, y: 0.0, right=lambda t, x, y: 0.0
        )
    elif boundary_segment == "lower":
        boundary_condition = BoundaryCondition(
            upper=lambda t, x, y: 0.0, left=lambda t, x, y: 0.0, right=lambda t, x, y: 0.0
        )
    elif boundary_segment == "left":
        boundary_condition = BoundaryCondition(
            upper=lambda t, x, y: 0.0, lower=lambda t, x, y: 0.0, right=lambda t, x, y: 0.0
        )
    elif boundary_segment == "right":
        boundary_condition = BoundaryCondition(
            upper=lambda t, x, y: 0.0, left=lambda t, x, y: 0.0, lower=lambda t, x, y: 0.0
        )

    heat_equation = HeatEquation(
        domain,
        lambda t, x, y: 0.0,
        boundary_condition,
        1.0,
        lambda x, y: 1,
        {"tol": 1e-10, "atol": "legacy"},
    )

    integration_control = IntegrationControl(0.0, 1, 1, delta_t)

    test_prob = om.Problem()
    test_prob.model.add_subsystem(
        "heat_equation_stage_component",
        HeatEquationStageComponent(
            heat_equation=heat_equation,
            integration_control=integration_control,
            shared_boundary=[boundary_segment],
        ),
    )
    test_prob.setup()
    test_prob.set_val("heat_equation_stage_component.heat", np.zeros(points_per_direction**2))
    test_prob.set_val(
        "heat_equation_stage_component.accumulated_stages", np.zeros(points_per_direction**2)
    )
    test_prob.set_val(
        "heat_equation_stage_component.boundary_segment_" + boundary_segment,
        np.zeros(points_per_direction),
    )
    for stage in range(butcher_tableau.number_of_stages()):
        integration_control.butcher_diagonal_element = butcher_tableau.butcher_matrix[stage, stage]
        test_prob.run_model()
        assert test_prob.get_val(
            "heat_equation_stage_component.result_stage_slope"
        ) == pytest.approx(np.zeros(points_per_direction**2))


@pytest.mark.heatequ
@pytest.mark.heatequ_stage_component
@pytest.mark.parametrize(
    "points_per_direction, butcher_tableau, delta_t",
    itertools.product(
        [11, 21, 51], [implicit_euler, two_stage_dirk, runge_kutta_four], [1e-2, 1e-3, 1e-4]
    ),
)
def test_heat_equation_stage_component_zero_vector_with_all_boundary(
    points_per_direction, butcher_tableau: ButcherTableau, delta_t
):
    domain = Domain([0.0, 1.0], [0.0, 1.0], points_per_direction, points_per_direction)

    boundary_condition = BoundaryCondition()

    heat_equation = HeatEquation(
        domain,
        lambda t, x, y: 0.0,
        boundary_condition,
        1.0,
        lambda x, y: 1,
        {"tol": 1e-10, "atol": "legacy"},
    )

    integration_control = IntegrationControl(0.0, 1, 1, delta_t)

    test_prob = om.Problem()
    test_prob.model.add_subsystem(
        "heat_equation_stage_component",
        HeatEquationStageComponent(
            heat_equation=heat_equation,
            integration_control=integration_control,
            shared_boundary=["lower", "left", "right", "upper"],
        ),
    )
    test_prob.setup()
    test_prob.set_val("heat_equation_stage_component.heat", np.zeros(points_per_direction**2))
    test_prob.set_val(
        "heat_equation_stage_component.accumulated_stages", np.zeros(points_per_direction**2)
    )
    test_prob.set_val(
        "heat_equation_stage_component.boundary_segment_left",
        np.zeros(points_per_direction),
    )
    test_prob.set_val(
        "heat_equation_stage_component.boundary_segment_right",
        np.zeros(points_per_direction),
    )
    test_prob.set_val(
        "heat_equation_stage_component.boundary_segment_lower",
        np.zeros(points_per_direction),
    )
    test_prob.set_val(
        "heat_equation_stage_component.boundary_segment_upper",
        np.zeros(points_per_direction),
    )
    for stage in range(butcher_tableau.number_of_stages()):
        integration_control.butcher_diagonal_element = butcher_tableau.butcher_matrix[stage, stage]
        test_prob.run_model()
        assert test_prob.get_val(
            "heat_equation_stage_component.result_stage_slope"
        ) == pytest.approx(np.zeros(points_per_direction**2))


# TODO: think about test cases where the solution of run_model given certain inputs is known


@pytest.mark.heatequ
@pytest.mark.heatequ_stage_component
@pytest.mark.parametrize(
    "points_per_direction, butcher_tableau, delta_t",
    itertools.product(
        [11, 21, 51], [implicit_euler, two_stage_dirk, runge_kutta_four], [1e-2, 1e-3, 1e-4]
    ),
)
def test_heat_equation_stage_component_partials(
    points_per_direction, butcher_tableau: ButcherTableau, delta_t
):
    domain = Domain([0.0, 1.0], [0.0, 1.0], points_per_direction, points_per_direction)

    boundary_condition = BoundaryCondition(
        upper=lambda t, x, y: 0.0,
        lower=lambda t, x, y: 0.0,
        left=lambda t, x, y: 0.0,
        right=lambda t, x, y: 0.0,
    )

    heat_equation = HeatEquation(
        domain,
        lambda t, x, y: 0.0,
        boundary_condition,
        1.0,
        lambda x, y: 1,
        {"tol": 1e-10, "atol": "legacy"},
    )

    integration_control = IntegrationControl(0.0, 1, 1, delta_t)

    test_prob = om.Problem()
    test_prob.model.add_subsystem(
        "heat_equation_stage_component",
        HeatEquationStageComponent(
            heat_equation=heat_equation, integration_control=integration_control
        ),
    )
    test_prob.setup()
    for stage in range(butcher_tableau.number_of_stages()):
        integration_control.butcher_diagonal_element = butcher_tableau.butcher_matrix[stage, stage]
        test_prob.run_model()
        # The component models a linear system, so we don't need a small step size form finite differences.
        # This additionally prevents cancellation errors in the fd-computation.
        test_data = test_prob.check_partials(out_stream=None, step=1e-1)
        assert_check_partials(test_data)


@pytest.mark.heatequ
@pytest.mark.heatequ_stage_component
@pytest.mark.parametrize(
    "points_per_direction, butcher_tableau, delta_t, boundary_segment",
    itertools.product(
        [11, 21, 51],
        [implicit_euler, two_stage_dirk, runge_kutta_four],
        [1e-2, 1e-3, 1e-4],
        ["upper", "lower", "left", "right"],
    ),
)
def test_heat_equation_stage_component_partials_with_one_boundary(
    points_per_direction, butcher_tableau: ButcherTableau, delta_t, boundary_segment
):
    domain = Domain([0.0, 1.0], [0.0, 1.0], points_per_direction, points_per_direction)

    if boundary_segment == "upper":
        boundary_condition = BoundaryCondition(
            lower=lambda t, x, y: 0.0, left=lambda t, x, y: 0.0, right=lambda t, x, y: 0.0
        )
    elif boundary_segment == "lower":
        boundary_condition = BoundaryCondition(
            upper=lambda t, x, y: 0.0, left=lambda t, x, y: 0.0, right=lambda t, x, y: 0.0
        )
    elif boundary_segment == "left":
        boundary_condition = BoundaryCondition(
            upper=lambda t, x, y: 0.0, lower=lambda t, x, y: 0.0, right=lambda t, x, y: 0.0
        )
    elif boundary_segment == "right":
        boundary_condition = BoundaryCondition(
            upper=lambda t, x, y: 0.0, left=lambda t, x, y: 0.0, lower=lambda t, x, y: 0.0
        )

    heat_equation = HeatEquation(
        domain,
        lambda t, x, y: 0.0,
        boundary_condition,
        1.0,
        lambda x, y: 1,
        {"tol": 1e-10, "atol": "legacy"},
    )

    integration_control = IntegrationControl(0.0, 1, 1, delta_t)

    test_prob = om.Problem()
    test_prob.model.add_subsystem(
        "heat_equation_stage_component",
        HeatEquationStageComponent(
            heat_equation=heat_equation,
            integration_control=integration_control,
            shared_boundary=[boundary_segment],
        ),
    )
    test_prob.setup()
    for stage in range(butcher_tableau.number_of_stages()):
        integration_control.butcher_diagonal_element = butcher_tableau.butcher_matrix[stage, stage]
        test_prob.run_model()
        # The component models a linear system, so we don't need a small step size form finite differences.
        # This additionally prevents cancellation errors in the fd-computation.
        test_data = test_prob.check_partials(out_stream=None, step=1e-1)
        assert_check_partials(test_data)
