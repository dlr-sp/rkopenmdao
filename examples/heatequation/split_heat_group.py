from typing import Callable

import openmdao.api as om
from openmdao.solvers.solver import NonlinearSolver, LinearSolver

from .boundary import BoundaryCondition
from .domain import Domain
from .flux_component import FluxComponent
from .heatequation import HeatEquation
from .heatequation_stage_component import (
    HeatEquationStageComponent,
)
from rkopenmdao.stage_value_component import StageValueComponent
from rkopenmdao.integration_control import IntegrationControl


def create_split_heat_group(
    points_per_direction: int,
    boundary_condition_1: BoundaryCondition,
    boundary_condition_2: BoundaryCondition,
    inhomogeneity_1: Callable[[float, float, float], float],
    inhomogeneity_2: Callable[[float, float, float], float],
    diffusivity_1: float,
    diffusivity_2: float,
    initial_condition_1: Callable[[float, float], float],
    initial_condition_2: Callable[[float, float], float],
    integration_control: IntegrationControl,
    gmres_args: dict,
    nonlinear_solver: NonlinearSolver = om.NewtonSolver(solve_subsystems=True),
    linear_solver: LinearSolver = om.PETScKrylov(),
) -> om.Group:
    """Creates openMDAO group that contains everything necessary to model two heat equations on a split domain."""
    points_x = points_per_direction // 2 + 1
    delta_x = 1.0 / (points_per_direction - 1)
    domain_half_1 = Domain([0.0, 0.5], [0.0, 1.0], points_x, points_per_direction)
    domain_half_2 = Domain([0.5, 1.0], [0.0, 1.0], points_x, points_per_direction)

    heat_equation_1 = HeatEquation(
        domain_half_1,
        inhomogeneity_1,
        boundary_condition_1,
        diffusivity_1,
        initial_condition_1,
        gmres_args,
    )

    heat_equation_2 = HeatEquation(
        domain_half_2,
        inhomogeneity_2,
        boundary_condition_2,
        diffusivity_2,
        initial_condition_2,
        gmres_args,
    )

    split_heat_group = om.Group()
    heat_group_1 = om.Group()
    heat_group_1.add_subsystem(
        "heat_component_1",
        HeatEquationStageComponent(
            heat_equation=heat_equation_1,
            shared_boundary=["right"],
            domain_num=1,
            integration_control=integration_control,
        ),
        promotes_inputs=[
            ("heat", "heat_1"),
            ("accumulated_stages", "accumulated_stages_1"),
        ],
        promotes_outputs=[
            ("result_stage_slope", "heat_slope_1"),
        ],
    )
    heat_group_1.add_subsystem(
        "heat_accumulator",
        StageValueComponent(integration_control=integration_control),
        promotes_inputs=[
            ("old_value", "heat_1"),
            ("acc_stages", "accumulated_stages_1"),
            ("stage_slope", "heat_slope_1"),
        ],
        promotes_outputs=[("stage_value", "stage_heat_1")],
    )
    heat_group_1.set_input_defaults("heat_1", val=heat_equation_1.initial_vector)
    heat_group_1.nonlinear_solver = om.NonlinearRunOnce()
    heat_group_1.linear_solver = om.LinearRunOnce()
    split_heat_group.add_subsystem("heat_1", heat_group_1, promotes=["*"])

    heat_group_2 = om.Group()
    heat_group_2.add_subsystem(
        "heat_component_2",
        HeatEquationStageComponent(
            heat_equation=heat_equation_2,
            shared_boundary=["left"],
            domain_num=2,
            integration_control=integration_control,
        ),
        promotes_inputs=[
            ("heat", "heat_2"),
            ("accumulated_stages", "accumulated_stages_2"),
        ],
        promotes_outputs=[
            ("result_stage_slope", "heat_slope_2"),
        ],
    )
    heat_group_2.add_subsystem(
        "heat_accumulator",
        StageValueComponent(integration_control=integration_control),
        promotes_inputs=[
            ("old_value", "heat_2"),
            ("acc_stages", "accumulated_stages_2"),
            ("stage_slope", "heat_slope_2"),
        ],
        promotes_outputs=[("stage_value", "stage_heat_2")],
    )
    heat_group_2.set_input_defaults("heat_2", val=heat_equation_2.initial_vector)
    heat_group_2.nonlinear_solver = om.NonlinearRunOnce()
    heat_group_2.linear_solver = om.LinearRunOnce()

    split_heat_group.add_subsystem("heat_2", heat_group_2, promotes=["*"])

    split_heat_group.add_subsystem(
        "flux_comp",
        FluxComponent(
            delta=delta_x, shape=points_per_direction, orientation="vertical"
        ),
    )

    left_boundary_indices_1 = domain_half_2.boundary_indices("left") + 1

    right_boundary_indices_1 = domain_half_1.boundary_indices("right") - 1

    split_heat_group.connect(
        "stage_heat_1",
        "flux_comp.left_side",
        src_indices=right_boundary_indices_1,
    )
    split_heat_group.connect(
        "stage_heat_2",
        "flux_comp.right_side",
        src_indices=left_boundary_indices_1,
    )

    split_heat_group.connect(
        "flux_comp.flux", "heat_component_1.boundary_segment_right"
    )
    split_heat_group.connect(
        "flux_comp.reverse_flux", "heat_component_2.boundary_segment_left"
    )

    split_heat_group.nonlinear_solver = nonlinear_solver
    split_heat_group.linear_solver = linear_solver

    return split_heat_group
