"""
Solving the heat equation in openMDAO via the Runge-Kutta-integrator.In this version, the heat equation is split in two
via a domain decomposition two emulate a multidisciplinary problem.
"""

from typing import List
import numpy as np
import openmdao.api as om

from heatequation.boundary import BoundaryCondition
from heatequation.split_heat_group import create_split_heat_group
from rkopenmdao.runge_kutta.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)
from rkopenmdao.runge_kutta.integration_control import IntegrationControl

from rkopenmdao.runge_kutta.butcher_tableaux import (
    third_order_third_weak_stage_order_four_stage_dirk,
)

from rkopenmdao.runge_kutta.functional_coefficients import (
    FunctionalCoefficients,
)


class HeatAverageOnSplit(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("points_per_direction", types=int)

    def setup(self):
        points_per_direction = self.options["points_per_direction"]
        points_x = points_per_direction // 2 + 1
        self.add_input(
            "heat_1",
            shape=points_per_direction * points_x,
            tags=["postproc_input_var", "heat_1"],
        )
        self.add_input(
            "heat_2",
            shape=points_per_direction * points_x,
            tags=["postproc_input_var", "heat_2"],
        )

        self.add_output(
            "heat_split_line_average",
            shape=1,
            tags=["postproc_output_var", "heat_split_line_average"],
        )

    def compute(self, inputs, outputs):
        points_per_direction = self.options["points_per_direction"]
        points_x = points_per_direction // 2 + 1
        split_line = 0.5 * (
            inputs["heat_1"][points_x - 1 : points_per_direction * points_x : points_x]
            + inputs["heat_2"][0 : points_per_direction * points_x : points_x]
        )
        outputs["heat_split_line_average"] = np.mean(split_line)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        points_per_direction = self.options["points_per_direction"]
        points_x = points_per_direction // 2 + 1

        if mode == "fwd":
            d_split_line = 0.5 * (
                d_inputs["heat_1"][
                    points_x - 1 : points_per_direction * points_x : points_x
                ]
                + d_inputs["heat_2"][0 : points_per_direction * points_x : points_x]
            )
            d_outputs["heat_split_line_average"] += np.mean(d_split_line)
        elif mode == "rev":
            d_inputs["heat_1"][
                points_x - 1 : points_per_direction * points_x : points_x
            ] += (0.5 * d_outputs["heat_split_line_average"] / points_per_direction)
            d_inputs["heat_2"][0 : points_per_direction * points_x : points_x] += (
                0.5 * d_outputs["heat_split_line_average"] / points_per_direction
            )


class TrapezoidalCoefficients(FunctionalCoefficients):
    def __init__(self, num_steps: int, delta: float, quantity_list: list) -> None:
        self.num_steps = num_steps
        self.delta = delta
        self.quantity_list = quantity_list

    def list_quantities(self) -> List[str]:
        return self.quantity_list

    def get_coefficient(self, time_step: int, quantity: str) -> float:
        if quantity in self.quantity_list:
            if time_step == 0 or time_step == self.num_steps:
                return 0.5 * self.delta
            else:
                return self.delta


if __name__ == "__main__":
    points_per_direction = 51
    delta_x = (points_per_direction - 1) ** -1
    points_x = points_per_direction // 2 + 1

    boundary_condition_1 = BoundaryCondition(
        upper=lambda t, x, y: 0.0,
        lower=lambda t, x, y: 0.0,
        left=lambda t, x, y: 0.0,
    )
    boundary_condition_2 = BoundaryCondition(
        upper=lambda t, x, y: 0.0,
        lower=lambda t, x, y: 0.0,
        right=lambda t, x, y: 0.0,
    )

    butcher_tableau = third_order_third_weak_stage_order_four_stage_dirk

    integration_control = IntegrationControl(0.0, 100, 1e-3)

    inner_prob = om.Problem()

    heat_group = create_split_heat_group(
        points_per_direction,
        boundary_condition_1,
        boundary_condition_2,
        lambda t, x, y: 0.0,
        lambda t, x, y: 0.0,
        1.0,
        1.0,
        lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + 1,
        lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + 1,
        integration_control,
        {"tol": 1e-15, "atol": "legacy"},
    )

    inner_prob.model.add_subsystem("heat_group", heat_group)

    inner_prob.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=False, iprint=2
    )
    inner_prob.model.linear_solver = om.PETScKrylov(maxiter=100, restart=10)
    inner_prob.model.linear_solver.precon = om.LinearBlockJac(iprint=-1, maxiter=2)

    postproc_problem = om.Problem()

    postproc_problem.model.add_subsystem(
        "HeatSplitLineAverager",
        HeatAverageOnSplit(points_per_direction=points_per_direction),
    )

    outer_prob = om.Problem()
    outer_prob.model.add_subsystem(
        "RK_Integrator",
        RungeKuttaIntegrator(
            time_stage_problem=inner_prob,
            postprocessing_problem=postproc_problem,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["heat_1", "heat_2"],
            postprocessing_quantities=["heat_split_line_average"],
            write_out_distance=1,
            write_file="heat_equ_om_split_functional.h5",
            functional_coefficients=TrapezoidalCoefficients(
                integration_control.num_steps, delta_x, ["heat_split_line_average"]
            ),
        ),
        promotes_inputs=["heat_1_initial", "heat_2_initial"],
    )

    outer_prob.setup()

    outer_prob.run_model()

    print(outer_prob["RK_Integrator.heat_split_line_average_final"])
    print(outer_prob["RK_Integrator.heat_split_line_average_functional"])
