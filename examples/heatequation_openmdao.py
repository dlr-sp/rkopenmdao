"""Solving the heat equation in openMDAO via the Runge-Kutta-integrator."""

import numpy as np
import openmdao.api as om

from rkopenmdao.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)
from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.butcher_tableaux import (
    third_order_third_weak_stage_order_four_stage_dirk,
)

from .heatequation.boundary import BoundaryCondition
from .heatequation.domain import Domain
from .heatequation.heatequation import HeatEquation
from .heatequation.heatequation_stage_component import (
    HeatEquationStageComponent,
)


if __name__ == "__main__":
    points_per_direction = 51
    delta_x = (points_per_direction - 1) ** -1
    delta_t = 1e-3

    domain = Domain([0, 1], [0, 1], points_per_direction, points_per_direction)

    boundary_condition = BoundaryCondition(
        upper=lambda t, x, y: 0.0,
        lower=lambda t, x, y: 0.0,
        right=lambda t, x, y: 0.0,
        left=lambda t, x, y: 0.0,
    )

    butcher_tableau = third_order_third_weak_stage_order_four_stage_dirk

    heat_equation = HeatEquation(
        domain,
        lambda t, x, y: 0,
        boundary_condition,
        1.0,
        lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + 1,
        {"tol": 1e-10, "atol": 1e-10},
    )

    integration_control = IntegrationControl(0.0, 100, delta_t)

    inner_prob = om.Problem()

    inner_prob.model.add_subsystem(
        "heat_equation",
        HeatEquationStageComponent(
            heat_equation=heat_equation,
            integration_control=integration_control,
        ),
        promotes_inputs=[
            ("heat", "heat_old"),
            ("accumulated_stages", "heat_acc_stages"),
        ],
        promotes_outputs=[("result_stage_slope", "heat_slope")],
    )

    inner_prob.model.nonlinear_solver = om.NewtonSolver(
        atol=1e-10,
        rtol=1e-10,
        maxiter=10,
        solve_subsystems=True,
        err_on_non_converge=True,
        iprint=2,
    )
    inner_prob.model.linear_solver = om.PETScKrylov(restart=20)
    inner_prob.model.linear_solver.precon = om.LinearBlockJac(maxiter=1, iprint=-1)

    inner_prob.setup()

    outer_prob = om.Problem()

    outer_prob.model.add_subsystem(
        "RK_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=inner_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["heat_0"],
            write_out_distance=1,
            write_file="heat_equ_om_single.h5",
        ),
    )

    outer_prob.setup()
    outer_prob.run_model()
