"""
Solving the heat equation in openMDAO via the Runge-Kutta-integrator. In this version,
the heat equation is split in two via a domain decomposition two emulate a
multidisciplinary problem.
"""

# pylint: disable=duplicate-code
import numpy as np
import openmdao.api as om

from rkopenmdao.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)
from rkopenmdao.integration_control import StepTerminationIntegrationControl

from rkopenmdao.butcher_tableaux import (
    third_order_third_weak_stage_order_four_stage_dirk,
)

from .heatequation.boundary import BoundaryCondition
from .heatequation.split_heat_group import create_split_heat_group

if __name__ == "__main__":
    POINTS_PER_DIRECTION = 51
    DELTA_X = (POINTS_PER_DIRECTION - 1) ** -1
    POINTS_X = POINTS_PER_DIRECTION // 2 + 1

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

    integration_control = StepTerminationIntegrationControl(1e-3, 100, 0.0)

    inner_prob = om.Problem()

    heat_group = create_split_heat_group(
        POINTS_PER_DIRECTION,
        boundary_condition_1,
        boundary_condition_2,
        lambda t, x, y: 0.0,
        lambda t, x, y: 0.0,
        1.0,
        1.0,
        lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + 1,
        lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) + 1,
        integration_control,
        {"rtol": 1e-10, "atol": 1e-10},
    )

    inner_prob.model.add_subsystem("heat_group", heat_group)

    inner_prob.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=False, iprint=2
    )
    inner_prob.model.linear_solver = om.PETScKrylov(maxiter=100, restart=10, iprint=2)
    inner_prob.model.linear_solver.precon = om.LinearBlockJac(iprint=-1, maxiter=2)

    outer_prob = om.Problem()
    outer_prob.model.add_subsystem(
        "RK_Integrator",
        RungeKuttaIntegrator(
            time_stage_problem=inner_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["heat_1", "heat_2"],
            write_out_distance=1,
            write_file="heat_equ_om_split.h5",
        ),
        promotes_inputs=["heat_1_initial", "heat_2_initial"],
    )

    outer_prob.setup()

    outer_prob.run_model()
