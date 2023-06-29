import numpy as np
import openmdao.api as om

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau

from runge_kutta_openmdao.heatequation.boundary import BoundaryCondition
from runge_kutta_openmdao.heatequation.boundary_comp import BoundaryComp
from runge_kutta_openmdao.heatequation.domain import Domain
from runge_kutta_openmdao.heatequation.heatequation import HeatEquation
from runge_kutta_openmdao.heatequation.heatequation_stage_component import (
    HeatEquationStageComponent,
)
from runge_kutta_openmdao.heatequation.flux_component import FluxComponent
from runge_kutta_openmdao.heatequation.split_heat_group import create_split_heat_group
from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl

from scipy.sparse.linalg import LinearOperator

from runge_kutta_openmdao.runge_kutta.stage_value_component import StageValueComponent

if __name__ == "__main__":
    points_per_direction = 11
    delta_x = (points_per_direction - 1) ** -1
    delta_t = delta_x**2 / 10

    domain = Domain([0, 1], [0, 1], points_per_direction, points_per_direction)

    boundary_condition = BoundaryCondition(
        upper=lambda t, x, y: 0.0, lower=lambda t, x, y: 0.0, right=lambda t, x, y: 0.0
    )

    # gamma = (2.0 - np.sqrt(2.0)) / 2.0
    # butcher_tableau = ButcherTableau(
    #     np.array(
    #         [
    #             [gamma, 0.0],
    #             [1 - gamma, gamma],
    #         ]
    #     ),
    #     np.array([1 - gamma, gamma]),
    #     np.array([gamma, 1.0]),
    # )

    butcher_tableau = ButcherTableau(np.array([[1.0]]), np.array([1.0]), np.array([1.0]))

    heat_equation = HeatEquation(
        domain,
        lambda t, x, y: 0,
        boundary_condition,
        1.0,
        lambda x, y: 1.0,
        {"tol": 1e-10, "atol": 1e-10},
    )

    integration_control = IntegrationControl(0.0, 1, 100, 1e-5)

    inner_prob = om.Problem()

    inner_prob.model.add_subsystem(
        "heat_equation",
        HeatEquationStageComponent(
            heat_equation=heat_equation,
            shared_boundary=["left"],
            integration_control=integration_control,
        ),
        promotes_inputs=[
            ("heat", "heat_old"),
            ("accumulated_stages", "heat_acc_stages"),
            "boundary_segment_left",
        ],
        promotes_outputs=[("result_stage_slope", "heat_slope")],
    )

    inner_prob.model.add_subsystem(
        "heat_value",
        StageValueComponent(integration_control=integration_control),
        promotes_inputs=[
            ("stage_slope", "heat_slope"),
            ("old_value", "heat_old"),
            ("acc_stages", "heat_acc_stages"),
        ],
        promotes_outputs=[("stage_value", "heat_value")],
    )

    inner_prob.model.add_subsystem(
        "left_boundary",
        BoundaryComp(domain=domain, segment="left", integration_control=integration_control),
        promotes_inputs=[("heat_field", "heat_value")],
        promotes_outputs=[("boundary_value", "boundary_segment_left")],
    )
    inner_prob.model.nonlinear_solver = om.NewtonSolver(
        atol=1e-8,
        rtol=1e-8,
        maxiter=10,
        solve_subsystems=True,
        err_on_non_converge=True,
        iprint=2,
    )
    inner_prob.model.linear_solver = om.PETScKrylov(restart=20, iprint=2)
    inner_prob.model.linear_solver.precon = om.LinearRunOnce()

    inner_prob.setup()

    outer_prob = om.Problem()

    outer_prob.model.add_subsystem(
        "RK_integrator",
        RungeKuttaIntegrator(
            inner_problem=inner_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            quantity_tags=["heat_0"],
            resets=True,
        ),
    )

    outer_prob.setup()
    outer_prob.run_model()
    # inner_prob.check_partials()
    outer_prob.check_partials()
