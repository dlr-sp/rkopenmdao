import numpy as np
import openmdao.api as om

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau

from runge_kutta_openmdao.heatequation.boundary import BoundaryCondition
from runge_kutta_openmdao.heatequation.domain import Domain
from runge_kutta_openmdao.heatequation.heatequation import HeatEquation
from runge_kutta_openmdao.heatequation.heatequation_stage_component import (
    HeatEquationStageComponent,
)
from runge_kutta_openmdao.heatequation.flux_component import FluxComponent

from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl

from scipy.sparse.linalg import LinearOperator

if __name__ == "__main__":
    points_per_direction = 51
    points_x = points_per_direction // 2 + 1
    delta_x = 1.0 / (points_per_direction - 1)
    domain_half_1 = Domain([0.0, 0.5], [0.0, 1.0], points_x, points_per_direction)
    domain_half_2 = Domain([0.5, 1.0], [0.0, 1.0], points_x, points_per_direction)

    def f(t: float):
        return np.exp(-8 * np.pi**2 * t)

    def f_prime(t: float):
        return -8 * np.pi**2 * g(t)

    def g(x: float):
        return np.cos(2 * np.pi * x)

    def g_prime(x: float):
        return -2 * np.pi * np.sin(2 * np.pi * x)

    def g_prime_prime(x: float):
        return -4 * np.pi**2 * np.cos(2 * np.pi * x)

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

    gamma = (2.0 - np.sqrt(2.0)) / 2.0
    butcher_tableau = ButcherTableau(
        np.array(
            [
                [gamma, 0.0],
                [1 - gamma, gamma],
            ]
        ),
        np.array([1 - gamma, gamma]),
        np.array([gamma, 1.0]),
    )
    atol = 1e-5
    rtol = 1e-4
    scipytol = 1e-4

    # butcher_tableau = ButcherTableau(
    #     np.array([[0.293, 0.0], [0.414, 0.293]]),
    #     np.array([0.5, 0.5]),
    #     np.array([0.293, 0.707]),
    # )
    # butcher_tableau = ButcherTableau(
    #     np.array([[0.5]]), np.array([1.0]), np.array([0.5])
    # )
    # butcher_tableau = ButcherTableau(
    #     np.array(
    #         [
    #             [0.5, 0.0, 0.0, 0.0],
    #             [0.167, 0.5, 0.0, 0.0],
    #             [-0.5, 0.5, 0.5, 0.0],
    #             [1.5, -1.5, 0.5, 0.5],
    #         ]
    #     ),
    #     np.array([1.5, -1.5, 0.5, 0.5]),
    #     np.array([0.5, 0.667, 0.5, 1.0]),
    # )

    heat_precon = LinearOperator(
        shape=(
            (points_per_direction * points_x),
            (points_per_direction * points_x),
        ),
        matvec=lambda x: delta_x**2 / -4 * x,
    )

    heat_equation_1 = HeatEquation(
        domain_half_1,
        lambda t, x, y: 0.0,
        boundary_condition_1,
        1.0,
        lambda x, y: g(x) * g(y) + 1,
        {"tol": 1e-15, "atol": "legacy", "M": heat_precon},
    )

    heat_equation_2 = HeatEquation(
        domain_half_2,
        lambda t, x, y: 0.0,
        boundary_condition_2,
        1.0,
        lambda x, y: g(x) * g(y) + 1,
        {"tol": 1e-15, "atol": "legacy", "M": heat_precon},
    )

    integration_control = IntegrationControl(0.0, 10000, 1e-5)

    inner_prob = om.Problem()

    inner_prob.model.add_subsystem(
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
            ("result_stage_heat", "stage_heat_1"),
            ("result_stage_slope", "stage_slope_1"),
        ],
    )
    inner_prob.model.add_subsystem(
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
            ("result_stage_heat", "stage_heat_2"),
            ("result_stage_slope", "stage_slope_2"),
        ],
    )
    inner_prob.model.add_subsystem(
        "flux_comp",
        FluxComponent(delta=delta_x, shape=points_per_direction, orientation="vertical"),
    )

    left_boundary_indices_1 = domain_half_2.boundary_indices("left") + 1

    right_boundary_indices_1 = domain_half_1.boundary_indices("right") - 1

    inner_prob.model.connect(
        "stage_heat_1",
        "flux_comp.left_side",
        src_indices=right_boundary_indices_1,
    )
    inner_prob.model.connect(
        "stage_heat_2",
        "flux_comp.right_side",
        src_indices=left_boundary_indices_1,
    )

    inner_prob.model.connect("flux_comp.flux", "heat_component_1.boundary_segment_right")
    inner_prob.model.connect("flux_comp.reverse_flux", "heat_component_2.boundary_segment_left")

    # inner_prob.model.nonlinear_solver = om.NonlinearBlockGS()
    newton = inner_prob.model.nonlinear_solver = om.NewtonSolver(
        solve_subsystems=True,
        iprint=2,
        # atol=atol,
        # rtol=rtol
    )
    inner_prob.model.linear_solver = om.PETScKrylov(iprint=2, atol=1e-12, rtol=1e-12)
    # inner_prob.model.linear_solver = om.ScipyKrylov(
    # iprint=2,
    # atol=scipytol
    # )
    # inner_prob.model.linear_solver = om.LinearBlockGS()

    outer_prob = om.Problem()
    outer_prob.model.add_subsystem(
        "RK_Integrator",
        RungeKuttaIntegrator(
            inner_problem=inner_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            write_file="inner_problem_stage.h5",
            quantity_tags=["heat_1", "heat_2"],
        ),
        promotes_inputs=["heat_1_initial", "heat_2_initial"],
    )

    var_comp = om.IndepVarComp("indep")
    var_comp.add_output("heat_1_initial", shape_by_conn=True, distributed=True)
    var_comp.add_output("heat_2_initial", shape_by_conn=True, distributed=True)

    outer_prob.model.add_subsystem("indep", var_comp, promotes=["*"])

    outer_prob.setup()
    outer_prob.set_val("heat_1_initial", heat_equation_1.initial_vector)
    outer_prob.set_val("heat_2_initial", heat_equation_2.initial_vector)
    outer_prob.run_model()
