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
from runge_kutta_openmdao.heatequation.split_heat_group import create_split_heat_group
from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)

from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl

from scipy.sparse.linalg import LinearOperator


if __name__ == "__main__":
    points_per_direction = 21
    points_x = points_per_direction // 2 + 1
    delta_x = 1.0 / (points_per_direction - 1)
    delta_t = delta_x**2 / 10

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

    # butcher_tableau = ButcherTableau(
    #     np.array([[0.293, 0.0], [0.414, 0.293]]),
    #     np.array([0.5, 0.5]),
    #     np.array([0.293, 0.707]),
    # )
    butcher_tableau = ButcherTableau(np.array([[0.0]]), np.array([1.0]), np.array([0.0]))
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
        matvec=lambda x: delta_x**2 / (delta_x**2 + 4 * delta_t) * x,
    )

    integration_control = IntegrationControl(0.0, 1, 100, delta_t)
    # heat_lin_solver = om.LinearBlockGS(atol=1e-12, rtol=1e-12, iprint=2)
    heat_lin_solver = om.PETScKrylov(restart=20, iprint=2, err_on_non_converge=True)
    heat_lin_solver.precon = om.LinearRunOnce()
    inner_prob = om.Problem()
    heat_group = create_split_heat_group(
        points_per_direction,
        boundary_condition_1,
        boundary_condition_2,
        lambda t, x, y: 0.0,
        lambda t, x, y: 0.0,
        1.0,
        1.0,
        lambda x, y: g(x) * g(y),
        lambda x, y: g(x) * g(y),
        integration_control,
        {"tol": 1e-10, "atol": 1e-10},  # "M": heat_precon},
        om.NewtonSolver(
            rtol=1e-10,
            solve_subsystems=True,
            maxiter=30,
            max_sub_solves=3,
            err_on_non_converge=True,
            iprint=2,
        ),
        heat_lin_solver,
    )

    inner_prob.model.add_subsystem("heat_group", heat_group)

    outer_prob = om.Problem()
    outer_prob.model.add_subsystem(
        "RK_Integrator",
        RungeKuttaIntegrator(
            inner_problem=inner_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            quantity_tags=["heat_1", "heat_2"],
        ),
        promotes_inputs=["heat_1_initial", "heat_2_initial"],
    )

    outer_prob.setup()

    outer_prob.run_model()

    print("done running model, starting checking partials")
    # for key, value in inner_prob.model._outputs.items():
    #     print(key, value)
    outer_prob.check_partials(step=1e-1)

    # inner_prob.check_partials(step=1e-1)
    # inner_prob.check_totals(
    #     of=[
    #         "heat_group.stage_heat_1",
    #         "heat_group.heat_slope_1",
    #         "heat_group.stage_heat_2",
    #         "heat_group.heat_slope_2",
    #     ],
    #     wrt=[
    #         "heat_group.heat_1",
    #         "heat_group.accumulated_stages_1",
    #         "heat_group.heat_2",
    #         "heat_group.accumulated_stages_2",
    #     ],
    #     step=1e-1,
    # )
