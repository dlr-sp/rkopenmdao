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
from runge_kutta_openmdao.heatequation.flux_integral_operator_component import (
    FluxIntegralOperatorComponent,
)
from runge_kutta_openmdao.heatequation.split_heat_group import create_split_heat_group
from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl

from scipy.sparse.linalg import LinearOperator

from pprint import pprint

if __name__ == "__main__":
    points_per_direction = 11
    points_x = points_per_direction // 2 + 1
    delta_x = 1.0 / (points_per_direction - 1)

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

    # butcher_tableau = ButcherTableau(
    #     np.array([[0.293, 0.0], [0.414, 0.293]]),
    #     np.array([0.5, 0.5]),
    #     np.array([0.293, 0.707]),
    # )

    # butcher_tableau = ButcherTableau(np.array([[0.5]]), np.array([1.0]), np.array([0.5]))

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

    integration_control = IntegrationControl(0.0, 1, 1e-4)

    inner_prob = om.Problem()

    heat_group = create_split_heat_group(
        points_per_direction,
        boundary_condition_1,
        boundary_condition_2,
        lambda t, x, y: 0.0,
        lambda t, x, y: 0.0,
        1.0,
        1.0,
        lambda x, y: 0.0,
        lambda x, y: 0.0,
        integration_control,
        {"tol": 1e-15, "atol": "legacy"},  # , "M": heat_precon},
        om.NewtonSolver(solve_subsystems=True, maxiter=30, atol=1e-10, rtol=1e-10),
        om.PETScKrylov(atol=1e-12, rtol=1e-12),
    )

    inner_prob.model.add_subsystem("heat_group", heat_group, promotes=["*"])

    inner_prob.model.add_subsystem(
        "flux_int_comp",
        FluxIntegralOperatorComponent(
            delta=delta_x,
            shape=points_per_direction,
            integration_control=integration_control,
        ),
    )

    inner_prob.model.connect("flux_comp.flux", "flux_int_comp.flux")

    inner_prob.model.nonlinear_solver = om.NonlinearRunOnce(iprint=2)
    inner_prob.model.linear_solver = om.LinearRunOnce()
    trapezoidal_rule = np.array([0.5, 0.5])
    outer_prob = om.Problem()
    outer_prob.model.add_subsystem(
        "RK_Integrator",
        RungeKuttaIntegrator(
            inner_problem=inner_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            write_file="inner_problem_stage.h5",
            # integrated_quantities=["heat_integral"],
            # quadrature_rule_weights=trapezoidal_rule,
            quantity_tags=["heat_1", "heat_2", "heat_integral"],
        ),
        promotes_inputs=["*"],
    )

    outer_prob.setup()
    outer_prob.run_model()

    # jvp_rev = outer_prob.compute_jacvec_product(
    #     of=["RK_Integrator.heat_integral_final"],
    #     wrt=[
    #         "RK_Integrator.heat_1_initial",
    #         "RK_Integrator.heat_2_initial",
    #         "RK_Integrator.heat_integral_initial",
    #     ],
    #     mode="rev",
    #     seed={"RK_Integrator.heat_integral_final": 1.0},
    # )
    # print(jvp_rev)
    #
    # jvp_fwd = outer_prob.compute_jacvec_product(
    #     of=["RK_Integrator.heat_integral_final"],
    #     wrt=[
    #         "RK_Integrator.heat_1_initial",
    #         "RK_Integrator.heat_2_initial",
    #         "RK_Integrator.heat_integral_initial",
    #     ],
    #     mode="fwd",
    #     seed={
    #         "RK_Integrator.heat_integral_initial": 1.0,
    #         "RK_Integrator.heat_1_initial": np.ones(66),
    #         "RK_Integrator.heat_2_initial": np.ones(66),
    #     },
    # )
    # print(jvp_fwd)

    # print("done running model, starting checking partials")

    # outer_prob.check_partials(step=1e-1)
    # inner_prob.check_partials(step=1e-1)
    # inner_prob.check_totals(
    #     of=[
    #         "heat_group.heat_component_1.result_stage_slope",
    #         "heat_group.heat_component_2.result_stage_slope",
    #         "flux_int_comp.integrated_flux_stage",
    #     ],
    #     wrt=[
    #         "heat_1",
    #         "accumulated_stages_1",
    #         "heat_2",
    #         "accumulated_stages_2",
    #         "flux_int_comp.initial_flux",
    #         "flux_int_comp.flux_acc_stage",
    #     ],
    #     step=1e-1
    #     # directional=True,
    # )

    # in_mod = inner_prob.model
    # pprint(in_mod._var_offsets)
