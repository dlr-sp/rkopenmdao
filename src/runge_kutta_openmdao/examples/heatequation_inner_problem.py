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
    points_per_direction = 51

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
        left=lambda t, x, y: -np.sin(np.pi * t) - 1,
    )
    boundary_condition_2 = BoundaryCondition(
        upper=lambda t, x, y: 0.0,
        lower=lambda t, x, y: 0.0,
        right=lambda t, x, y: -np.cos(np.pi * t) - 1,
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

    integration_control = IntegrationControl(0.0, 1000, 100, 1e-5)

    inner_prob = om.Problem()

    heat_group = create_split_heat_group(
        points_per_direction,
        boundary_condition_1,
        boundary_condition_2,
        lambda t, x, y: 0.0,
        lambda t, x, y: 0,
        1.0,
        1.0,
        lambda x, y: 0.0,
        lambda x, y: 0.0,
        integration_control,
        {"tol": 1e-15, "atol": "legacy", "M": heat_precon},
    )

    inner_prob.model.add_subsystem("heat_group", heat_group)

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
