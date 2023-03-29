import numpy as np
import openmdao.api as om

from src.runge_kutta_openmdao.runge_kutta.runge_kutta import ButcherTableau

from examples.heatequation.src.boundary import BoundaryCondition
from examples.heatequation.src.domain import Domain
from examples.heatequation.heat_equation import HeatEquation
from examples.heatequation.heat_equation_stage_inner_component import (
    HeatEquationStageComponent,
)

from src.runge_kutta_openmdao.runge_kutta_integrator import RungeKuttaIntegrator


class MiddleNeumann(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("delta")
        self.options.declare("shape")

    def setup(self):
        self.add_input("left_side", shape=self.options["shape"])
        self.add_input("right_side", shape=self.options["shape"])
        self.add_output("flux", shape=self.options["shape"])
        self.add_output("reverse_flux", shape=self.options["shape"])

    def compute(self, inputs, outputs):  # pylint: disable=arguments-differ
        outputs["flux"] = (
            0.5 * (inputs["right_side"] - inputs["left_side"]) / self.options["delta"]
        )
        outputs["reverse_flux"] = (
            0.5 * (inputs["left_side"] - inputs["right_side"]) / self.options["delta"]
        )

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode
    ):  # pylint: disable=arguments-differ
        if mode == "fwd":
            d_outputs["flux"] += (
                0.5
                * (d_inputs["right_side"] - d_inputs["left_side"])
                / self.options["delta"]
            )
            d_outputs["reverse_flux"] -= (
                0.5
                * (d_inputs["right_side"] - d_inputs["left_side"])
                / self.options["delta"]
            )
        elif mode == "rev":
            d_inputs["right_side"] += 0.5 * d_outputs["flux"] / self.options["delta"]
            d_inputs["left_side"] -= 0.5 * d_outputs["flux"] / self.options["delta"]
            d_inputs["left_side"] += (
                0.5 * d_outputs["reverse_flux"] / self.options["delta"]
            )
            d_inputs["right_side"] -= (
                0.5 * d_outputs["reverse_flux"] / self.options["delta"]
            )


if __name__ == "__main__":
    points_per_direction = 11
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

    heat_equation_1 = HeatEquation(
        domain_half_1,
        lambda t, x, y: 0.0,
        boundary_condition_1,
        1.0,
        lambda x, y: g(x) * g(y),
        {"tol": 1e-12, "atol": "legacy"},
    )

    heat_equation_2 = HeatEquation(
        domain_half_2,
        lambda t, x, y: 0.0,
        boundary_condition_2,
        1.0,
        lambda x, y: g(x) * g(y),
        {"tol": 1e-12, "atol": "legacy"},
    )

    inner_prob = om.Problem()

    inner_prob.model.add_subsystem(
        "heat_stage_1",
        HeatEquationStageComponent(
            heat_equation=heat_equation_1, shared_boundary=["right"], domain_num=1
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
        "heat_stage_2",
        HeatEquationStageComponent(
            heat_equation=heat_equation_2, shared_boundary=["left"], domain_num=2
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
        "flux_comp", MiddleNeumann(delta=delta_x, shape=points_per_direction)
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

    inner_prob.model.connect("flux_comp.flux", "heat_stage_1.boundary_segment_right")
    inner_prob.model.connect(
        "flux_comp.reverse_flux", "heat_stage_2.boundary_segment_left"
    )

    newton = inner_prob.model.nonlinear_solver = om.NewtonSolver(
        iprint=0,
        solve_subsystems=True,
        # atol=atol,
        # rtol=rtol
    )
    # newton.linesearch = om.ArmijoGoldsteinLS(iprint=2, atol=atol, rtol=rtol)
    inner_prob.model.linear_solver = om.ScipyKrylov(
        # atol=scipytol
    )

    outer_prob = om.Problem()
    outer_prob.model.add_subsystem(
        "RK_Integrator",
        RungeKuttaIntegrator(
            inner_problem=inner_prob,
            butcher_tableau=butcher_tableau,
            num_steps=1000,
            initial_time=0.0,
            delta_t=1e-4,
            write_file="inner_problem_stage.txt",
            quantity_tags=["heat_1", "heat_2"],
        ),
        promotes_inputs=["heat_1_initial", "heat_2_initial"],
    )

    outer_prob.setup()
    outer_prob.set_val("heat_1_initial", heat_equation_1.initial_vector)
    outer_prob.set_val("heat_2_initial", heat_equation_2.initial_vector)
    outer_prob.run_model()

    print("done running model, starting checking partials")
    # outer_prob.check_partials(step=1e-4)
