import numpy as np
import openmdao.api as om

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau
from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl


class TestComp6(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        outputs["x_stage"] = 0.5 * delta_t * butcher_diagonal_element + np.sqrt(
            0.25 * delta_t**2 * butcher_diagonal_element**2
            + inputs["x"]
            + delta_t * inputs["acc_stages"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        divisor = 2 * np.sqrt(
            0.25 * delta_t**2 * butcher_diagonal_element**2
            + inputs["x"]
            + delta_t * inputs["acc_stages"]
        )
        if mode == "fwd":
            d_outputs["x_stage"] += d_inputs["x"] / divisor
            d_outputs["x_stage"] += delta_t * d_inputs["acc_stages"] / divisor

        elif mode == "rev":
            d_inputs["x"] += d_outputs["x_stage"] / divisor
            d_inputs["acc_stages"] += delta_t * d_outputs["x_stage"] / divisor


# butcher_tableau = ButcherTableau(
#     np.array(
#         [
#             [0.0, 0.0, 0.0, 0.0],
#             [0.5, 0.0, 0.0, 0.0],
#             [0.0, 0.5, 0.0, 0.0],
#             [0.0, 0.0, 1.0, 0.0],
#         ]
#     ),
#     np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]),
#     np.array([0.0, 0.5, 0.5, 1.0]),
# )

# alpha = 2.0 * np.cos(np.pi / 18.0) / np.sqrt(3.0)
#
# butcher_tableau = ButcherTableau(
#     np.array(
#         [
#             [0.5 * (1 + alpha), 0.0, 0.0],
#             [-0.5 * alpha, 0.5 * (1 + alpha), 0.0],
#             [1 + alpha, -(1 + 2 * alpha), 0.5 * (1 + alpha)],
#         ]
#     ),
#     np.array([1 / (6 * alpha**2), 1 - 1 / (3 * alpha**2), 1 / (6 * alpha**2)]),
#     np.array([0.5 * (1 + alpha), 0.5, 0.5 * (1 - alpha)]),
# )

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
#     np.array(
#         [
#             [0.5, 0.0],
#             [0.0, 0.5],
#         ]
#     ),
#     np.array([0.5, 0.5]),
#     np.array([0.5, 0.5]),
# )


# butcher_tableau = ButcherTableau(
#     np.array(
#         [
#             [0.5],
#         ]
#     ),
#     np.array([1.0]),
#     np.array([0.5]),
# )

num_steps = 1
integration_control = IntegrationControl(0.0, num_steps, 1e-1)

trapezoidal_rule = np.ones(num_steps + 1)
trapezoidal_rule[0] = trapezoidal_rule[num_steps] = 0.5

inner_prob = om.Problem()

inner_prob.model.add_subsystem(
    "x_comp", TestComp6(integration_control=integration_control)
)

newton = inner_prob.model.nonlinear_solver = om.NewtonSolver(
    iprint=0, solve_subsystems=True
)

inner_prob.model.linear_solver = om.ScipyKrylov(maxiter=20, iprint=0)

outer_prob = om.Problem()
outer_prob.model.add_subsystem(
    "RK_Integrator",
    RungeKuttaIntegrator(
        time_stage_problem=inner_prob,
        butcher_tableau=butcher_tableau,
        integration_control=integration_control,
        # integrated_quantities=["x"],
        # quadrature_rule_weights=trapezoidal_rule,
        time_integration_quantities=["x"],
    ),
    promotes=["*"],
)

outer_prob.setup()
outer_prob.set_val("x_initial", 1.0)
outer_prob.run_model()


outer_prob.set_val("x_initial", 1.0)

outer_prob.check_partials()
