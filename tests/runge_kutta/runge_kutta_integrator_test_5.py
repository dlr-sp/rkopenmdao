import numpy as np
import openmdao.api as om

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau
from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl


class TestComp5_1(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("y", shape=1, tags=["step_input_var", "y"])
        self.add_input("acc_stages_y", shape=1, tags=["accumulated_stage_var", "y"])
        self.add_input("y_stage", shape=1)
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        outputs["x_stage"] = (
            inputs["y"]
            + delta_t * inputs["acc_stages_y"]
            + delta_t * butcher_diagonal_element * inputs["y_stage"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            d_outputs["x_stage"] += d_inputs["y"]
            d_outputs["x_stage"] += delta_t * d_inputs["acc_stages_y"]
            d_outputs["x_stage"] += (
                delta_t * butcher_diagonal_element * d_inputs["y_stage"]
            )
        elif mode == "rev":
            d_inputs["y"] += d_outputs["x_stage"]
            d_inputs["acc_stages_y"] += delta_t * d_outputs["x_stage"]
            d_inputs["y_stage"] += (
                delta_t * butcher_diagonal_element * d_outputs["x_stage"]
            )


class TestComp5_2(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages_x", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_input("x_stage", shape=1)
        self.add_output("y_stage", shape=1, tags=["stage_output_var", "y"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        outputs["y_stage"] = (
            inputs["x"]
            + delta_t * inputs["acc_stages_x"]
            + delta_t * butcher_diagonal_element * inputs["x_stage"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            d_outputs["y_stage"] += d_inputs["x"]
            d_outputs["y_stage"] += delta_t * d_inputs["acc_stages_x"]
            d_outputs["y_stage"] += (
                delta_t * butcher_diagonal_element * d_inputs["x_stage"]
            )
        elif mode == "rev":
            d_inputs["x"] += d_outputs["y_stage"]
            d_inputs["acc_stages_x"] += delta_t * d_outputs["y_stage"]
            d_inputs["x_stage"] += (
                delta_t * butcher_diagonal_element * d_outputs["y_stage"]
            )


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

butcher_tableau = ButcherTableau(
    np.array(
        [
            [1.0],
        ]
    ),
    np.array([1.0]),
    np.array([1.0]),
)

integration_control = IntegrationControl(0.0, 1, 10, 1e-1)

inner_prob = om.Problem()

inner_prob.model.add_subsystem(
    "x_comp", TestComp5_1(integration_control=integration_control)
)
inner_prob.model.add_subsystem(
    "y_comp", TestComp5_2(integration_control=integration_control)
)

inner_prob.model.connect("x_comp.x_stage", "y_comp.x_stage")
inner_prob.model.connect("y_comp.y_stage", "x_comp.y_stage")

newton = inner_prob.model.nonlinear_solver = om.NewtonSolver(
    iprint=0, solve_subsystems=True
)

inner_prob.model.linear_solver = om.LinearBlockGS(maxiter=20)

outer_prob = om.Problem()
outer_prob.model.add_subsystem(
    "RK_Integrator",
    RungeKuttaIntegrator(
        inner_problem=inner_prob,
        butcher_tableau=butcher_tableau,
        integration_control=integration_control,
        quantity_tags=["x", "y"],
    ),
    promotes_inputs=["x_initial", "y_initial"],
)

outer_prob.setup()
outer_prob.set_val("x_initial", 1.0)
outer_prob.set_val("y_initial", 1.0)
outer_prob.run_model()
print(integration_control.stage_time)

outer_prob.check_partials()