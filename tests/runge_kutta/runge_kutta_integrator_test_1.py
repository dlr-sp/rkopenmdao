import numpy as np
import openmdao.api as om

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau
from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl


class TestComp1(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        outputs["x_stage"] = (inputs["x"] + delta_t * inputs["acc_stages"]) / (
            1 - delta_t * self.options["integration_control"].butcher_diagonal_element
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        divisor = (
            1 - delta_t * self.options["integration_control"].butcher_diagonal_element
        )
        if mode == "fwd":
            d_outputs["x_stage"] += d_inputs["x"] / divisor
            d_outputs["x_stage"] += delta_t * d_inputs["acc_stages"] / divisor
        elif mode == "rev":
            d_inputs["x"] += d_outputs["x_stage"] / divisor
            d_inputs["acc_stages"] += delta_t * d_outputs["x_stage"] / divisor


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
#             [1.0],
#         ]
#     ),
#     np.array([1.0]),
#     np.array([1.0]),
# )

integration_control = IntegrationControl(0.0, 1, 10, 1e-1)

inner_prob = om.Problem()

inner_prob.model.add_subsystem(
    "x_comp", TestComp1(integration_control=integration_control)
)

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
        quantity_tags=["x"],
    ),
    promotes_inputs=["x_initial"],
)

outer_prob.setup()
outer_prob.set_val("x_initial", 1.0)
outer_prob.run_model()

outer_prob.check_partials(form="central")
