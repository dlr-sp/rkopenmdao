import numpy as np
import openmdao.api as om

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau
from runge_kutta_openmdao.runge_kutta.runge_kutta_integrator import (
    RungeKuttaIntegrator,
)
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl


class TestComp2(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("x", shape=1, tags=["step_input_var", "x"])
        self.add_input("acc_stages", shape=1, tags=["accumulated_stage_var", "x"])
        self.add_output("x_stage", shape=1, tags=["stage_output_var", "x"])

    def compute(self, inputs, outputs):
        stages_time = self.options["integration_control"].stage_time
        outputs["x_stage"] = stages_time

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            d_outputs["x_stage"] += 0
        elif mode == "rev":
            d_inputs["x"] += 0
            d_inputs["acc_stages"] += 0


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

alpha = 2.0 * np.cos(np.pi / 18.0) / np.sqrt(3.0)

butcher_tableau = ButcherTableau(
    np.array(
        [
            [0.5 * (1 + alpha), 0.0, 0.0],
            [-0.5 * alpha, 0.5 * (1 + alpha), 0.0],
            [1 + alpha, -(1 + 2 * alpha), 0.5 * (1 + alpha)],
        ]
    ),
    np.array([1 / (6 * alpha**2), 1 - 1 / (3 * alpha**2), 1 / (6 * alpha**2)]),
    np.array([0.5 * (1 + alpha), 0.5, 0.5 * (1 - alpha)]),
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
#     np.array(
#         [
#             [0.5],
#         ]
#     ),
#     np.array([1.0]),
#     np.array([0.5]),
# )

integration_control = IntegrationControl(0.0, 20, 10, 1e-1)

inner_prob = om.Problem()

inner_prob.model.add_subsystem(
    "x_comp", TestComp2(integration_control=integration_control)
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
