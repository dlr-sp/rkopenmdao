import numpy as np
import openmdao.api as om

from rkopenmdao.integration_control import IntegrationControl


class SimpleLinearODE(om.ExplicitComponent):
    """Component modelling the ODE y' = -y , y(0) = 1 (by default)"""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("y_old", val=1.0, tags=["y", "step_input_var"])
        self.add_input(
            "y_accumulated_stages", val=0.0, tags=["y", "accumulated_stage_var"]
        )
        self.add_output("y_stage", val=1.0, tags=["y", "stage_output_var"])

    def compute(self, inputs, outputs):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        outputs["y_stage"] = -(
            inputs["y_old"] + delta_t * inputs["y_accumulated_stages"]
        ) / (1 + delta_t * butcher_diagonal_element)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        factor = (1 + delta_t * butcher_diagonal_element) ** -1
        if mode == "fwd":
            d_outputs["y_stage"] -= factor * d_inputs["y_old"]
            d_outputs["y_stage"] -= delta_t * factor * d_inputs["y_accumulated_stages"]
        elif mode == "rev":
            d_inputs["y_old"] -= factor * d_outputs["y_stage"]
            d_inputs["y_accumulated_stages"] -= delta_t * factor * d_outputs["y_stage"]

    @staticmethod
    def simple_linear_solution(
        time: float,
        initial_values=1.0,
        initial_time=0.0,
    ):
        """Analytical solution to y' = -y"""
        return initial_values / np.exp(-initial_time) * np.exp(-time)
