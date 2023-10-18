"""
Given an om.Component modelling f(t,x) (explicit) or R(t,x) (implicit),
constructs a group that models the problem dx/dt = f(t,x) / dx/dt = R(t,x)
"""

import openmdao.api as om


class StageAssembler(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("shape", types=int, lower=1, desc="shape of x")
        self.options.declare("quantity", types="str", desc="quantity to be used as tag")

        self.options.declare("delta_t", default=1e-4, types=float)
        self.options.declare("butcher_diagonal_element", default=0.0, types=float)

    def setup(self):
        self.add_input(
            "step_input",
            shape=self.options["shape"],
            tags=[self.options["quantity"], "step_input_var"],
        )
        self.add_input(
            "accumulated_stages",
            shape=self.options["shape"],
            tags=[self.options["quantity"], "accumulated_stage_var"],
        )
        self.add_input("current_stage_slope", shape=self.options["shape"])
        self.add_output("stage_value", shape=self.options["shape"])

    def compute(self, inputs, outputs):
        delta_t = self.options["delta_t"]
        butcher_diagonal_element = self.options["butcher_diagonal_element"]
        outputs["stage_value"] = inputs["step_input"] + delta_t * (
            inputs["accumulated_stages"]
            + butcher_diagonal_element * inputs["current_stage_slope"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["delta_t"]
        butcher_diagonal_element = self.options["butcher_diagonal_element"]
        if mode == "fwd":
            d_outputs["stage_value"] += d_inputs["step_input"]
            d_outputs["stage_value"] += delta_t * d_inputs["accumulated_stages"]
            d_outputs["stage_value"] += (
                delta_t * butcher_diagonal_element * d_inputs["current_stage_slope"]
            )
        elif mode == "rev":
            d_inputs["step_input"] += d_outputs["stage_value"]
            d_inputs["accumulated_stages"] += delta_t * d_outputs["stage_value"]
            d_inputs["current_stage_slope"] += (
                delta_t * butcher_diagonal_element * d_outputs["stage_value"]
            )


class StageSlope(om.ImplicitComponent):
    def initialize(self):
        self.options.declare("shape", types=int, lower=1, desc="shape of x")
        self.options.declare("quantity", types="str", desc="quantity to be used as tag")

        # self.options.declare("mass_matrix")

    def setup(self):
        self.add_input("func_eval", shape=self.options["shape"])
        self.add_output(
            "stage_slope",
            shape=self.options["shape"],
            tags=[self.options["quantity"], "stage_output_var"],
        )

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals["stage_slope"] = outputs["stage_slope"] - inputs["func_eval"]

    def solve_nonlinear(self, inputs, outputs):
        outputs["stage_slope"] = inputs["func_eval"]

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        if mode == "fwd":
            d_residuals["stage_slope"] += d_outputs["stage_slope"]
            d_residuals["stage_slope"] -= d_inputs["func_eval"]
        elif mode == "rev":
            d_outputs["stage_slope"] += d_residuals["stage_slope"]
            d_inputs["func_eval"] -= d_residuals["stage_slope"]

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == "fwd":
            d_outputs["stage_slope"] = d_residuals["stage_slope"]
        elif mode == "rev":
            d_residuals["stage_slope"] = d_outputs["stage_slope"]
