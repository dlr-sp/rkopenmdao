import openmdao.api as om

from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl


class StageValueComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)

    def setup(self):
        self.add_input("stage_slope", shape_by_conn=True)
        self.add_input("old_value", copy_shape="stage_slope")
        self.add_input("acc_stages", copy_shape="stage_slope")

        self.add_output("stage_value", copy_shape="stage_slope")

    def compute(self, inputs, outputs):
        outputs["stage_value"] = inputs["old_value"] + self.options[
            "integration_control"
        ].delta_t * (
            inputs["acc_stages"]
            + self.options["integration_control"].butcher_diagonal_element * inputs["stage_slope"]
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options["integration_control"].butcher_diagonal_element
        if mode == "fwd":
            d_outputs["stage_value"] += d_inputs["old_value"]
            d_outputs["stage_value"] += delta_t * d_inputs["acc_stages"]
            d_outputs["stage_value"] += delta_t * butcher_diagonal_element * d_inputs["stage_slope"]
        elif mode == "rev":
            d_inputs["old_value"] += d_outputs["stage_value"]
            d_inputs["acc_stages"] += delta_t * d_outputs["stage_value"]
            d_inputs["stage_slope"] += delta_t * butcher_diagonal_element * d_outputs["stage_value"]
