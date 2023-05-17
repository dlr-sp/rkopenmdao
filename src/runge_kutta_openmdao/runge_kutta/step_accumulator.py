import openmdao.api as om

from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl
from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau


class StepAccumulator(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)
        self.options.declare("quantity_metadata", types=dict)
        self.options.declare("butcher_tableau", types=ButcherTableau)

    def setup(self):
        butcher: ButcherTableau = self.options["butcher_tableau"]
        for quantity, metadata in self.options["quantity_metadata"].items():
            self.add_input(
                quantity + "_old",
                shape=metadata["shape"],
                distributed=metadata["shape"] != metadata["global_shape"],
            )
            for i in range(butcher.number_of_stages()):
                self.add_input(
                    quantity + f"_stage_{i}",
                    shape=metadata["shape"],
                    distributed=metadata["shape"] != metadata["global_shape"],
                )
            self.add_output(
                quantity + "_new",
                shape=metadata["shape"],
                distributed=metadata["shape"] != metadata["global_shape"],
            )

    def compute(self, inputs, outputs):
        butcher: ButcherTableau = self.options["butcher_tableau"]
        delta_t = self.options["integration_control"].delta_t
        for quantity in self.options["quantity_metadata"]:
            outputs[quantity + "_new"] = inputs[quantity + "_old"]
            for i in range(butcher.number_of_stages()):
                outputs[quantity + "_new"] += (
                    delta_t
                    * butcher.butcher_weight_vector[i]
                    * inputs[quantity + f"_stage_{i}"]
                )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        butcher: ButcherTableau = self.options["butcher_tableau"]
        delta_t = self.options["integration_control"].delta_t
        if mode == "fwd":
            for quantity in self.options["quantity_metadata"]:
                d_outputs[quantity + "_new"] += d_inputs[quantity + "_old"]
                for i in range(butcher.number_of_stages()):
                    d_outputs[quantity + "_new"] += (
                        delta_t
                        * butcher.butcher_weight_vector[i]
                        * d_inputs[quantity + f"_stage_{i}"]
                    )
        elif mode == "rev":
            for quantity in self.options["quantity_metadata"]:
                d_inputs[quantity + "_old"] += d_outputs[quantity + "_new"]
                for i in range(butcher.number_of_stages()):
                    d_inputs[quantity + f"_stage_{i}"] += (
                        delta_t
                        * butcher.butcher_weight_vector[i]
                        * d_outputs[quantity + "_new"]
                    )
