import openmdao.api as om

from runge_kutta_openmdao.runge_kutta.butcher_tableau import ButcherTableau


class StageAccumulator(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("my_stage", types=int)
        self.options.declare("quantity_metadata", types=dict)
        self.options.declare("butcher_tableau", types=ButcherTableau)

    def setup(self):
        my_stage = self.options["my_stage"]
        for quantity, metadata in self.options["quantity_metadata"].items():
            for i in range(my_stage):
                self.add_input(
                    quantity + f"_stage_{i}",
                    shape=metadata["shape"],
                    distributed=metadata["shape"] != metadata["global_shape"],
                )
            self.add_output(
                quantity + f"_accumulated_stages_{my_stage}",
                shape=metadata["shape"],
                distributed=metadata["shape"] != metadata["global_shape"],
            )

    def compute(self, inputs, outputs):
        my_stage = self.options["my_stage"]
        butcher: ButcherTableau = self.options["butcher_tableau"]
        for quantity in self.options["quantity_metadata"]:
            for i in range(my_stage):
                outputs[quantity + f"_accumulated_stages_{my_stage}"] += (
                    butcher.butcher_matrix[my_stage, i]
                    * inputs[quantity + f"_stage_{i}"]
                )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        my_stage = self.options["my_stage"]
        butcher: ButcherTableau = self.options["butcher_tableau"]
        if mode == "fwd":
            for quantity in self.options["quantity_metadata"]:
                for i in range(my_stage):
                    d_outputs[quantity + f"_accumulated_stages_{my_stage}"] += (
                        butcher.butcher_matrix[my_stage, i]
                        * d_inputs[quantity + f"_stage_{i}"]
                    )
        elif mode == "rev":
            for quantity in self.options["quantity_metadata"]:
                for i in range(my_stage):
                    d_inputs[quantity + f"_stage_{i}"] += (
                        butcher.butcher_matrix[my_stage, i]
                        * d_outputs[quantity + f"_accumulated_stages_{my_stage}"]
                    )
