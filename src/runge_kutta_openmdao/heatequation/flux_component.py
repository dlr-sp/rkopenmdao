import openmdao.api as om


class FluxComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("delta")
        self.options.declare("shape")
        self.options.declare("orientation", values=["vertical", "horizontal"])

    def setup(self):
        delta = self.options["delta"]
        if self.options["orientation"] == "vertical":
            self.add_input("left_side", shape=self.options["shape"])
            self.add_input("right_side", shape=self.options["shape"])
        else:
            self.add_input("lower_side", shape=self.options["shape"])
            self.add_input("upper_side", shape=self.options["shape"])
        self.add_output("flux", shape=self.options["shape"])
        self.add_output(
            "reverse_flux",
            shape=self.options["shape"],
        )

    def compute(self, inputs, outputs):  # pylint: disable=arguments-differ
        positive_side = "right_side" if self.options["orientation"] == "vertical" else "upper_side"
        negative_side = "left_side" if self.options["orientation"] == "vertical" else "lower_side"
        outputs["flux"] = (
            0.5 * (inputs[positive_side] - inputs[negative_side]) / self.options["delta"]
        )
        outputs["reverse_flux"] = (
            0.5 * (inputs[negative_side] - inputs[positive_side]) / self.options["delta"]
        )

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode
    ):  # pylint: disable=arguments-differ
        positive_side = "right_side" if self.options["orientation"] == "vertical" else "upper_side"
        negative_side = "left_side" if self.options["orientation"] == "vertical" else "lower_side"
        if mode == "fwd":
            d_outputs["flux"] += (
                0.5 * (d_inputs[positive_side] - d_inputs[negative_side]) / self.options["delta"]
            )
            d_outputs["reverse_flux"] -= (
                0.5 * (d_inputs[positive_side] - d_inputs[negative_side]) / self.options["delta"]
            )
        elif mode == "rev":
            d_inputs[positive_side] += 0.5 * d_outputs["flux"] / self.options["delta"]
            d_inputs[negative_side] -= 0.5 * d_outputs["flux"] / self.options["delta"]
            d_inputs[negative_side] += 0.5 * d_outputs["reverse_flux"] / self.options["delta"]
            d_inputs[positive_side] -= 0.5 * d_outputs["reverse_flux"] / self.options["delta"]
