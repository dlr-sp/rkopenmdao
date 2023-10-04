import openmdao.api as om


class FluxComponent(om.ImplicitComponent):
    """Component to bridge the gap between two split heat equations in openMDAO"""

    def initialize(self):
        self.options.declare("delta", desc="step size")
        self.options.declare("shape", desc="number of grid points")
        self.options.declare(
            "orientation",
            values=["vertical", "horizontal"],
            desc="in which direction the domain is split",
        )

    def setup(self):
        delta = self.options["delta"]
        if self.options["orientation"] == "vertical":
            self.add_input("left_side", shape=self.options["shape"])
            self.add_input("right_side", shape=self.options["shape"])
        else:
            self.add_input("lower_side", shape=self.options["shape"])
            self.add_input("upper_side", shape=self.options["shape"])
        self.add_output(
            "flux",
            shape=self.options["shape"],
        )
        self.add_output(
            "reverse_flux",
            shape=self.options["shape"],
        )

    def apply_nonlinear(self, inputs, outputs, residuals):
        positive_side = (
            "right_side" if self.options["orientation"] == "vertical" else "upper_side"
        )
        negative_side = (
            "left_side" if self.options["orientation"] == "vertical" else "lower_side"
        )
        residuals["flux"] = (
            0.5 * (inputs[positive_side] - inputs[negative_side])
            - self.options["delta"] * outputs["flux"]
        )
        residuals["reverse_flux"] = (
            0.5 * (inputs[positive_side] - inputs[negative_side])
            + self.options["delta"] * outputs["reverse_flux"]
        )

    def solve_nonlinear(self, inputs, outputs):
        positive_side = (
            "right_side" if self.options["orientation"] == "vertical" else "upper_side"
        )
        negative_side = (
            "left_side" if self.options["orientation"] == "vertical" else "lower_side"
        )
        outputs["flux"] = (
            0.5
            * (inputs[positive_side] - inputs[negative_side])
            / self.options["delta"]
        )
        outputs["reverse_flux"] = (
            0.5
            * (inputs[negative_side] - inputs[positive_side])
            / self.options["delta"]
        )

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        positive_side = (
            "right_side" if self.options["orientation"] == "vertical" else "upper_side"
        )
        negative_side = (
            "left_side" if self.options["orientation"] == "vertical" else "lower_side"
        )
        if mode == "fwd":
            if "flux" in d_residuals:
                if positive_side in d_inputs:
                    d_residuals["flux"] += 0.5 * d_inputs[positive_side]
                if negative_side in d_inputs:
                    d_residuals["flux"] -= 0.5 * d_inputs[negative_side]
                if "flux" in d_outputs:
                    d_residuals["flux"] -= self.options["delta"] * d_outputs["flux"]

            if "reverse_flux":
                if positive_side in d_inputs:
                    d_residuals["reverse_flux"] += 0.5 * d_inputs[positive_side]
                if negative_side in d_inputs:
                    d_residuals["reverse_flux"] -= 0.5 * d_inputs[negative_side]
                if "flux" in d_outputs:
                    d_residuals["reverse_flux"] += (
                        self.options["delta"] * d_outputs["reverse_flux"]
                    )

        elif mode == "rev":
            if "flux" in d_residuals:
                if positive_side in d_inputs:
                    d_inputs[positive_side] += 0.5 * d_residuals["flux"]
                if negative_side in d_inputs:
                    d_inputs[negative_side] -= 0.5 * d_residuals["flux"]
                if "flux" in d_outputs:
                    d_outputs["flux"] -= self.options["delta"] * d_residuals["flux"]

            if "reverse_flux":
                if positive_side in d_inputs:
                    d_inputs[positive_side] += 0.5 * d_residuals["reverse_flux"]
                if negative_side in d_inputs:
                    d_inputs[negative_side] -= 0.5 * d_residuals["reverse_flux"]
                if "flux" in d_outputs:
                    d_outputs["reverse_flux"] += (
                        self.options["delta"] * d_residuals["reverse_flux"]
                    )

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == "fwd":
            d_outputs["flux"] = -d_residuals["flux"] / self.options["delta"]
            d_outputs["reverse_flux"] = (
                d_residuals["reverse_flux"] / self.options["delta"]
            )
        elif mode == "rev":
            d_residuals["flux"] = -d_outputs["flux"] / self.options["delta"]
            d_residuals["reverse_flux"] = (
                d_outputs["reverse_flux"] / self.options["delta"]
            )
