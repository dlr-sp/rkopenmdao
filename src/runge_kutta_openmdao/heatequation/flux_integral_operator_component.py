"""
Simple integration via trapezoidal rule"""

import numpy as np
import openmdao.api as om


class FluxIntegralOperatorComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("delta")
        self.options.declare("shape")
        # we add that later, for now without
        # self.options.declare("heat_coefficient", default = lambda x: 1.0)

    def setup(self):
        self.add_input("flux", shape=self.options["shape"])
        self.add_output("integrated_flux", shape=1, tags="functional")

    def compute(self, inputs, outputs):  # pylint: disable=arguments-differ
        outputs["integrated_flux"] = self.options["delta"] * (
            np.sum(inputs["flux"]) - 0.5 * (inputs["flux"][0] - inputs["flux"][1])
        )

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode
    ):  # pylint: disable=arguments-differ
        if mode == "fwd":
            d_outputs["integrated_flux"] = self.options["delta"] * (
                np.sum(d_inputs["flux"])
                - 0.5 * (d_inputs["flux"][0] - d_inputs["flux"][1])
            )
        elif mode == "rev":
            d_inputs["flux"] = self.options["delta"] * (
                np.sum(d_outputs["integrated_flux"])
                - 0.5
                * (d_outputs["integrated_flux"][0] - d_outputs["integrated_flux"][1])
            )
