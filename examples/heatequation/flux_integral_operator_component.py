# pylint: disable=missing-module-docstring
import numpy as np
import openmdao.api as om

from rkopenmdao.integration_control import IntegrationControl


class FluxIntegralOperatorComponent(om.ExplicitComponent):
    """
    Simple integration via trapezoidal rule
    """

    def initialize(self):
        self.options.declare("delta")
        self.options.declare("shape")
        self.options.declare("integration_control", types=IntegrationControl)
        self.options.declare("heat_coefficient", default=1e-5)

    def setup(self):
        self.add_input("flux", shape=self.options["shape"])
        self.add_input(
            "initial_flux", val=0.0, shape=1, tags=["step_input_var", "heat_integral"]
        )
        self.add_input(
            "flux_acc_stage",
            val=0.0,
            shape=1,
            tags=["accumulated_stage_var", "heat_integral"],
        )
        self.add_output(
            "integrated_flux_stage",
            shape=1,
            tags=["stage_output_var", "heat_integral"],
        )

    def compute(self, inputs, outputs):  # pylint: disable=arguments-differ
        delta_t = self.options["integration_control"].delta_t
        outputs["integrated_flux_stage"] = (
            self.options["delta"]
            * self.options["heat_coefficient"]
            * (np.sum(inputs["flux"]) - 0.5 * (inputs["flux"][0] + inputs["flux"][-1]))
            - inputs["initial_flux"]
            - delta_t * inputs["flux_acc_stage"]
        ) / (delta_t * self.options["integration_control"].butcher_diagonal_element)

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode
    ):  # pylint: disable=arguments-differ
        delta_t = self.options["integration_control"].delta_t
        divisor = delta_t * self.options["integration_control"].butcher_diagonal_element

        if mode == "fwd":
            d_outputs["integrated_flux_stage"] += (
                self.options["delta"]
                * self.options["heat_coefficient"]
                * (
                    np.sum(d_inputs["flux"])
                    - 0.5 * (d_inputs["flux"][0] + d_inputs["flux"][-1])
                )
            )
            d_outputs["integrated_flux_stage"] -= d_inputs["initial_flux"]
            d_outputs["integrated_flux_stage"] -= delta_t * d_inputs["flux_acc_stage"]
            d_outputs["integrated_flux_stage"] /= divisor
        elif mode == "rev":
            d_inputs["flux"] += np.full_like(
                d_inputs["flux"],
                self.options["delta"]
                * self.options["heat_coefficient"]
                * d_outputs["integrated_flux_stage"],
            )
            d_inputs["flux"][0] -= (
                0.5
                * self.options["delta"]
                * self.options["heat_coefficient"]
                * d_outputs["integrated_flux_stage"]
            )
            d_inputs["flux"][-1] -= (
                0.5
                * self.options["delta"]
                * self.options["heat_coefficient"]
                * d_outputs["integrated_flux_stage"]
            )
            d_inputs["flux"] /= divisor

            d_inputs["initial_flux"] -= d_outputs["integrated_flux_stage"] / divisor
            d_inputs["flux_acc_stage"] -= (
                delta_t * d_outputs["integrated_flux_stage"] / divisor
            )
