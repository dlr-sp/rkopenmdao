"""Place for some components that are useful in multiple scenarios"""

import openmdao.api as om

from rkopenmdao.integration_control import IntegrationControl


class StageValueComponent(om.ExplicitComponent):
    """
    General purpose component to compute the state at stage time (which is needed for coupling) from the old information
    and the newly computed stage variable.
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)
        self.options.declare(
            "tag",
            types=(type(None), str),
            default=None,
            desc="""
            If not None, then this tag, along with step_input_var accumulated_stage_var will be set for the respective
            inputs. No tag will be set for the stage value, as it is an input here, and thus there must be a output
            where it is computed and where then the tags can be set by the user.""",
        )

    def setup(self):
        tag = self.options["tag"]
        self.add_input("stage_slope", shape_by_conn=True)
        self.add_input(
            "old_value",
            copy_shape="stage_slope",
            tags=[tag, "step_input_var"] if tag is not None else [],
        )
        self.add_input(
            "acc_stages",
            copy_shape="stage_slope",
            tags=[tag, "accumulated_stage_var"] if tag is not None else [],
        )

        self.add_output("stage_value", copy_shape="stage_slope")

    def compute(self, inputs, outputs):  # pylint: disable = arguments-differ
        outputs["stage_value"] = inputs["old_value"] + self.options[
            "integration_control"
        ].delta_t * (
            inputs["acc_stages"]
            + self.options["integration_control"].butcher_diagonal_element
            * inputs["stage_slope"]
        )

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode
    ):  # pylint: disable = arguments-differ
        delta_t = self.options["integration_control"].delta_t
        butcher_diagonal_element = self.options[
            "integration_control"
        ].butcher_diagonal_element
        if mode == "fwd":
            if "stage_value" in d_outputs:
                if "old_value" in d_inputs:
                    d_outputs["stage_value"] += d_inputs["old_value"]
                if "acc_stages" in d_inputs:
                    d_outputs["stage_value"] += delta_t * d_inputs["acc_stages"]
                if "stage_slope" in d_inputs:
                    d_outputs["stage_value"] += (
                        delta_t * butcher_diagonal_element * d_inputs["stage_slope"]
                    )
        elif mode == "rev":
            if "stage_value" in d_outputs:
                if "old_value" in d_inputs:
                    d_inputs["old_value"] += d_outputs["stage_value"]
                if "acc_stages" in d_inputs:
                    d_inputs["acc_stages"] += delta_t * d_outputs["stage_value"]
                if "stage_slope" in d_inputs:
                    d_inputs["stage_slope"] += (
                        delta_t * butcher_diagonal_element * d_outputs["stage_value"]
                    )


class StageUpdateComponent(om.ExplicitComponent):
    """
    General purpose component to compute the stage at stage time (which is needed for coupling) from the old information
    and the newly computed stage state. Can also be used to passthrough a variable without time-derivative to the
    RK-scheme.
    """

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)
        self.options.declare("quantity_list", types=list)

    def setup(self):
        for quantity in self.options["quantity_list"]:
            self.add_input(quantity, shape_by_conn=True)
            self.add_input(
                quantity + "_old",
                tags=[quantity, "step_input_var"],
                copy_shape=quantity,
            )
            self.add_input(
                quantity + "_acc_stages",
                tags=[quantity, "accumulated_stage_var"],
                copy_shape=quantity,
            )
            self.add_output(
                quantity + "_update",
                tags=[quantity, "stage_output_var"],
                copy_shape=quantity,
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        delta_t = self.options["integration_control"].delta_t
        a_ii = self.options["integration_control"].butcher_diagonal_element
        for quantity in self.options["quantity_list"]:
            outputs[quantity + "_update"] = (
                inputs[quantity]
                - inputs[quantity + "_old"]
                - delta_t * inputs[quantity + "_acc_stages"]
            ) / (delta_t * a_ii)

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):
        delta_t = self.options["integration_control"].delta_t
        a_ii = self.options["integration_control"].butcher_diagonal_element
        if mode == "fwd":
            for quantity in self.options["quantity_list"]:
                if quantity + "_update" in d_outputs:
                    if quantity in d_inputs:
                        d_outputs[quantity + "_update"] += (d_inputs[quantity]) / (
                            delta_t * a_ii
                        )
                    if quantity + "_old" in d_inputs:
                        d_outputs[quantity + "_update"] -= (
                            d_inputs[quantity + "_old"]
                        ) / (delta_t * a_ii)
                    d_outputs[quantity + "_update"] -= (
                        d_inputs[quantity + "_acc_stages"]
                    ) / a_ii
        elif mode == "rev":
            for quantity in self.options["quantity_list"]:
                if quantity + "_update" in d_outputs:
                    if quantity in d_inputs:
                        d_inputs[quantity] += (d_outputs[quantity + "_update"]) / (
                            delta_t * a_ii
                        )
                    if quantity + "_old" in d_inputs:
                        d_inputs[quantity + "_old"] -= (
                            d_outputs[quantity + "_update"]
                        ) / (delta_t * a_ii)
                    d_inputs[quantity + "_acc_stages"] -= (
                        d_outputs[quantity + "_update"]
                    ) / a_ii
