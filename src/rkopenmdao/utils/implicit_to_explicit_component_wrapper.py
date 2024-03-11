# pylint: disable=missing-module-docstring


import openmdao.api as om


class ImplicitToExplicitComponentWrapper(om.ExplicitComponent):
    """
    Wraps an implicit component to an explicit compontent,
    such that the residual of the wrapped implicit component is the
    output of the resulting explicit component, and the inputs and
    outputs of the implicit component are the inputs of the explicit
    component
    """

    def _declare_options(self):
        super()._declare_options()
        self.options.declare(
            "implicit_component",
            types=om.ImplicitComponent,
            desc="Implicit component to be wrapped",
        )

    def setup(self):
        implicit_component: om.ImplicitComponent = self.options["implicit_component"]
        input_list = implicit_component.list_inputs(
            val=False, shape=True, out_steam=None
        )
        for input_var_name, metadata in input_list:
            self.add_input(input_var_name, shape=metadata["shape"])

        output_list = implicit_component.list_outputs(
            val=True, shape=True, out_stream=None
        )
        for output_var_name, metadata in output_list:
            self.add_input(output_var_name, shape=metadata["shape"])
            self.add_output(output_var_name + "_residual", shape=metadata["shape"])

    def compute(self, inputs, outputs):  # pylint: disable=arguments-differ
        implicit_component: om.ImplicitComponent = self.options["implicit_component"]
        implicit_inputs = om.DefaultVector("nonlinear", "inputs", implicit_component)
        implicit_outputs = om.DefaultVector("nonlinear", "output", implicit_component)
        implicit_residuals = om.DefaultVector(
            "nonlinear", "residual", implicit_component
        )

        for input_var_name in implicit_inputs.keys():
            implicit_inputs[input_var_name] = inputs[input_var_name]

        for output_var_name in implicit_outputs.keys():
            implicit_outputs[output_var_name] = inputs[output_var_name]

        implicit_component.apply_nonlinear(
            implicit_inputs, implicit_outputs, implicit_residuals
        )

        for output_var_name, residual_value in implicit_residuals.items():
            outputs[output_var_name + "_residual"] = residual_value

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode
    ):  # pylint: disable=arguments-differ
        implicit_component: om.ImplicitComponent = self.options["implicit_component"]
        implicit_inputs = om.DefaultVector("nonlinear", "inputs", implicit_component)
        implicit_outputs = om.DefaultVector("nonlinear", "output", implicit_component)
        implicit_d_inputs = om.DefaultVector("linear", "inputs", implicit_component)
        implicit_d_outputs = om.DefaultVector("linear", "output", implicit_component)
        implicit_d_residuals = om.DefaultVector(
            "linear", "residual", implicit_component
        )
        for input_var_name in implicit_inputs.keys():
            implicit_inputs[input_var_name] = inputs[input_var_name]
            if mode == "fwd":
                implicit_d_inputs[input_var_name] = d_inputs[input_var_name]

        for output_var_name in implicit_outputs.keys():
            implicit_outputs[output_var_name] = inputs[output_var_name]
            if mode == "fwd":
                implicit_d_outputs[output_var_name] = d_inputs[output_var_name]
            elif mode == "rev":
                implicit_d_residuals[output_var_name] = d_outputs[
                    output_var_name + "_residual"
                ]

        implicit_component.apply_linear(
            implicit_inputs,
            implicit_outputs,
            implicit_d_inputs,
            implicit_d_outputs,
            implicit_d_residuals,
        )

        if mode == "rev":
            for input_var_name, d_input_value in implicit_d_inputs.items():
                d_inputs[input_var_name] += d_input_value

        for (
            output_var_name
        ) in implicit_d_outputs.keys():  # pylint: disable=consider-using-dict-items
            if mode == "fwd":
                d_outputs[output_var_name + "_residual"] += implicit_d_residuals[
                    output_var_name
                ]
            elif mode == "rev":
                d_inputs[output_var_name] += implicit_d_outputs[output_var_name]
