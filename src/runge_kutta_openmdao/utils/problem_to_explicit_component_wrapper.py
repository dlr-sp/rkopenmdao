import openmdao.api as om


class ProblemToExplicitComponentWrapper(om.ExplicitComponent):
    def __init__(
        self, inner_problem: om.Problem, input_dict: dict, output_dict: dict, **kwargs
    ):
        super().__init__(**kwargs)
        self._inner_problem: om.Problem = inner_problem
        self._input_dict: dict = input_dict
        self._output_dict: dict = output_dict
        self._inner_input_metadata: dict = {}
        self._inner_output_metadata: dict = {}

    def setup(self):
        # TODO: get metadata from inner problem
        self._inner_problem.setup()
        self._inner_problem.final_setup()

        self._inner_input_metadata = dict(
            self._inner_problem.model.get_io_metadata(
                iotypes="input",
                metadata_keys=["shape", "global_shape"],
                get_remote=False,
            )
        )

        self._inner_output_metadata = dict(
            self._inner_problem.model.get_io_metadata(
                iotypes="output",
                metadata_keys=["shape", "global_shape"],
                get_remote=False,
            )
        )

        for inner_var, outer_var in self._input_dict.items():
            self.add_input(
                outer_var,
                shape=self._inner_input_metadata[inner_var]["shape"],
                distributed=self._inner_input_metadata[inner_var]["shape"]
                != self._inner_input_metadata[inner_var]["global_shape"],
            )

        for inner_var, outer_var in self._output_dict.items():
            self.add_output(
                outer_var,
                shape=self._inner_output_metadata[inner_var]["shape"],
                distributed=self._inner_output_metadata[inner_var]["shape"]
                != self._inner_output_metadata[inner_var]["global_shape"],
            )

    def compute(self, inputs, outputs):
        for inner_var, outer_var in self._input_dict.items():
            self._inner_problem.set_val(name=inner_var, val=inputs[outer_var])

        self._inner_problem.run_model()

        for inner_var, outer_var in self._output_dict.items():
            outputs[outer_var] = self._inner_problem.get_val(
                inner_var, get_remote=False
            )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            self._compute_jacvec_product_fwd(inputs, d_inputs, d_outputs)
        elif mode == "rev":
            self._compute_jacvec_product_rev(inputs, d_inputs, d_outputs)

    def _compute_jacvec_product_fwd(self, inputs, d_inputs, d_outputs):
        for inner_var, outer_var in self._input_dict.items():
            self._inner_problem.set_val(name=inner_var, val=inputs[outer_var])
        self._inner_problem.run_model()

        of_vars = list(self._output_dict.keys())
        wrt_vars = list(self._input_dict.keys())

        seed = {}
        for inner_var, outer_var in self._input_dict.items():
            seed[inner_var] = d_inputs[outer_var]
        self._inner_problem.model._linearize(None)

        jvp = self._inner_problem.compute_jacvec_product(of_vars, wrt_vars, "fwd", seed)

        for inner_var, outer_var in self._output_dict.items():
            d_outputs[outer_var] += jvp[inner_var]

    def _compute_jacvec_product_rev(self, inputs, d_inputs, d_outputs):
        for inner_var, outer_var in self._input_dict.items():
            self._inner_problem.set_val(name=inner_var, val=inputs[outer_var])
        self._inner_problem.run_model()

        of_vars = list(self._output_dict.keys())
        wrt_vars = list(self._input_dict.keys())

        seed = {}
        for inner_var, outer_var in self._output_dict.items():
            seed[inner_var] = d_outputs[outer_var]

        self._inner_problem.model._linearize(None)
        jvp = self._inner_problem.compute_jacvec_product(of_vars, wrt_vars, "rev", seed)

        for inner_var, outer_var in self._input_dict.items():
            # TODO: need to do allreduce in case of parallel groups?
            d_inputs[outer_var] += jvp[inner_var]
