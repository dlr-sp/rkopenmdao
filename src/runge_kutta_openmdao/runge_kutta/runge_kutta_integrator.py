import openmdao.api as om
import h5py
import numpy as np


from .butcher_tableau import ButcherTableau
from .integration_control import IntegrationControl


class RungeKuttaIntegrator(om.ExplicitComponent):
    """
    Outer component for solving time-dependent problems with Runge-Kutta-schemes.
    Needs an inner problem that models one stage of the RK-method.
    The calculation of the value at the next time step is done in this component
    outside of the inner problem.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quantity_to_inner_vars: dict = {}
        self.stage_cache: dict = {}
        self.d_stage_cache: dict = {}
        self._time_cache = {}
        self.of_vars: list = []
        self.of_vars_functional: list = []
        self.wrt_vars: list = []

    def initialize(self):
        self.options.declare(
            "inner_problem", types=om.Problem, desc="The inner problem"
        )

        # self.options.declare(
        #     "postprocessing_problem",
        #     types=(om.Problem, None),
        #     desc="A problem used to calculate additional values (e.g. functionals) based on the results of the timesteps.",
        #     default=None,
        # )

        self.options.declare(
            "butcher_tableau",
            types=ButcherTableau,
            desc="The butcher tableau for the RK-scheme",
        )
        self.options.declare("integration_control", types=IntegrationControl)

        self.options.declare(
            "write_file",
            types=str,
            default="data.h5",
            desc="The file where the results of each time steps are written. In parallel, the rank is prepended",
        )

        self.options.declare(
            "quantity_tags",
            types=list,
            desc="tags used to differentiate the quantitys",
        )

        self.options.declare(
            "quadrature_rule_weights",
            types=(np.ndarray, None),
            default=None,
            desc="Quadrature rule weights, needs to be of size num_steps + 1 from integration_control",
        )

        self.options.declare(
            "integrated_quantities",
            types=list,
            default=[],
            desc="""subset of the quantities from quantity_tags indicating which of them gets integrated via the given quadrature rule.
            If there is no quadrature rule while integrated_quantities isn't empty, an error gets thrown.""",
        )

    def setup(self):
        assert set(self.options["integrated_quantities"]).issubset(
            set(self.options["quantity_tags"])
        )
        self.options["inner_problem"].setup()

        self._setup_variable_information()

        self._setup_wrt_and_of_vars()

        # TODO: maybe add methods for time-independent in/outputs

    def compute(self, inputs, outputs):  # pylint: disable = arguments-differ
        self._setup_stage_cache()
        self._transfer_input_vector_from_outer_to_inner_problem(inputs)

        self._create_file_and_write_initial_data(inputs)

        self._integrated_quantities_initial_values(inputs, outputs)

        self._run_steps(outputs)

        self._transfer_output_vector_from_inner_to_outer_problem(outputs)

    def _integrated_quantities_initial_values(self, input_vector, output_vector):
        integrated_quantities = self.options["integrated_quantities"]
        quadrature_rule_weights = self.options["quadrature_rule_weights"]
        delta_t = self.options["integration_control"].delta_t
        for quantity in integrated_quantities:
            output_vector[quantity + "_integrated"] += (
                delta_t
                * quadrature_rule_weights[0]
                * input_vector[quantity + "_initial"]
            )

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode
    ):  # pylint: disable = arguments-differ
        self._setup_stage_cache()
        self._setup_d_stage_cache(mode)
        if mode == "fwd":
            self._compute_jacvec_product_fwd(inputs, d_inputs, d_outputs)

        elif mode == "rev":
            self._compute_jacvec_product_rev(inputs, d_inputs, d_outputs)

    def _compute_jacvec_product_fwd(self, inputs, d_inputs, d_outputs):
        contributions = self._vector_class("linear", "input", self)
        self.vector_set_between_runge_kutta_parts(contributions, d_inputs)

        self._transfer_input_vector_from_outer_to_inner_problem(inputs)

        self._integrated_quantities_initial_values(d_inputs, d_outputs)

        self._run_steps_jacvec_fwd(d_outputs, contributions)

        self.vector_scaled_add_between_runge_kutta_parts(d_outputs, 1.0, contributions)

    def _compute_jacvec_product_rev(self, inputs, d_inputs, d_outputs):
        contributions = self._vector_class("linear", "output", self)
        contributions.set_vec(d_outputs)

        self._setup_time_cache()

        self._run_checkpoints_jacvec_rev(contributions, d_outputs)
        self.vector_scaled_add_between_runge_kutta_parts(d_inputs, 1.0, contributions)
        self._add_functional_contributions(d_inputs, contributions, d_outputs)

    def _add_functional_contributions(
        self, d_input_vector, contributions, d_output_vector
    ):
        delta_t = self.options["integration_control"].delta_t
        quadrature_rule_weights = self.options["quadrature_rule_weights"]
        for quantity in self.options["integrated_quantities"]:
            d_input_vector[quantity + "_initial"] += (
                delta_t
                * quadrature_rule_weights[0]
                * d_output_vector[quantity + "_integrated"]
                + contributions[quantity + "_integrated"]
            )

    def _setup_variable_information(self):
        inner_problem: om.Problem = self.options["inner_problem"]

        old_step_input_vars = dict(
            inner_problem.model.list_inputs(
                val=False,
                prom_name=True,
                tags=["step_input_var"],
                out_stream=None,
                all_procs=True,
                # is_indep_var=True,
            )
        )

        acc_stage_input_vars = dict(
            inner_problem.model.list_inputs(
                val=False,
                prom_name=True,
                tags=["accumulated_stage_var"],
                out_stream=None,
                all_procs=True,
                # is_indep_var=True,
            )
        )

        stage_output_vars = dict(
            inner_problem.model.list_outputs(
                val=False,
                prom_name=True,
                tags=["stage_output_var"],
                out_stream=None,
                all_procs=True,
            )
        )

        for quantity in self.options["quantity_tags"]:
            self._fill_quantity_to_inner_vars(
                quantity, old_step_input_vars, acc_stage_input_vars, stage_output_vars
            )

            if self.quantity_to_inner_vars[quantity]["stage_output_var"] is None:
                raise AssertionError(
                    f"There is no stage_output_var for quantity {quantity}"
                )

            if (
                self.quantity_to_inner_vars[quantity]["step_input_var"] is not None
                and self.quantity_to_inner_vars[quantity]["accumulated_stage_var"]
                is None
            ) or (
                self.quantity_to_inner_vars[quantity]["step_input_var"] is None
                and self.quantity_to_inner_vars[quantity]["accumulated_stage_var"]
                is not None
            ):
                raise AssertionError(
                    f"There needs to be either both or none of step_input_var and accumulated_stage_var for quantity {quantity}"
                )

            self._add_inputs_and_outputs(quantity)

    def _fill_quantity_to_inner_vars(
        self, quantity, old_step_input_vars, acc_stage_input_vars, stage_output_vars
    ):
        inner_problem: om.Problem = self.options["inner_problem"]
        quantity_inputs = dict(
            inner_problem.model.list_inputs(
                val=False,
                prom_name=True,
                tags=[quantity],
                shape=True,
                global_shape=True,
                out_stream=None,
                # all_procs=True,
                # is_indep_var=True,
            )
        )

        quantity_outputs = dict(
            inner_problem.model.list_outputs(
                val=False,
                prom_name=True,
                tags=[quantity],
                shape=True,
                global_shape=True,
                out_stream=None,
                # all_procs=True,
            )
        )

        self.quantity_to_inner_vars[quantity] = {
            "step_input_var": None,
            "accumulated_stage_var": None,
            "stage_output_var": None,
            "shape": None,
            "global_shape": None,
        }

        # TODO: use own error types instead of AssertionError, like in the DevRound

        for var, metadata in quantity_inputs.items():
            if var in old_step_input_vars:
                self.quantity_to_inner_vars[quantity]["step_input_var"] = var
                if self.quantity_to_inner_vars[quantity]["shape"] is None:
                    self.quantity_to_inner_vars[quantity]["shape"] = metadata["shape"]
                else:
                    assert (
                        metadata["shape"]
                        == self.quantity_to_inner_vars[quantity]["shape"]
                    )
                if self.quantity_to_inner_vars[quantity]["global_shape"] is None:
                    self.quantity_to_inner_vars[quantity]["global_shape"] = metadata[
                        "global_shape"
                    ]
                else:
                    assert (
                        metadata["global_shape"]
                        == self.quantity_to_inner_vars[quantity]["global_shape"]
                    )

            elif var in acc_stage_input_vars:
                self.quantity_to_inner_vars[quantity]["accumulated_stage_var"] = var
                if self.quantity_to_inner_vars[quantity]["shape"] is None:
                    self.quantity_to_inner_vars[quantity]["shape"] = metadata["shape"]
                else:
                    assert (
                        metadata["shape"]
                        == self.quantity_to_inner_vars[quantity]["shape"]
                    )
                if self.quantity_to_inner_vars[quantity]["global_shape"] is None:
                    self.quantity_to_inner_vars[quantity]["global_shape"] = metadata[
                        "global_shape"
                    ]
                else:
                    assert (
                        metadata["global_shape"]
                        == self.quantity_to_inner_vars[quantity]["global_shape"]
                    )

        for var, metadata in quantity_outputs.items():
            if var in stage_output_vars:
                self.quantity_to_inner_vars[quantity]["stage_output_var"] = var
            if self.quantity_to_inner_vars[quantity]["shape"] is None:
                self.quantity_to_inner_vars[quantity]["shape"] = metadata["shape"]
            else:
                assert (
                    metadata["shape"] == self.quantity_to_inner_vars[quantity]["shape"]
                )
            if self.quantity_to_inner_vars[quantity]["global_shape"] is None:
                self.quantity_to_inner_vars[quantity]["global_shape"] = metadata[
                    "global_shape"
                ]
            else:
                assert (
                    metadata["global_shape"]
                    == self.quantity_to_inner_vars[quantity]["global_shape"]
                )

    def _add_inputs_and_outputs(self, quantity):
        self.add_input(
            quantity + "_initial",
            shape=self.quantity_to_inner_vars[quantity]["shape"],
            distributed=self.quantity_to_inner_vars[quantity]["shape"]
            != self.quantity_to_inner_vars[quantity]["global_shape"],
        )
        self.add_output(
            quantity + "_final",
            copy_shape=quantity + "_initial",
            distributed=self.quantity_to_inner_vars[quantity]["shape"]
            != self.quantity_to_inner_vars[quantity]["global_shape"],
        )
        if quantity in self.options["integrated_quantities"]:
            self.add_output(
                quantity + "_integrated",
                copy_shape=quantity + "_initial",
                distributed=self.quantity_to_inner_vars[quantity]["shape"]
                != self.quantity_to_inner_vars[quantity]["global_shape"],
            )

    def _setup_wrt_and_of_vars(self):
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            self.of_vars.append(var_dict["stage_output_var"])
            self.wrt_vars.append(var_dict["step_input_var"])
            self.wrt_vars.append(var_dict["accumulated_stage_var"])
            if quantity in self.options["integrated_quantities"]:
                self.of_vars_functional.append(var_dict["stage_output_var"])

    def _update_step_info(self, step):
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        self.options["integration_control"].step = step
        self.options["integration_control"].step_time_old = initial_time + delta_t * (
            step - 1
        )
        self.options["integration_control"].step_time_new = (
            initial_time + delta_t * step
        )

    def _run_steps(self, output_vector):
        num_steps = self.options["integration_control"].num_steps
        for step in range(1, num_steps + 1):
            self._update_step_info(step)
            self._run_stages()
            self._update_step_fwd()
            self._integrated_quantities_contribution(output_vector, step)
            self._save_to_file_at_checkpoint(step)

    def _integrated_quantities_contribution(self, output_vector, step):
        inner_problem: om.Problem = self.options["inner_problem"]
        integrated_quantities = self.options["integrated_quantities"]
        quadrature_rule_weights = self.options["quadrature_rule_weights"]
        delta_t = self.options["integration_control"].delta_t
        for quantity in integrated_quantities:
            step_variable_name = self.quantity_to_inner_vars[quantity]["step_input_var"]
            output_vector[quantity + "_integrated"] += (
                delta_t
                * quadrature_rule_weights[step]
                * inner_problem.get_val(step_variable_name)
            )

    def _run_steps_jacvec_fwd(self, d_output_vector, contributions):
        num_steps = self.options["integration_control"].num_steps
        for step in range(1, num_steps + 1):
            seed = self._fwd_seed_step(contributions)

            self._update_step_info(step)
            self._run_stages_jacvec_fwd(seed)

            self._update_step_fwd()
            self._update_contribution(contributions)
            self._integrated_quantities_contribution_jacvec_fwd(
                d_output_vector, contributions, step
            )

    def _integrated_quantities_contribution_jacvec_fwd(
        self, output_vector, contributions, step
    ):
        integrated_quantities = self.options["integrated_quantities"]
        quadrature_rule_weights = self.options["quadrature_rule_weights"]
        delta_t = self.options["integration_control"].delta_t
        for quantity in integrated_quantities:
            output_vector[quantity + "_integrated"] += (
                delta_t
                * quadrature_rule_weights[step]
                * contributions[quantity + "_initial"]
            )

    def _run_checkpoints_jacvec_rev(self, intermediate_d_outputs, d_outputs):
        for checkpoint in self._checkpoint_generator_reverse():
            self._write_checkpoint_into_inner_problem(checkpoint)
            self._forward_iteration_part(checkpoint)
            self._backward_iteration_part(checkpoint, d_outputs, intermediate_d_outputs)
            self._clear_time_cache()

    def _checkpoint_generator_reverse(self):
        checkpoint_distance = self.options["integration_control"].checkpoint_distance
        num_steps = self.options["integration_control"].num_steps
        num_iterations = (num_steps - 1) // checkpoint_distance
        for i in range(num_iterations, -1, -1):
            yield checkpoint_distance * i

    def _forward_iteration_part(self, checkpoint):
        checkpoint_distance = self.options["integration_control"].checkpoint_distance
        num_steps = self.options["integration_control"].num_steps
        for step in range(
            checkpoint + 1,
            min(num_steps, checkpoint + checkpoint_distance) + 1,
        ):
            self._update_step_info(step)
            self._run_stages_jacvec_rev_forward_part()
            self._update_step_fwd()

    # def _backward_iteration_part(
    #     self, time_stepping_contributions, functional_contributions, checkpoint
    # ):
    #     checkpoint_distance = self.options["integration_control"].checkpoint_distance
    #     num_steps = self.options["integration_control"].num_steps
    #     for step in range(
    #         min(num_steps, checkpoint + checkpoint_distance), checkpoint, -1
    #     ):
    #         self._update_step_info(step)
    #         self._run_stages_jacvec_rev_backward_part(
    #             time_stepping_contributions, functional_contributions
    #         )
    #         self._update_contribution(time_stepping_contributions)
    #         self._update_functional_contribution(functional_contributions)

    def _backward_iteration_part(self, checkpoint, d_outputs, intermediate_d_outputs):
        delta_t = self.options["integration_control"].delta_t
        checkpoint_distance = self.options["integration_control"].checkpoint_distance
        num_steps = self.options["integration_control"].num_steps
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        inner_problem: om.Problem = self.options["inner_problem"]
        quadrature_rule_weights = self.options["quadrature_rule_weights"]
        for step in range(
            min(num_steps, checkpoint + checkpoint_distance), checkpoint, -1
        ):
            checkpoint_step = (
                step % self.options["integration_control"].checkpoint_distance
            )
            self._update_step_info(step)
            new_d_outputs = self._vector_class("linear", "output", self)
            new_d_outputs.set_vec(intermediate_d_outputs)

            for stage in range(butcher_tableau.number_of_stages() - 1, -1, -1):
                self._update_stage_info(stage)
                time_stepping_seed = {}
                functional_seed = {}
                for quantity, var_dict in self.quantity_to_inner_vars.items():
                    time_stepping_seed[var_dict["stage_output_var"]] = (
                        butcher_tableau.butcher_weight_vector[stage]
                        * intermediate_d_outputs[quantity + "_final"]
                    )
                    for prev_stage in range(
                        stage + 1, butcher_tableau.number_of_stages()
                    ):
                        time_stepping_seed[var_dict["stage_output_var"]] += (
                            butcher_tableau.butcher_matrix[prev_stage, stage]
                            * self.d_stage_cache[prev_stage][quantity + "_final"]
                        )
                    if quantity in self.options["integrated_quantities"]:
                        functional_seed[
                            var_dict["stage_output_var"]
                        ] = butcher_tableau.butcher_weight_vector[stage] * (
                            delta_t
                            * quadrature_rule_weights[step]
                            * d_outputs[quantity + "_integrated"]
                            + intermediate_d_outputs[quantity + "_integrated"]
                        )
                        for prev_stage in range(
                            stage + 1, butcher_tableau.number_of_stages()
                        ):
                            functional_seed[var_dict["stage_output_var"]] += (
                                butcher_tableau.butcher_matrix[prev_stage, stage]
                                * self.d_stage_cache[prev_stage][
                                    quantity + "_integrated"
                                ]
                            )
                inner_problem.model._inputs.set_vec(
                    self._time_cache[checkpoint_step][stage]["input"]
                )
                inner_problem.model._outputs.set_vec(
                    self._time_cache[checkpoint_step][stage]["output"]
                )
                inner_problem.model._linearize(None)
                jvp_time_stepping = inner_problem.compute_jacvec_product(
                    self.of_vars, self.wrt_vars, "rev", time_stepping_seed
                )
                if self.of_vars_functional:
                    jvp_functional = inner_problem.compute_jacvec_product(
                        self.of_vars_functional, self.wrt_vars, "rev", functional_seed
                    )
                for quantity, var_dict in self.quantity_to_inner_vars.items():
                    new_d_outputs[quantity + "_final"] += (
                        delta_t * jvp_time_stepping[var_dict["step_input_var"]]
                    )
                    self.d_stage_cache[stage].set_var(
                        quantity + "_final",
                        jvp_time_stepping[var_dict["accumulated_stage_var"]],
                    )
                    if quantity in self.options["integrated_quantities"]:
                        new_d_outputs[quantity + "_integrated"] += (
                            delta_t * jvp_functional[var_dict["step_input_var"]]
                        )
                        self.d_stage_cache[stage].set_var(
                            quantity + "_integrated",
                            jvp_functional[var_dict["accumulated_stage_var"]],
                        )
            intermediate_d_outputs.set_vec(new_d_outputs)

    def _update_stage_info(self, stage):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        self.options["integration_control"].stage = stage
        self.options["integration_control"].stage_time = (
            self.options["integration_control"].step_time_old
            + self.options["integration_control"].delta_t
            * butcher_tableau.butcher_time_stages[stage]
        )
        self.options[
            "integration_control"
        ].butcher_diagonal_element = butcher_tableau.butcher_matrix[stage, stage]

    def _run_stages(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        for stage in range(butcher_tableau.number_of_stages()):
            self._update_stage_info(stage)
            self._dirk_stage(stage)

    def _run_stages_jacvec_fwd(self, seed):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        for stage in range(butcher_tableau.number_of_stages()):
            self._update_stage_info(stage)
            self._dirk_stage(stage)
            self._dirk_stage_jacvec_fwd(stage, seed)

    def _run_stages_jacvec_rev_forward_part(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        for stage in range(butcher_tableau.number_of_stages()):
            self._update_stage_info(stage)
            self._dirk_stage_rev_forward(stage)

    # def _run_stages_jacvec_rev_backward_part(
    #     self, time_stepping_contributions, functional_contributions
    # ):
    #     butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
    #     for stage in range(butcher_tableau.number_of_stages()):
    #         self._update_stage_info(stage)

    #         self.d_stage_cache[stage] = self._dirk_stage_rev_backward(
    #             stage, time_stepping_contributions
    #         )
    #         self.d_stage_functional_cache[stage] = self._dirk_stage_rev_backward(
    #             stage, functional_contributions
    #         )

    def _dirk_stage(self, stage):
        inner_problem: om.Problem = self.options["inner_problem"]
        self._accumulate_stages_into_inner_problem(stage)
        inner_problem.run_model()
        self._cache_stage(stage)

    def _dirk_stage_jacvec_fwd(self, stage, seed):
        inner_problem: om.Problem = self.options["inner_problem"]
        self._accumulate_d_stages_into_seed_fwd(stage, seed)
        inner_problem.model._linearize(None)

        jvp = inner_problem.compute_jacvec_product(
            of=self.of_vars, wrt=self.wrt_vars, mode="fwd", seed=seed
        )

        self._cache_d_stage(stage, jvp)

    def _dirk_stage_rev_forward(self, stage):
        inner_problem: om.Problem = self.options["inner_problem"]
        self._accumulate_stages_into_inner_problem(stage)
        inner_problem.run_model()
        self._cache_stage(stage)
        self._cache_inner_state()

    # def _dirk_stage_rev_backward(self, stage, contributions):
    #     step = (
    #         self.options["integration_control"].step
    #         % self.options["integration_control"].checkpoint_distance
    #     )
    #     inner_problem: om.Problem = self.options["inner_problem"]
    #     butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
    #     self._update_stage_info(stage)

    #     result = self._vector_class("linear", "output", self)
    #     intermediate = self._vector_class("linear", "output", self)

    #     seed = {}
    #     for quantity, var_dict in self.quantity_to_inner_vars.items():
    #         # fill with "d_residual" values (according to documentation)
    #         seed[var_dict["stage_output_var"]] = contributions[
    #             quantity + "_final"
    #         ].copy()

    #     inner_problem.model._inputs.set_vec(self._time_cache[step][stage]["input"])
    #     inner_problem.model._outputs.set_vec(self._time_cache[step][stage]["output"])
    #     inner_problem.model._linearize(None)
    #     jvp = inner_problem.compute_jacvec_product(
    #         self.of_vars, self.wrt_vars, "rev", seed
    #     )

    #     if self.of_vars_functional:
    #         seed_functional = {}
    #         for quantity, var_dict in self.quantity_to_inner_vars.items():
    #             seed_functional[
    #                 var_dict["stage_output_var"]
    #             ] = functional_contributions[quantity + "_integrated"].copy()

    #         jvp_functional = inner_problem.compute_jacvec_product(
    #             self.of_vars_functional, self.wrt_vars, "rev", seed_functional
    #         )

    #     for quantity, var_dict in self.quantity_to_inner_vars.items():
    #         result[quantity + "_final"] += jvp[var_dict["step_input_var"]]
    #         intermediate[quantity + "_final"] = jvp[var_dict["accumulated_stage_var"]]

    #     for prev_stage in range(stage):
    #         self.vector_scaled_add_between_runge_kutta_parts(
    #             result,
    #             butcher_tableau.butcher_matrix[stage, prev_stage],
    #             self._dirk_stage_rev_backward(prev_stage, intermediate),
    #         )

    #     return result

    def _accumulate_stages_into_inner_problem(self, stage: int):
        accumulated_stages = self._vector_class("nonlinear", "input", self)
        # accumulate previous stages for current stage
        self._accumulate_stages(stage, accumulated_stages)

        self._fill_inner_accumulated_stages(accumulated_stages)

        # set accumulated previous stage in inner problem

    def _accumulate_stages(self, stage, accumulated_stages):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        for prev_stage in range(stage):
            self.vector_scaled_add_between_runge_kutta_parts(
                accumulated_stages,
                butcher_tableau.butcher_matrix[stage, prev_stage],
                self.stage_cache[prev_stage],
            )

    def _fill_inner_accumulated_stages(self, vector):
        inner_problem: om.Problem = self.options["inner_problem"]
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            inner_problem.set_val(
                var_dict["accumulated_stage_var"],
                vector[quantity + "_initial"],
            )

    def _update_step_fwd(self):
        step_contributions = self._vector_class("nonlinear", "input", self)

        self._step_contribution(step_contributions)

        self._add_contribution_to_inner_old_step(step_contributions)

    def _step_contribution(self, vector):
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        for stage in range(butcher_tableau.number_of_stages()):
            self.vector_scaled_add_between_runge_kutta_parts(
                vector,
                delta_t * butcher_tableau.butcher_weight_vector[stage],
                self.stage_cache[stage],
            )

    def _add_contribution_to_inner_old_step(self, vector):
        inner_problem: om.Problem = self.options["inner_problem"]
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            inner_problem[var_dict["step_input_var"]] += vector[quantity + "_initial"]

    def _accumulate_d_stages_into_seed_fwd(self, stage, seed):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        # accumulate previous stages for current stage
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            seed[var_dict["accumulated_stage_var"]].fill(0.0)
            for prev_stage in range(stage):
                seed[var_dict["accumulated_stage_var"]] += (
                    butcher_tableau.butcher_matrix[stage, prev_stage]
                    * self.d_stage_cache[prev_stage][quantity + "_initial"]
                )

    def _update_contribution(self, contributions):
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        # compute contribution to new step

        for stage in range(butcher_tableau.number_of_stages()):
            self.vector_scaled_add_between_runge_kutta_parts(
                contributions,
                delta_t * butcher_tableau.butcher_weight_vector[stage],
                self.d_stage_cache[stage],
            )

    # def _update_functional_contribution(self, functional_contributions):
    #     delta_t = self.options["integration_control"].delta_t
    #     butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

    #     for stage in range(butcher_tableau.number_of_stages()):
    #         self.vector_scaled_add_between_runge_kutta_parts(
    #             functional_contributions,
    #             delta_t * butcher_tableau.butcher_weight_vector[stage],
    #             self.d_stage_functional_cache[stage],
    #         )

    def _fwd_seed_step(self, contributions):
        seed = {}
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            # fill with "d_residual" values (according to documentation)
            seed[var_dict["step_input_var"]] = contributions[
                quantity + "_initial"
            ].copy()
            seed[var_dict["accumulated_stage_var"]] = np.zeros_like(
                contributions[quantity + "_initial"]
            )
        return seed

    # cache related functions

    def _setup_stage_cache(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        for stage in range(butcher_tableau.number_of_stages()):
            self.stage_cache[stage] = self._vector_class("nonlinear", "input", self)

    def _setup_d_stage_cache(self, mode):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        for stage in range(butcher_tableau.number_of_stages()):
            self.d_stage_cache[stage] = self._vector_class(
                "linear", "input" if mode == "fwd" else "output", self
            )

    def _cache_stage(self, stage):
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            inner_problem: om.Problem = self.options["inner_problem"]
            new_stage_name = var_dict["stage_output_var"]
            self.stage_cache[stage].set_var(
                quantity + "_initial",
                inner_problem.get_val(new_stage_name),
            )

    def _cache_d_stage(self, stage, jvp):
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            new_stage_name = var_dict["stage_output_var"]
            self.d_stage_cache[stage].set_var(
                quantity + "_initial",
                jvp[new_stage_name],
            )

    def _setup_time_cache(self):
        inner_problem: om.Problem = self.options["inner_problem"]
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        for i in range(self.options["integration_control"].checkpoint_distance):
            self._time_cache[i] = {}
            for j in range(butcher_tableau.number_of_stages()):
                self._time_cache[i][j] = {
                    "input": inner_problem.model._vector_class(
                        "nonlinear", "input", inner_problem.model
                    ),
                    "output": inner_problem.model._vector_class(
                        "nonlinear", "output", inner_problem.model
                    ),
                }

    def _cache_inner_state(self):
        inner_problem: om.Problem = self.options["inner_problem"]
        stage = self.options["integration_control"].stage
        step = self.options["integration_control"].step
        checkpoint_distance = self.options["integration_control"].checkpoint_distance
        self._time_cache[step % checkpoint_distance][stage]["input"].add_scal_vec(
            1.0, inner_problem.model._inputs
        )
        self._time_cache[step % checkpoint_distance][stage]["output"].add_scal_vec(
            1.0, inner_problem.model._outputs
        )

    def _clear_time_cache(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        for i in range(self.options["integration_control"].checkpoint_distance):
            for j in range(butcher_tableau.number_of_stages()):
                self._time_cache[i][j]["input"].imul(0.0)
                self._time_cache[i][j]["output"].imul(0.0)

    # Transfer functions between inner and outer

    def _transfer_input_vector_from_outer_to_inner_problem(self, outer_inputs):
        inner_problem: om.Problem = self.options["inner_problem"]
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            inner_problem.set_val(
                var_dict["step_input_var"],
                outer_inputs[quantity + "_initial"],
            )

    def _transfer_output_vector_from_inner_to_outer_problem(self, outputs):
        inner_problem: om.Problem = self.options["inner_problem"]
        # get the result at the end TODO: how parallel (probably via get_remote option)
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            outputs[quantity + "_final"] = inner_problem.get_val(
                var_dict["step_input_var"]
            )

    # file operation
    def _create_file_and_write_initial_data(self, vector):
        with h5py.File(
            f"{self.comm.rank}_" + self.options["write_file"], mode="w"
        ) as f:
            for quantity in self.quantity_to_inner_vars:
                f.create_dataset(quantity + "/0", data=vector[quantity + "_initial"])

    def _save_to_file_at_checkpoint(self, step):
        inner_problem: om.Problem = self.options["inner_problem"]
        checkpoint_distance = self.options["integration_control"].checkpoint_distance
        num_steps = self.options["integration_control"].num_steps
        if step % checkpoint_distance == 0 or step == num_steps:
            with h5py.File(
                f"{self.comm.rank}_" + self.options["write_file"], mode="r+"
            ) as f:
                for quantity, var_dict in self.quantity_to_inner_vars.items():
                    f.create_dataset(
                        quantity + "/" + str(step),
                        data=inner_problem[var_dict["step_input_var"]],
                    )

    def _write_checkpoint_into_inner_problem(self, checkpoint):
        inner_problem: om.Problem = self.options["inner_problem"]
        with h5py.File(
            f"{self.comm.rank}_" + self.options["write_file"], mode="r"
        ) as f:
            for quantity, var_dict in self.quantity_to_inner_vars.items():
                inner_problem.set_val(
                    var_dict["step_input_var"],
                    f.get(quantity + "/" + str(checkpoint)),
                )

    def vector_scaled_add_between_runge_kutta_parts(
        self, result_vector, scale_vector, initial_vector
    ):
        result_suffix = "_initial" if result_vector._kind == "input" else "_final"
        initial_suffix = "_initial" if initial_vector._kind == "input" else "_final"
        for quantity in self.quantity_to_inner_vars:
            result_vector[quantity + result_suffix] += (
                scale_vector * initial_vector[quantity + initial_suffix]
            )

    def vector_set_between_runge_kutta_parts(self, result_vector, initial_vector):
        result_suffix = "_initial" if result_vector._kind == "input" else "_final"
        initial_suffix = "_initial" if initial_vector._kind == "input" else "_final"
        for quantity in self.quantity_to_inner_vars:
            result_vector[quantity + result_suffix] = initial_vector[
                quantity + initial_suffix
            ]

    def vector_set_between_functional_parts(self, result_vector, initial_vector):
        for quantity in self.quantity_to_inner_vars:
            result_vector[quantity + "_integrated"] = initial_vector[
                quantity + "_integrated"
            ]
