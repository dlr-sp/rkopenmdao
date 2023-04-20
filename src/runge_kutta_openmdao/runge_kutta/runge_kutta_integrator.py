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

    def initialize(self):
        self.options.declare(
            "inner_problem", types=om.Problem, desc="The inner problem"
        )
        self.options.declare(
            "butcher_tableau",
            types=ButcherTableau,
            desc="The btucher tableau for the RK-scheme",
        )
        self.options.declare("integration_control", types=IntegrationControl)

        self.options.declare(
            "write_file",
            types=str,
            default="data.h5",
            desc="The file where the results of each time steps are written.",
        )

        self.options.declare(
            "quantity_tags",
            types=list,
            desc="tags used to differentiate the quantitys",
        )

        self.options.declare("quantity_to_inner_vars", default={})

    def setup(self):
        self.options["inner_problem"].setup()

        self._setup_inputs_and_outputs_and_fill_quantity_to_inner_vars()

        # TODO: maybe add methods for time-independent in/outputs

    def compute(self, inputs, outputs):  # pylint: disable = arguments-differ
        inner_problem: om.Problem = self.options["inner_problem"]
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        num_steps = self.options["integration_control"].num_steps
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        stage_cache = {}
        for stage in range(butcher_tableau.number_of_stages()):
            stage_cache[stage] = om.DefaultVector("nonlinear", "input", self)

        # set initial values
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            old_variable_name = var_tuple[0]
            inner_problem.set_val(old_variable_name, inputs[quantity + "_initial"])

        with h5py.File(self.options["write_file"], mode="w") as f:
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                f.create_dataset(quantity + "/0", data=inputs[quantity + "_initial"])

        for step in range(1, num_steps + 1):
            time_old = initial_time + delta_t * (step - 1)
            time_new = initial_time + delta_t * step
            self._update_step_info(step, time_old, time_new)
            for stage in range(butcher_tableau.number_of_stages()):
                self._update_stage_info(
                    stage,
                    time_old + delta_t * butcher_tableau.butcher_time_stages[stage],
                    butcher_tableau.butcher_matrix[stage, stage],
                )
                self._accumulate_stages_and_set_inner_fwd(stage, stage_cache)

                inner_problem.run_model()

                self._cache_stage(stage, stage_cache)

            self._update_step_fwd(stage_cache)

            if (
                step % self.options["integration_control"].checkpoint_distance == 0
                or step == num_steps
            ):
                with h5py.File(self.options["write_file"], mode="r+") as f:
                    for quantity, var_tuple in self.options[
                        "quantity_to_inner_vars"
                    ].items():
                        f.create_dataset(
                            quantity + "/" + str(step), data=inner_problem[var_tuple[0]]
                        )

        # get the result at the end
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            outputs[quantity + "_final"] = inner_problem.get_val(var_tuple[0])

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode
    ):  # pylint: disable = arguments-differ
        # reverse mode is currently not implemented
        if mode == "fwd":
            self._compute_jacvec_product_fwd(inputs, d_inputs, d_outputs)
        elif mode == "rev":
            self._compute_jacvec_product_rev(inputs, d_inputs, d_outputs)

    def _compute_jacvec_product_fwd(self, inputs, d_inputs, d_outputs):
        inner_problem: om.Problem = self.options["inner_problem"]
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        num_steps = self.options["integration_control"].num_steps
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        stage_cache = {}
        d_stage_cache = {}
        for stage in range(butcher_tableau.number_of_stages()):
            stage_cache[stage] = om.DefaultVector("nonlinear", "input", self)
            d_stage_cache[stage] = om.DefaultVector("linear", "input", self)

        contributions = om.DefaultVector("linear", "input", self)
        contributions.add_scal_vec(1.0, d_inputs)

        of_vars = []
        wrt_vars = []

        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            of_vars.append(var_tuple[2])
            wrt_vars.append(var_tuple[0])
            wrt_vars.append(var_tuple[1])

        # set initial values
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            old_variable_name = var_tuple[0]
            inner_problem.set_val(old_variable_name, inputs[quantity + "_initial"])

        for step in range(1, num_steps + 1):
            seed = {}
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                # fill with "d_residual" values (according to documentation)
                seed[var_tuple[0]] = contributions[quantity + "_initial"].copy()
                seed[var_tuple[1]] = np.zeros_like(contributions[quantity + "_initial"])

            time_old = initial_time + delta_t * (step - 1)
            time_new = initial_time + delta_t * step
            self._update_step_info(step, time_old, time_new)
            for stage in range(butcher_tableau.number_of_stages()):
                self._update_stage_info(
                    stage,
                    time_old + delta_t * butcher_tableau.butcher_time_stages[stage],
                    butcher_tableau.butcher_matrix[stage, stage],
                )
                self._accumulate_stages_and_set_inner_fwd(stage, stage_cache)
                self._accumulate_d_stages_into_seed_fwd(stage, d_stage_cache, seed)

                inner_problem.run_model()
                inner_problem.model._linearize(None)

                jvp = inner_problem.compute_jacvec_product(
                    of=of_vars, wrt=wrt_vars, mode="fwd", seed=seed
                )

                # for quantity, var_tuple in self.options[
                #     "quantity_to_inner_vars"
                # ].items():
                #     contributions[quantity + "_initial"] += (
                #         delta_t
                #         * butcher_tableau.butcher_weight_vector[stage]
                #         * jvp[var_tuple[2]]
                #     )

                self._cache_stage(stage, stage_cache)
                self._cache_d_stage(stage, d_stage_cache, jvp)

            self._update_step_fwd(stage_cache)
            self._update_contribution(d_stage_cache, contributions)

        # get the result at the end
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            d_outputs[quantity + "_final"] += contributions[quantity + "_initial"]

    # TODO: debug

    def _compute_jacvec_product_rev(self, inputs, d_inputs, d_outputs):
        # some variable initializations

        inner_problem: om.Problem = self.options["inner_problem"]
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        num_steps = self.options["integration_control"].num_steps
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        # For the reverse derivative, we need to go backward in time
        # Because of this, we need to start with result of the time-stepping

        time_stepping_result = om.DefaultVector("nonlinear", "output", self)
        with h5py.File(self.options["write_file"], mode="r") as f:
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                time_stepping_result[quantity + "_final"] = f.get(
                    quantity + "/" + str(num_steps)
                )

        stage_cache = {}
        d_stage_cache = {}
        fwd_input_cache = {}
        for stage in range(butcher_tableau.number_of_stages()):
            stage_cache[stage] = om.DefaultVector("nonlinear", "input", self)
            d_stage_cache[stage] = om.DefaultVector("linear", "output", self)
            fwd_input_cache[stage] = om.DefaultVector(
                "nonlinear", "input", inner_problem.model
            )

        contributions = om.DefaultVector("linear", "output", self)
        contributions.add_scal_vec(1.0, d_outputs)

        of_vars = []
        wrt_vars = []

        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            of_vars.append(var_tuple[2])
            wrt_vars.append(var_tuple[0])
            wrt_vars.append(var_tuple[1])

        # set initial values
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            old_variable_name = var_tuple[0]
            inner_problem.set_val(
                old_variable_name, time_stepping_result[quantity + "_final"]
            )

        # time-stepping:
        for step in range(num_steps, 0, -1):
            # first: step backwards in time,
            time_old = initial_time + delta_t * step
            time_new = initial_time + delta_t * (step - 1)
            self._update_step_info(step, time_old, time_new)
            self.options["integration_control"].delta_t *= -1
            for stage in range(butcher_tableau.number_of_stages()):
                self._update_stage_info(
                    stage,
                    time_old + delta_t * butcher_tableau.butcher_time_stages[stage],
                    butcher_tableau.butcher_matrix[stage, stage],
                )

                self._accumulate_stages_and_set_inner_fwd(stage, stage_cache)

                inner_problem.run_model()

                self._cache_stage(stage, stage_cache)

            self._update_step_rev(stage_cache)

            # Note: Caching of this result is not necessary as long as we don't call an _update_step_* again

            # second: step forward, caching the complete input-vector of the inner problem
            self.options["integration_control"].delta_t *= -1
            for stage in range(butcher_tableau.number_of_stages()):
                self._update_stage_info(
                    stage,
                    time_new + delta_t * butcher_tableau.butcher_time_stages[stage],
                    butcher_tableau.butcher_matrix[stage, stage],
                )

                self._accumulate_stages_and_set_inner_fwd(stage, stage_cache)
                inner_problem.run_model()
                self._cache_stage(stage, stage_cache)
                # need to cache complete input_vector for linearize to work later
                fwd_input_cache[stage].add_scal_vec(1.0, inner_problem.model._inputs)

                d_stage_cache[stage] = self._reverse_jacvec_stage(
                    stage, fwd_input_cache, contributions, of_vars, wrt_vars
                )

            self._update_contribution(d_stage_cache, contributions)

        # get the result at the end
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            d_inputs[quantity + "_initial"] += contributions[quantity + "_final"]

        print(d_inputs)

    def _setup_inputs_and_outputs_and_fill_quantity_to_inner_vars(self):
        inner_problem = self.options["inner_problem"]

        step_input_vars = dict(
            inner_problem.model.list_inputs(
                val=False,
                shape=True,
                tags=["step_input_var"],
                out_stream=None,
            )
        )
        for quantity in self.options["quantity_tags"]:
            quantity_inputs = inner_problem.model.list_inputs(
                val=False,
                prom_name=True,
                shape=True,
                tags=[quantity],
                out_stream=None,
            )
            quantity_outputs = inner_problem.model.list_outputs(
                val=False,
                tags=[quantity],
                out_stream=None,
            )
            self.options["quantity_to_inner_vars"][quantity] = [
                None,
                None,
                quantity_outputs[0][0],
            ]
            for var, metadata in quantity_inputs:
                if var in step_input_vars:
                    self.options["quantity_to_inner_vars"][quantity][0] = var
                    self.add_input(quantity + "_initial", shape=metadata["shape"])
                    self.add_output(quantity + "_final", shape=metadata["shape"])
                else:
                    self.options["quantity_to_inner_vars"][quantity][1] = var

    def _update_step_info(self, step, step_time_old, step_time_new):
        self.options["integration_control"].step = step
        self.options["integration_control"].step_time_old = step_time_old
        self.options["integration_control"].step_time_new = step_time_new

    def _update_stage_info(self, stage, stage_time, butcher_diagonal_element):
        self.options["integration_control"].step = stage
        self.options["integration_control"].step_time = stage_time
        self.options[
            "integration_control"
        ].butcher_diagonal_element = butcher_diagonal_element

    def _accumulate_stages_and_set_inner_fwd(self, stage: int, stage_cache: dict):
        inner_problem: om.Problem = self.options["inner_problem"]
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        accumulated_stages = om.DefaultVector("nonlinear", "input", self)
        # accumulate previous stages for current stage
        for prev_stage in range(stage):
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                accumulated_stages[quantity + "_initial"] += (
                    butcher_tableau.butcher_matrix[stage, prev_stage]
                    * stage_cache[prev_stage][quantity + "_initial"]
                )

        # set accumulated previous stage in inner problem
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            accumulated_variable_name = var_tuple[1]
            inner_problem.set_val(
                accumulated_variable_name,
                accumulated_stages[quantity + "_initial"],
            )

    def _accumulate_stages_and_set_inner_rev(self, stage: int, stage_cache: dict):
        inner_problem: om.Problem = self.options["inner_problem"]
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        accumulated_stages = om.DefaultVector("nonlinear", "input", self)
        # accumulate previous stages for current stage
        for prev_stage in range(stage):
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                accumulated_stages[quantity + "_initial"] += (
                    butcher_tableau.butcher_matrix[stage, prev_stage]
                    - butcher_tableau.butcher_weight_vector[stage]
                ) * stage_cache[prev_stage][quantity + "_initial"]

        # set accumulated previous stage in inner problem
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            accumulated_variable_name = var_tuple[1]
            inner_problem.set_val(
                accumulated_variable_name,
                accumulated_stages[quantity + "_initial"],
            )

    def _cache_stage(self, stage, stage_cache):
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            inner_problem: om.Problem = self.options["inner_problem"]
            new_stage_name = var_tuple[2]
            stage_cache[stage].set_var(
                quantity + "_initial",
                inner_problem.get_val(new_stage_name),
            )

    def _update_step_fwd(self, stage_cache):
        inner_problem: om.Problem = self.options["inner_problem"]
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        stage_contributions = om.DefaultVector("nonlinear", "input", self)
        # compute contribution to new step
        for stage in range(butcher_tableau.number_of_stages()):
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                stage_contributions[quantity + "_initial"] += (
                    delta_t
                    * butcher_tableau.butcher_weight_vector[stage]
                    * stage_cache[stage][quantity + "_initial"]
                )

        # add that contribution to the old step
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            inner_problem[var_tuple[0]] += stage_contributions[quantity + "_initial"]

    def _update_step_rev(self, stage_cache):
        inner_problem: om.Problem = self.options["inner_problem"]
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        stage_contributions = om.DefaultVector("nonlinear", "input", self)
        # compute contribution to new step
        for stage in range(butcher_tableau.number_of_stages()):
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                stage_contributions[quantity + "_initial"] += (
                    delta_t
                    * butcher_tableau.butcher_weight_vector[stage]
                    * stage_cache[stage][quantity + "_initial"]
                )

        # add that contribution to the old step
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            inner_problem[var_tuple[0]] -= stage_contributions[quantity + "_initial"]

    def _accumulate_d_stages_into_seed_fwd(self, stage, d_stage_cache, seed):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        # accumulate previous stages for current stage
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            seed[var_tuple[1]].fill(0.0)
            for prev_stage in range(stage):
                seed[var_tuple[1]] += (
                    butcher_tableau.butcher_matrix[stage, prev_stage]
                    * d_stage_cache[prev_stage][quantity + "_initial"]
                )

    def _cache_d_stage(self, stage, d_stage_cache, jvp):
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            new_stage_name = var_tuple[2]
            d_stage_cache[stage].set_var(
                quantity + "_initial",
                jvp[new_stage_name],
            )

    def _update_contribution(self, d_stage_cache, contributions):
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        # compute contribution to new step

        for stage in range(butcher_tableau.number_of_stages()):
            contributions.add_scal_vec(
                delta_t * butcher_tableau.butcher_weight_vector[stage],
                d_stage_cache[stage],
            )

    def _reverse_jacvec_stage(
        self, stage, fwd_input_cache, contributions, of_vars, wrt_vars
    ):
        inner_problem: om.Problem = self.options["inner_problem"]
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        result = om.DefaultVector("linear", "output", self)
        intermediate = om.DefaultVector("linear", "output", self)
        result.add_scal_vec(1.0, contributions)

        seed = {}
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            # fill with "d_residual" values (according to documentation)
            seed[var_tuple[2]] = contributions[quantity + "_final"].copy()

        inner_problem.model._inputs = fwd_input_cache[stage]
        inner_problem.model._linearize(None)
        jvp = inner_problem.compute_jacvec_product(of_vars, wrt_vars, "rev", seed)
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            result[quantity + "_final"] += jvp[var_tuple[0]]
            intermediate[quantity + "_final"] = jvp[var_tuple[1]]

        for prev_stage in range(stage):
            result.add_scal_vec(
                butcher_tableau.butcher_matrix[stage, prev_stage],
                self._reverse_jacvec_stage(
                    prev_stage, fwd_input_cache, intermediate, of_vars, wrt_vars
                ),
            )

        return result
