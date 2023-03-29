# from typing import Dict, Tuple
import numpy as np
import openmdao.api as om


from src.runge_kutta_openmdao.runge_kutta.runge_kutta import ButcherTableau


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
            desc="The butcher tableau for the RK-scheme",
        )

        self.options.declare(
            "num_steps",
            types=int,
            default=1,
            desc="number of time steps to be computed",
        )
        self.options.declare(
            "initial_time",
            types=float,
            default=0.0,
            desc="Time at the start of the time integration",
        )
        self.options.declare("delta_t", types=float, desc="Size of the time step.")
        self.options.declare(
            "write_file",
            types=(str, None),
            default=None,
            desc="If not none, a file where the results of each time steps are written.",
        )

        self.options.declare(
            "quantity_tags",
            types=list,
            desc="tags used to differentiate the quantitys",
        )

        self.options.declare("quantity_to_inner_vars", default={})

    def setup(self):
        inner_problem: om.Problem = self.options["inner_problem"].setup()

        self._setup_inputs_and_outputs_and_fill_quantity_to_inner_vars()

        # TODO: maybe add methods for time-independent in/outputs

    def compute(self, inputs, outputs):
        inner_problem: om.Problem = self.options["inner_problem"]
        time = self.options["initial_time"]
        delta_t = self.options["delta_t"]
        num_steps = self.options["num_steps"]
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        stage_cache = {}
        for stage in range(butcher_tableau.number_of_stages()):
            stage_cache[stage] = om.DefaultVector("nonlinear", "input", self)

        accumulated_stages = om.DefaultVector("nonlinear", "input", self)

        for subsys in inner_problem.model.system_iter(include_self=True, recurse=True):
            if "delta_t" in subsys.options:
                subsys.options["delta_t"] = self.options["delta_t"]

        # set initial values
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            old_variable_name = var_tuple[0]
            inner_problem.set_val(old_variable_name, inputs[quantity + "_initial"])

        if self.options["write_file"] is not None:
            with open(self.options["write_file"], mode="w", encoding="utf-8") as f:
                first_line = "time"
                second_line = str(time)
                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    first_line += "," + var_tuple[0]
                    second_line += "," + np.array2string(
                        a=inputs[quantity + "_initial"],
                        precision=16,
                        max_line_width=40 * inputs[quantity + "_initial"].size,
                        separator=" ",
                    )
                first_line += "\n"
                second_line += "\n"
                f.write(first_line)
                f.write(second_line)

        for step in range(num_steps):
            for subsys in inner_problem.model.system_iter(
                include_self=True, recurse=True
            ):
                if "step" in subsys.options:
                    subsys.options["step"] = step

            for stage in range(butcher_tableau.number_of_stages()):
                for subsys in inner_problem.model.system_iter(
                    include_self=True, recurse=True
                ):
                    if "stage_time" in subsys.options:
                        subsys.options["stage_time"] = (
                            time + delta_t * butcher_tableau.butcher_time_stages[stage]
                        )
                    if "butcher_diagonal_element" in subsys.options:
                        subsys.options[
                            "butcher_diagonal_element"
                        ] = butcher_tableau.butcher_matrix[stage, stage]
                    if "stage" in subsys.options:
                        subsys.options["stage"] = stage

                # accumulate previous stages for current stage
                for prev_stage in range(stage):
                    for quantity, var_tuple in self.options[
                        "quantity_to_inner_vars"
                    ].items():
                        accumulated_stages[quantity + "_initial"] += (
                            butcher_tableau.butcher_matrix[stage, prev_stage]
                            * stage_cache[prev_stage][quantity + "_initial"]
                        )

                # set accumulated previous stage in inner problem
                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    accumulated_variable_name = var_tuple[1]
                    inner_problem.set_val(
                        accumulated_variable_name,
                        accumulated_stages[quantity + "_initial"],
                    )

                inner_problem.run_model()
                # inner_problem.check_partials()
                # exit()

                # cache computed stage
                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    new_stage_name = var_tuple[2]
                    stage_cache[stage].set_var(
                        quantity + "_initial",
                        inner_problem.get_val(new_stage_name),
                    )
                accumulated_stages.asarray().fill(0.0)
            time = (step + 1) * delta_t

            # compute contribution to new step
            for stage in range(butcher_tableau.number_of_stages()):
                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    accumulated_stages[quantity + "_initial"] += (
                        delta_t
                        * butcher_tableau.butcher_weight_vector[stage]
                        * stage_cache[stage][quantity + "_initial"]
                    )

            # add that contribution to the old step
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                inner_problem[var_tuple[0]] += accumulated_stages[quantity + "_initial"]
            accumulated_stages.asarray().fill(0.0)

            if self.options["write_file"] is not None:
                with open(self.options["write_file"], mode="a", encoding="utf-8") as f:
                    line = str(time)
                    for quantity, var_tuple in self.options[
                        "quantity_to_inner_vars"
                    ].items():
                        line += "," + np.array2string(
                            a=inner_problem[var_tuple[0]],
                            precision=16,
                            max_line_width=40 * inputs[quantity + "_initial"].size,
                            separator=" ",
                        )
                    line += "\n"
                    f.write(line)

        # get the result at the end
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            outputs[quantity + "_final"] = inner_problem.get_val(var_tuple[0])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            self._compute_jacvec_product_fwd(inputs, d_inputs, d_outputs)
        elif mode == "rev":
            self._compute_jacvec_product_rev(inputs, d_inputs, d_outputs)

    def _compute_jacvec_product_fwd(self, inputs, d_inputs, d_outputs):
        inner_problem: om.Problem = self.options["inner_problem"]
        time = self.options["initial_time"]
        delta_t = self.options["delta_t"]
        num_steps = self.options["num_steps"]
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        stage_cache = {}
        for stage in range(butcher_tableau.number_of_stages()):
            stage_cache[stage] = om.DefaultVector("nonlinear", "input", self)

        accumulated_stages = om.DefaultVector("nonlinear", "input", self)

        contributions = om.DefaultVector("linear", "input", self)
        contributions.add_scal_vec(1.0, d_inputs)

        of_vars = []
        wrt_vars = []

        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            of_vars.append(var_tuple[2])
            wrt_vars.append(var_tuple[0])

        for subsys in inner_problem.model.system_iter(include_self=True, recurse=True):
            if "delta_t" in subsys.options:
                subsys.options["delta_t"] = self.options["delta_t"]

        # set initial values
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            old_variable_name = var_tuple[0]
            inner_problem.set_val(old_variable_name, inputs[quantity + "_initial"])

        for step in range(num_steps):
            for subsys in inner_problem.model.system_iter(
                include_self=True, recurse=True
            ):
                if "step" in subsys.options:
                    subsys.options["step"] = step

            seed = {}
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                # fill witth "d_residual" values (according to documentation)
                seed[var_tuple[0]] = contributions[quantity + "_initial"].copy()

            for stage in range(butcher_tableau.number_of_stages()):
                for subsys in inner_problem.model.system_iter(
                    include_self=True, recurse=True
                ):
                    if "stage_time" in subsys.options:
                        subsys.options["stage_time"] = (
                            time + delta_t * butcher_tableau.butcher_time_stages[stage]
                        )
                    if "butcher_diagonal_element" in subsys.options:
                        subsys.options[
                            "butcher_diagonal_element"
                        ] = butcher_tableau.butcher_matrix[stage, stage]
                    if "stage" in subsys.options:
                        subsys.options["stage"] = stage

                # accumulate previous stages for current stage
                for prev_stage in range(stage):
                    for quantity, var_tuple in self.options[
                        "quantity_to_inner_vars"
                    ].items():
                        accumulated_stages[quantity + "_initial"] += (
                            butcher_tableau.butcher_matrix[stage, prev_stage]
                            * stage_cache[prev_stage][quantity + "_initial"]
                        )

                # set accumulated previous stage in inner problem
                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    accumulated_variable_name = var_tuple[1]
                    inner_problem.set_val(
                        accumulated_variable_name,
                        accumulated_stages[quantity + "_initial"],
                    )

                inner_problem.run_model()

                jvp = inner_problem.compute_jacvec_product(
                    of=of_vars, wrt=wrt_vars, mode="fwd", seed=seed
                )

                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    contributions[quantity + "_initial"] += (
                        delta_t
                        * butcher_tableau.butcher_weight_vector[stage]
                        * jvp[var_tuple[2]]
                    )

                # cache computed stage
                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    new_stage_name = var_tuple[2]
                    stage_cache[stage].set_var(
                        quantity + "_initial",
                        inner_problem.get_val(new_stage_name),
                    )
                accumulated_stages.asarray().fill(0.0)
            time = (step + 1) * delta_t
            # print(time)

            # compute contribution to new step
            for stage in range(butcher_tableau.number_of_stages()):
                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    accumulated_stages[quantity + "_initial"] += (
                        delta_t
                        * butcher_tableau.butcher_weight_vector[stage]
                        * stage_cache[stage][quantity + "_initial"]
                    )

            # add that contribution to the old step
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                inner_problem[var_tuple[0]] += accumulated_stages[quantity + "_initial"]
            accumulated_stages.asarray().fill(0.0)

        # get the result at the end
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            d_outputs[quantity + "_final"] += contributions[quantity + "_initial"]

    def _compute_jacvec_product_rev(self, inputs, d_inputs, d_outputs):
        outputs = om.DefaultVector("nonlinear", "output", self)
        self.compute(inputs, outputs)

        inner_problem: om.Problem = self.options["inner_problem"]
        time = self.options["initial_time"]
        delta_t = self.options["delta_t"]
        num_steps = self.options["num_steps"]
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        time = time + (num_steps - 1) * delta_t

        stage_cache = {}
        for stage in range(butcher_tableau.number_of_stages()):
            stage_cache[stage] = om.DefaultVector("nonlinear", "input", self)

        accumulated_stages = om.DefaultVector("nonlinear", "input", self)

        contributions = om.DefaultVector("linear", "output", self)
        contributions.add_scal_vec(1.0, d_outputs)

        of_vars = []
        wrt_vars = []

        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            of_vars.append(var_tuple[2])
            wrt_vars.append(var_tuple[0])

        for subsys in inner_problem.model.system_iter(include_self=True, recurse=True):
            if "delta_t" in subsys.options:
                subsys.options["delta_t"] = self.options["delta_t"]

        # set initial values
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            old_variable_name = var_tuple[0]
            inner_problem.set_val(old_variable_name, outputs[quantity + "_final"])

        for step in range(num_steps - 1, -1, -1):
            for subsys in inner_problem.model.system_iter(
                include_self=True, recurse=True
            ):
                if "step" in subsys.options:
                    subsys.options["step"] = step

            seed = {}
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                # fill witth "d_residual" values (according to documentation)
                seed[var_tuple[2]] = contributions[quantity + "_final"].copy()

            for stage in range(butcher_tableau.number_of_stages()):
                for subsys in inner_problem.model.system_iter(
                    include_self=True, recurse=True
                ):
                    if "stage_time" in subsys.options:
                        subsys.options["stage_time"] = (
                            time + delta_t * butcher_tableau.butcher_time_stages[stage]
                        )
                    if "butcher_diagonal_element" in subsys.options:
                        subsys.options["butcher_diagonal_element"] = (
                            butcher_tableau.butcher_matrix[stage, stage]
                            - butcher_tableau.butcher_weight_vector[stage]
                        )
                    if "stage" in subsys.options:
                        subsys.options["stage"] = stage

                # accumulate previous stages for current stage
                for prev_stage in range(stage):
                    for quantity, var_tuple in self.options[
                        "quantity_to_inner_vars"
                    ].items():
                        accumulated_stages[quantity + "_initial"] += (
                            butcher_tableau.butcher_matrix[stage, prev_stage]
                            * butcher_tableau.butcher_weight_vector[stage]
                        ) * stage_cache[prev_stage][quantity + "_initial"]

                # set accumulated previous stage in inner problem
                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    accumulated_variable_name = var_tuple[1]
                    inner_problem.set_val(
                        accumulated_variable_name,
                        accumulated_stages[quantity + "_initial"],
                    )

                inner_problem.run_model()

                jvp = inner_problem.compute_jacvec_product(
                    of=of_vars, wrt=wrt_vars, mode="rev", seed=seed
                )

                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    contributions[quantity + "_final"] += (
                        delta_t
                        * butcher_tableau.butcher_weight_vector[stage]
                        * jvp[var_tuple[0]]
                    )

                # cache computed stage
                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    new_stage_name = var_tuple[2]
                    stage_cache[stage].set_var(
                        quantity + "_initial",
                        inner_problem.get_val(new_stage_name),
                    )
                accumulated_stages.asarray().fill(0.0)

            time = (step - 1) * delta_t

            # print(time)

            # compute contribution to new step
            for stage in range(butcher_tableau.number_of_stages()):
                for quantity, var_tuple in self.options[
                    "quantity_to_inner_vars"
                ].items():
                    accumulated_stages[quantity + "_initial"] += (
                        delta_t
                        * butcher_tableau.butcher_weight_vector[stage]
                        * stage_cache[stage][quantity + "_initial"]
                    )

            # add that contribution to the old step
            for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
                inner_problem[var_tuple[0]] -= accumulated_stages[quantity + "_initial"]
            accumulated_stages.asarray().fill(0.0)

        # get the result at the end
        for quantity, var_tuple in self.options["quantity_to_inner_vars"].items():
            d_inputs[quantity + "_initial"] += contributions[quantity + "_final"]

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
