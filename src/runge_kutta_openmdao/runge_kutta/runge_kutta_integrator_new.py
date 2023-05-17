import openmdao.api as om
import h5py
import numpy as np


from .butcher_tableau import ButcherTableau
from .integration_control import IntegrationControl
from .stage_accumulator import StageAccumulator
from .step_accumulator import StepAccumulator
from .stage_problem_to_explicit_component_wrapper import (
    StageProblemToExplicitComponentWrapper,
)


class RungeKuttaIntegrator(om.ExplicitComponent):
    """
    Outer component for solving time-dependent problems with Runge-Kutta-schemes.
    Needs an inner problem that models one stage of the RK-method.
    The calculation of the value at the next time step is done in this component
    outside of the inner problem.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.quantity_to_inner_vars = {}
        self._step_problem: om.Problem = None

    def initialize(self):
        self.options.declare(
            "inner_problem", types=om.Problem, desc="The inner problem"
        )
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

        # self.options.declare(
        #     "quantity_sizes",
        #     types=dict,
        #     default = None,
        #     desc="""dict of dicts of the form
        #         {quantity : {rank: vector_size},...}
        #         used to distribute the variables across the processes.
        #         The default None only works without parallelization,
        #         and causes the component to copy the size of the
        #         respective quantity from the inner problem""",
        # )

        # self.options.declare(
        #     "quantity_offsets",
        #     types=dict,
        #     default= None,
        #     desc="""dict of dicts of the form
        #         {quantity : {rank: offset},...}
        #         used for the variable offset for the inner variables.
        #         The default None only works without parallelization,
        #         in which case no parallelization is necessary
        #     """,
        # )

    def setup(self):
        self.options["inner_problem"].setup()
        self.options["inner_problem"].final_setup()

        self._setup_inputs_and_outputs_and_fill_quantity_to_inner_vars()

        self._setup_step_problem()

        # TODO: maybe add methods for time-independent in/outputs

    def compute(self, inputs, outputs):  # pylint: disable = arguments-differ
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        num_steps = self.options["integration_control"].num_steps

        # set initial values #TODO: how to parallalize
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            self._step_problem.set_val(
                f"indep.{quantity}_old",
                inputs[quantity + "_initial"],
            )
            self._step_problem.set_val(
                f"indep.{quantity}_accumulated_stages_0", np.zeros(var_dict["shape"])
            )

        # TODO: how does h5py work with parallelization -> probably need file per process?
        with h5py.File(
            f"{self.comm.rank}_" + self.options["write_file"], mode="w"
        ) as f:
            for quantity, var_dict in self.quantity_to_inner_vars.items():
                f.create_dataset(quantity + "/0", data=inputs[quantity + "_initial"])

        for step in range(1, num_steps + 1):
            # TODO: this is probably enough on rank 0
            # if self.comm.rank == 0:
            #     print(f"===starting timestep {step}===")
            time_old = initial_time + delta_t * (step - 1)
            time_new = initial_time + delta_t * step
            self._update_step_info(step, time_old, time_new)

            self._step_problem.run_model()

            # write to file at last time step (and during checkpoints)
            if (
                step % self.options["integration_control"].checkpoint_distance == 0
                or step == num_steps
            ):
                with h5py.File(
                    f"{self.comm.rank}_" + self.options["write_file"], mode="r+"
                ) as f:
                    for quantity, var_dict in self.quantity_to_inner_vars.items():
                        f.create_dataset(
                            quantity + "/" + str(step),
                            data=self._step_problem.get_val(
                                f"step_accumulator.{quantity}_new",
                                get_remote=var_dict["shape"]
                                == var_dict["global_shape"],
                            ),
                        )

            for quantity, var_dict in self.quantity_to_inner_vars.items():
                self._step_problem.set_val(
                    f"indep.{quantity}_old",
                    self._step_problem.get_val(
                        f"step_accumulator.{quantity}_new",
                        get_remote=var_dict["shape"] == var_dict["global_shape"],
                    ),
                )
        # get the result at the end
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            outputs[quantity + "_final"] = self._step_problem.get_val(
                f"step_accumulator.{quantity}_new",
                get_remote=var_dict["shape"] == var_dict["global_shape"],
            )

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode
    ):  # pylint: disable = arguments-differ
        if mode == "fwd":
            self._compute_jacvec_product_fwd(inputs, d_inputs, d_outputs)
        elif mode == "rev":
            self._compute_jacvec_product_rev(inputs, d_inputs, d_outputs)

    def _compute_jacvec_product_fwd(self, inputs, d_inputs, d_outputs):
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        num_steps = self.options["integration_control"].num_steps

        d_outputs.add_scal_vec(1.0, d_inputs)

        of_vars = []
        wrt_vars = []

        for quantity, var_dict in self.quantity_to_inner_vars.items():
            of_vars.append(f"step_accumulator.{quantity}_new")
            wrt_vars.append(f"indep.{quantity}_old")

        # set initial values #TODO: how to parallalize
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            self._step_problem.set_val(
                f"indep.{quantity}_old",
                inputs[quantity + "_initial"],
            )

        for step in range(1, num_steps + 1):
            time_old = initial_time + delta_t * (step - 1)
            time_new = initial_time + delta_t * step
            self._update_step_info(step, time_old, time_new)

            self._step_problem.run_model()
            self._step_problem.model._linearize(None)

            seed = {}
            for quantity, var_dict in self.quantity_to_inner_vars.items():
                # fill with "d_residual" values (according to documentation)
                seed[f"indep.{quantity}_old"] = d_outputs[quantity + "_final"].copy()

            jvp = self._step_problem.compute_jacvec_product(
                of=of_vars, wrt=wrt_vars, mode="fwd", seed=seed
            )

            for quantity, var_dict in self.quantity_to_inner_vars.items():
                d_outputs[quantity + "_final"] = jvp[f"step_accumulator.{quantity}_new"]

            for quantity, var_dict in self.quantity_to_inner_vars.items():
                self._step_problem.set_val(
                    f"indep.{quantity}_old",
                    self._step_problem.get_val(
                        f"step_accumulator.{quantity}_new",
                        get_remote=var_dict["shape"] == var_dict["global_shape"],
                    ),
                )

    def _compute_jacvec_product_rev(self, inputs, d_inputs, d_outputs):
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        num_steps = self.options["integration_control"].num_steps

        time_stepping_result = self._vector_class("nonlinear", "output", self)
        with h5py.File(
            f"{self.comm.rank}_" + self.options["write_file"], mode="r"
        ) as f:
            for quantity, var_dict in self.quantity_to_inner_vars.items():
                time_stepping_result[quantity + "_final"] = f.get(
                    quantity + "/" + str(num_steps)
                )

        d_inputs.add_scal_vec(1.0, d_outputs)

        of_vars = []
        wrt_vars = []

        for quantity, var_dict in self.quantity_to_inner_vars.items():
            of_vars.append(f"step_accumulator.{quantity}_new")
            wrt_vars.append(f"indep.{quantity}_old")

        # set initial values #TODO: how to parallalize
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            self._step_problem.set_val(
                f"indep.{quantity}_old",
                time_stepping_result[quantity + "_final"],
            )

        for step in range(num_steps, 0, -1):
            self.options["integration_control"].delta_t *= -1
            time_old = initial_time + delta_t * step
            time_new = initial_time + delta_t * (step - 1)
            self._update_step_info(step, time_old, time_new)

            self._step_problem.run_model()

            for quantity, var_dict in self.quantity_to_inner_vars.items():
                self._step_problem.set_val(
                    f"indep.{quantity}_old",
                    self._step_problem.get_val(
                        f"step_accumulator.{quantity}_new",
                        get_remote=var_dict["shape"] == var_dict["global_shape"],
                    ),
                )

            self._step_problem.model._linearize(None)

            self.options["integration_control"].delta_t *= -1
            time_old = initial_time + delta_t * (step - 1)
            time_new = initial_time + delta_t * step

            self._step_problem.run_model()

            seed = {}
            for quantity, var_dict in self.quantity_to_inner_vars.items():
                # fill with "d_output" values (according to documentation)
                seed[f"step_accumulator.{quantity}_new"] = d_inputs[
                    quantity + "_initial"
                ].copy()

            jvp = self._step_problem.compute_jacvec_product(
                of=of_vars, wrt=wrt_vars, mode="rev", seed=seed
            )

            for quantity, var_dict in self.quantity_to_inner_vars.items():
                d_inputs[quantity + "_initial"] = jvp[f"indep.{quantity}_old"]

    def _setup_inputs_and_outputs_and_fill_quantity_to_inner_vars(self):
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

        # old_step_input_vars = inner_problem.model.get_io_metadata(
        #     iotypes="input",
        #     metadata_keys=["tags"],
        #     is_indep_var=True,
        #     tags="step_input_var",
        #     get_remote=False,
        #     return_rel_names=True,
        # )

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

        # acc_stage_input_vars = inner_problem.model.get_io_metadata(
        #     iotypes="input",
        #     metadata_keys=["tags"],
        #     is_indep_var=True,
        #     tags="accumulated_stage_var",
        #     get_remote=False,
        #     return_rel_names=True,
        # )

        stage_output_vars = dict(
            inner_problem.model.list_outputs(
                val=False,
                prom_name=True,
                tags=["stage_output_var"],
                out_stream=None,
                all_procs=True,
            )
        )

        # stage_output_vars = inner_problem.model.get_io_metadata(
        #     iotypes="output",
        #     metadata_keys=["tags"],
        #     tags="stage_output_var",
        #     get_remote=False,
        #     return_rel_names=True,
        # )

        for quantity in self.options["quantity_tags"]:
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
            # quantity_inputs = inner_problem.model.get_io_metadata(
            #     iotypes="input",
            #     metadata_keys=["shape", "global_shape", "tags"],
            #     is_indep_var=True,
            #     tags=quantity,
            #     get_remote=False,
            #     return_rel_names=True,
            # )

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

            # quantity_outputs = inner_problem.model.get_io_metadata(
            #     iotypes="output",
            #     metadata_keys=["shape", "global_shape", "tags"],
            #     tags=quantity,
            #     get_remote=False,
            #     return_rel_names=True,
            # )

            self.quantity_to_inner_vars[quantity] = {
                "step_input_var": None,
                "step_input_var_rep": None,
                "accumulated_stage_var": None,
                "accumulated_stage_var_rep": None,
                "stage_output_var": None,
                "stage_output_var_rep": None,
                "shape": None,
                "global_shape": None,
            }

            # TODO: use own error types instead of AssertionError, like in the DevRound

            for var, metadata in quantity_inputs.items():
                if var in old_step_input_vars:
                    self.quantity_to_inner_vars[quantity]["step_input_var"] = var
                    self.quantity_to_inner_vars[quantity][
                        "step_input_var_rep"
                    ] = var.replace(".", "_")
                    if self.quantity_to_inner_vars[quantity]["shape"] is None:
                        self.quantity_to_inner_vars[quantity]["shape"] = metadata[
                            "shape"
                        ]
                    else:
                        assert (
                            metadata["shape"]
                            == self.quantity_to_inner_vars[quantity]["shape"]
                        )
                    if self.quantity_to_inner_vars[quantity]["global_shape"] is None:
                        self.quantity_to_inner_vars[quantity][
                            "global_shape"
                        ] = metadata["global_shape"]
                    else:
                        assert (
                            metadata["global_shape"]
                            == self.quantity_to_inner_vars[quantity]["global_shape"]
                        )

                elif var in acc_stage_input_vars:
                    self.quantity_to_inner_vars[quantity]["accumulated_stage_var"] = var
                    self.quantity_to_inner_vars[quantity][
                        "accumulated_stage_var_rep"
                    ] = var.replace(".", "_")
                    if self.quantity_to_inner_vars[quantity]["shape"] is None:
                        self.quantity_to_inner_vars[quantity]["shape"] = metadata[
                            "shape"
                        ]
                    else:
                        assert (
                            metadata["shape"]
                            == self.quantity_to_inner_vars[quantity]["shape"]
                        )
                    if self.quantity_to_inner_vars[quantity]["global_shape"] is None:
                        self.quantity_to_inner_vars[quantity][
                            "global_shape"
                        ] = metadata["global_shape"]
                    else:
                        assert (
                            metadata["global_shape"]
                            == self.quantity_to_inner_vars[quantity]["global_shape"]
                        )

            for var, metadata in quantity_outputs.items():
                if var in stage_output_vars:
                    self.quantity_to_inner_vars[quantity]["stage_output_var"] = var
                    self.quantity_to_inner_vars[quantity][
                        "stage_output_var_rep"
                    ] = var.replace(".", "_")
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

            # if self.quantity_to_inner_vars[quantity]["stage_output_var"] is None:
            #     raise AssertionError(
            #         f"There is no stage_output_var for quantity {quantity}"
            #     )

            # if (
            #     self.quantity_to_inner_vars[quantity]["step_input_var"] is not None
            #     and self.quantity_to_inner_vars[quantity]["accumulated_stage_var"]
            #     is None
            # ):
            #     raise AssertionError(
            #         f"There needs to be either both or none of step_input_var and accumulated_stage_var for quantity {quantity}"
            #     )

            # if (
            #     self.quantity_to_inner_vars[quantity]["step_input_var"] is None
            #     and self.quantity_to_inner_vars[quantity]["accumulated_stage_var"]
            #     is not None
            # ):
            #     raise AssertionError(
            #         f"There needs to be either both or none of step_input_var and accumulated_stage_var for quantity {quantity}"
            #     )

            self.add_input(
                quantity + "_initial",
                shape=quantity_inputs[
                    self.quantity_to_inner_vars[quantity]["step_input_var"]
                ]["shape"],
                distributed=quantity_inputs[
                    self.quantity_to_inner_vars[quantity]["step_input_var"]
                ]["shape"]
                != quantity_inputs[
                    self.quantity_to_inner_vars[quantity]["step_input_var"]
                ]["global_shape"],
            )
            self.add_output(
                quantity + "_final",
                copy_shape=quantity + "_initial",
                distributed=quantity_inputs[
                    self.quantity_to_inner_vars[quantity]["step_input_var"]
                ]["shape"]
                != quantity_inputs[
                    self.quantity_to_inner_vars[quantity]["step_input_var"]
                ]["global_shape"],
            )

    def _update_step_info(self, step, step_time_old, step_time_new):
        self.options["integration_control"].step = step
        self.options["integration_control"].step_time_old = step_time_old
        self.options["integration_control"].step_time_new = step_time_new

    def _update_stage_info(self, stage, stage_time, butcher_diagonal_element):
        self.options["integration_control"].stage = stage
        self.options["integration_control"].stage_time = stage_time
        self.options[
            "integration_control"
        ].butcher_diagonal_element = butcher_diagonal_element

    def _setup_step_problem(self):
        self._step_problem = om.Problem(comm=self.options["inner_problem"].comm)
        step_model = self._step_problem.model
        butcher: ButcherTableau = self.options["butcher_tableau"]

        input_dict = {}
        output_dict = {}
        ivp = om.IndepVarComp()
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            input_dict[var_dict["step_input_var"]] = var_dict["step_input_var_rep"]
            input_dict[var_dict["accumulated_stage_var"]] = var_dict[
                "accumulated_stage_var_rep"
            ]
            output_dict[var_dict["stage_output_var"]] = var_dict["stage_output_var_rep"]

            ivp.add_output(
                f"{quantity}_old",
                shape=var_dict["shape"],
                distributed=var_dict["shape"] != var_dict["global_shape"],
            )
            ivp.add_output(
                f"{quantity}_accumulated_stages_{0}",
                shape=var_dict["shape"],
                distributed=var_dict["shape"] != var_dict["global_shape"],
            )
        step_model.add_subsystem("indep", ivp)

        for i in range(butcher.number_of_stages()):
            step_model.add_subsystem(
                f"stage_{i}",
                StageProblemToExplicitComponentWrapper(
                    inner_problem=self.options["inner_problem"],
                    input_dict=input_dict,
                    output_dict=output_dict,
                    mystage=i,
                    butcher_time_stage=butcher.butcher_time_stages[i],
                    butcher_diagonal_element=butcher.butcher_matrix[i, i],
                    integration_control=self.options["integration_control"],
                ),
            )
            if i != butcher.number_of_stages() - 1:
                step_model.add_subsystem(
                    f"stage_accumulator_{i+1}",
                    StageAccumulator(
                        my_stage=i + 1,
                        quantity_metadata=self.quantity_to_inner_vars,
                        butcher_tableau=butcher,
                    ),
                )

            else:
                step_model.add_subsystem(
                    "step_accumulator",
                    StepAccumulator(
                        integration_control=self.options["integration_control"],
                        quantity_metadata=self.quantity_to_inner_vars,
                        butcher_tableau=butcher,
                    ),
                )
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            for i in range(butcher.number_of_stages()):
                # ivp old step-> all future stages
                step_model.connect(
                    f"indep.{quantity}_old",
                    f"stage_{i}." + var_dict["step_input_var_rep"],
                )
                # stage -> all future stage accumulators
                for j in range(i + 1, butcher.number_of_stages()):
                    step_model.connect(
                        f"stage_{i}." + var_dict["stage_output_var_rep"],
                        f"stage_accumulator_{j}.{quantity}_stage_{i}",
                    )
                # stage accumulator -> same stage
                if i != 0:
                    step_model.connect(
                        f"stage_accumulator_{i}.{quantity}_accumulated_stages_{i}",
                        f"stage_{i}." + var_dict["accumulated_stage_var_rep"],
                    )
                # stage -> step accumulator
                step_model.connect(
                    f"stage_{i}." + var_dict["stage_output_var_rep"],
                    f"step_accumulator.{quantity}_stage_{i}",
                )
            # ivp initial acc_stages -> first stage
            step_model.connect(
                f"indep.{quantity}_accumulated_stages_{0}",
                "stage_0." + var_dict["accumulated_stage_var_rep"],
            )
            # ivp old step -> step accumulator
            step_model.connect(
                f"indep.{quantity}_old", f"step_accumulator.{quantity}_old"
            )

        self._step_problem.model.nonlinear_solver = om.NonlinearRunOnce()
        self._step_problem.model.linear_solver = om.LinearRunOnce()

        self._step_problem.setup()
        self._step_problem.final_setup()

    def _accumulate_stages_and_set_inner(self, stage: int, stage_cache: dict):
        inner_problem: om.Problem = self.options["inner_problem"]
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        accumulated_stages = self._vector_class("nonlinear", "input", self)
        # accumulate previous stages for current stage
        for prev_stage in range(stage):
            accumulated_stages.add_scal_vec(
                butcher_tableau.butcher_matrix[stage, prev_stage],
                stage_cache[prev_stage],
            )

        # set accumulated previous stage in inner problem TODO: parallel set_val
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            accumulated_variable_name = var_dict["accumulated_stage_var_rep"]
            inner_problem.set_val(
                accumulated_variable_name,
                accumulated_stages[quantity + "_initial"],
            )

    def _cache_stage(self, stage, stage_cache):
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            inner_problem: om.Problem = self.options["inner_problem"]
            new_stage_name = var_dict["stage_output_var_rep"]
            # TODO: parallel
            stage_cache[stage].set_var(
                quantity + "_initial",
                inner_problem.get_val(new_stage_name),
            )

    def _update_step_fwd(self, stage_cache):
        inner_problem: om.Problem = self.options["inner_problem"]
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        stage_contributions = self._vector_class("nonlinear", "input", self)
        # compute contribution to new step
        for stage in range(butcher_tableau.number_of_stages()):
            stage_contributions.add_scal_vec(
                delta_t * butcher_tableau.butcher_weight_vector[stage],
                stage_cache[stage],
            )

        # add that contribution to the old step TODO: parallel
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            inner_problem[var_dict["step_input_var_rep"]] += stage_contributions[
                quantity + "_initial"
            ]

    def _update_step_rev(self, stage_cache):
        inner_problem: om.Problem = self.options["inner_problem"]
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        stage_contributions = self._vector_class("nonlinear", "input", self)
        # compute contribution to new step
        for stage in range(butcher_tableau.number_of_stages()):
            stage_contributions.add_scal_vec(
                delta_t * butcher_tableau.butcher_weight_vector[stage],
                stage_cache[stage],
            )

        # add that contribution to the old step TODO: parallel
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            inner_problem[var_dict["step_input_var_rep"]] -= stage_contributions[
                quantity + "_initial"
            ]

    def _accumulate_d_stages_into_seed_fwd(self, stage, d_stage_cache, seed):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        # accumulate previous stages for current stage
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            seed[var_dict["accumulated_stage_var_rep"]].fill(0.0)
            for prev_stage in range(stage):
                seed[var_dict["accumulated_stage_var_rep"]] += (
                    butcher_tableau.butcher_matrix[stage, prev_stage]
                    * d_stage_cache[prev_stage][quantity + "_initial"]
                )

    def _cache_d_stage(self, stage, d_stage_cache, jvp):
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            new_stage_name = var_dict["stage_output_var_rep"]
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
        delta_t = self.options["integration_control"].delta_t
        step_time = self.options["integration_control"].step_time_new

        self._update_stage_info(
            stage,
            step_time + delta_t * butcher_tableau.butcher_time_stages[stage],
            butcher_tableau.butcher_matrix[stage, stage],
        )

        result = self._vector_class("linear", "output", self)
        intermediate = self._vector_class("linear", "output", self)
        # result.add_scal_vec(1.0, contributions)

        seed = {}
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            # fill with "d_residual" values (according to documentation)
            seed[var_dict["stage_output_var_rep"]] = contributions[
                quantity + "_final"
            ].copy()

        inner_problem.model._inputs = fwd_input_cache[stage]
        inner_problem.model._linearize(None)
        jvp = inner_problem.compute_jacvec_product(of_vars, wrt_vars, "rev", seed)
        for quantity, var_dict in self.quantity_to_inner_vars.items():
            result[quantity + "_final"] += jvp[var_dict["step_input_var_rep"]]
            intermediate[quantity + "_final"] = jvp[
                var_dict["accumulated_stage_var_rep"]
            ]

        for prev_stage in range(stage):
            result.add_scal_vec(
                butcher_tableau.butcher_matrix[stage, prev_stage],
                self._reverse_jacvec_stage(
                    prev_stage, fwd_input_cache, intermediate, of_vars, wrt_vars
                ),
            )

        return result
