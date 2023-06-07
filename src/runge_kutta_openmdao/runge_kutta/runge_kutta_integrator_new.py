import openmdao.api as om
import numpy as np
import pyrevolve as pr

from .butcher_tableau import ButcherTableau
from .integration_control import IntegrationControl
from .runge_kutta_scheme import RungeKuttaScheme
from .inner_problem_computation_functors import (
    InnerProblemComputeFunctor,
    InnerProblemComputeJacvecFunctor,
    InnerProblemComputeTransposeJacvecFunctor,
)

from .runge_kutta_integrator_pyrevolve_classes import (
    RungeKuttaIntegratorSymbol,
    RungeKuttaCheckpoint,
    RungeKuttaForwardOperator,
    RungeKuttaReverseOperator,
)


class RungeKuttaIntegrator(om.ExplicitComponent):
    """
    Outer component for solving time-dependent problems with Runge-Kutta-schemes.
    Needs an inner problem that models one stage of the RK-method.
    The calculation of the value at the next time step is done in this component
    outside the inner problem.
    """

    quantity_metadata: dict
    stage_cache: dict
    d_stage_cache: dict
    d_stage_cache_functional: dict
    _time_cache: dict
    of_vars: list
    wrt_vars: list

    runge_kutta_scheme: RungeKuttaScheme

    numpy_array_size: int
    numpy_functional_size: int

    cached_input: np.ndarray
    revolver: pr.BaseRevolver
    forward_operator: RungeKuttaForwardOperator
    reverse_operator: RungeKuttaReverseOperator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quantity_metadata = {}
        self.stage_cache = {}
        self.d_stage_cache = {}
        self.d_stage_cache_functional = {}
        self._time_cache = {}
        self.of_vars = []
        self.wrt_vars = []

        self.runge_kutta_scheme = None

        self.numpy_array_size = 0
        self.numpy_functional_size = 0

        self.cached_input = None
        self.revolver = None
        self.forward_operator: RungeKuttaForwardOperator = None
        self.reverse_operator: RungeKuttaReverseOperator = None

    def initialize(self):
        self.options.declare("inner_problem", types=om.Problem, desc="The inner problem")

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
            desc="tags used to differentiate the quantities",
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
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        assert set(self.options["integrated_quantities"]).issubset(
            set(self.options["quantity_tags"])
        )
        self.options["inner_problem"].setup()
        self.options["inner_problem"].final_setup()

        self._setup_variable_information()

        self._setup_wrt_and_of_vars()

        self.setup_runge_kutta_scheme()

        self.forward_operator = RungeKuttaForwardOperator(
            self.numpy_array_size,
            self.numpy_functional_size,
            butcher_tableau.number_of_stages(),
            self._run_step,
        )

        self.reverse_operator = RungeKuttaReverseOperator(
            self.numpy_array_size, butcher_tableau.number_of_stages(), self._run_step_jacvec_rev
        )

        # TODO: maybe add methods for time-independent in/outputs

    def compute(self, inputs, outputs):  # pylint: disable = arguments-differ
        num_steps = self.options["integration_control"].num_steps
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        quadrature_rule_weights = self.options["quadrature_rule_weights"]

        serialized_state_symbol = RungeKuttaIntegratorSymbol(self.numpy_array_size)
        self.to_numpy_array(inputs, serialized_state_symbol.data)
        self.cached_input = serialized_state_symbol.data.copy()

        checkpoint_dict = {"serialized_state": serialized_state_symbol}

        functional_part = np.zeros(self.numpy_functional_size)
        if self.options["integrated_quantities"]:
            functional_part.data += (
                delta_t
                * quadrature_rule_weights[0]
                * self.get_functional_contribution(serialized_state_symbol.data)
            )

        checkpoint = RungeKuttaCheckpoint(checkpoint_dict)

        self.forward_operator.serialized_state_symbol = serialized_state_symbol
        self.forward_operator.functional_part = functional_part

        self.revolver = pr.Revolver(
            checkpoint,
            self.forward_operator,
            self.reverse_operator,
            num_steps if num_steps == 1 else None,
            num_steps,
        )

        self.revolver.apply_forward()

        self.from_numpy_array(serialized_state_symbol.data, outputs)

        if self.options["integrated_quantities"]:
            self.add_functional_part_to_om_vec(functional_part, outputs)

    def _run_step(self, step, serialized_state, functional_part, stage_cache, accumulated_stages):
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        quadrature_rule_weights = self.options["quadrature_rule_weights"]
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        self.options["integration_control"].step = step
        time = initial_time + step * delta_t
        self.options["integration_control"].step_time_old = time
        self.options["integration_control"].step_time_new = time + delta_t
        for stage in range(butcher_tableau.number_of_stages()):
            self.options["integration_control"].stage = stage
            if stage != 0:
                accumulated_stages = self.runge_kutta_scheme.compute_accumulated_stages(
                    stage, stage_cache
                )
            else:
                accumulated_stages.fill(0.0)
            stage_cache[stage, :] = self.runge_kutta_scheme.compute_stage(
                stage, delta_t, time, serialized_state, accumulated_stages
            )
        serialized_state = self.runge_kutta_scheme.compute_step(
            delta_t, serialized_state, stage_cache
        )
        if self.options["integrated_quantities"]:
            functional_part += (
                delta_t
                * quadrature_rule_weights[step]
                * self.get_functional_contribution(serialized_state)
            )
        return serialized_state, functional_part

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode
    ):  # pylint: disable = arguments-differ
        if mode == "fwd":
            print("fwd_mode_jacvec")
            self._compute_jacvec_product_fwd(inputs, d_inputs, d_outputs)
        elif mode == "rev":
            print("rev_mode_jacvec")
            self._compute_jacvec_product_rev(inputs, d_inputs, d_outputs)

    def _compute_jacvec_product_fwd(self, inputs, d_inputs, d_outputs):
        num_steps = self.options["integration_control"].num_steps
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        quadrature_rule_weights = self.options["quadrature_rule_weights"]

        serialized_state = np.zeros(self.numpy_array_size)
        self.to_numpy_array(inputs, serialized_state)
        serialized_state_perturbations = np.zeros(self.numpy_array_size)
        self.to_numpy_array(d_inputs, serialized_state_perturbations)

        if self.options["integrated_quantities"]:
            functional_part_perturbations = np.zeros(self.numpy_functional_size)
            functional_part_perturbations += (
                delta_t
                * quadrature_rule_weights[0]
                * self.get_functional_contribution(serialized_state_perturbations)
            )

        stage_cache = np.zeros((butcher_tableau.number_of_stages(), serialized_state.size))
        stage_perturbations_cache = np.zeros(
            (butcher_tableau.number_of_stages(), serialized_state.size)
        )
        accumulated_stages = np.zeros_like(serialized_state)
        accumulated_stage_perturbations = np.zeros_like(serialized_state)

        for step in range(1, num_steps + 1):
            time = initial_time + step * delta_t
            for stage in range(butcher_tableau.number_of_stages()):
                if stage != 0:
                    accumulated_stages = self.runge_kutta_scheme.compute_accumulated_stages(
                        stage, stage_cache
                    )
                    accumulated_stage_perturbations = (
                        self.runge_kutta_scheme.compute_accumulated_stage_perturbations(
                            stage, stage_perturbations_cache
                        )
                    )
                else:
                    accumulated_stages.fill(0.0)
                    accumulated_stage_perturbations.fill(0.0)
                stage_cache[stage, :] = self.runge_kutta_scheme.compute_stage(
                    stage, delta_t, time, serialized_state, accumulated_stages
                )
                stage_perturbations_cache[stage, :] = self.runge_kutta_scheme.compute_stage_jacvec(
                    stage,
                    delta_t,
                    time,
                    serialized_state_perturbations,
                    accumulated_stage_perturbations,
                )
            serialized_state = self.runge_kutta_scheme.compute_step(
                delta_t, serialized_state, stage_cache
            )
            serialized_state_perturbations = self.runge_kutta_scheme.compute_step_jacvec(
                delta_t, serialized_state_perturbations, stage_perturbations_cache
            )
            if self.options["integrated_quantities"]:
                functional_part_perturbations += (
                    delta_t
                    * quadrature_rule_weights[step]
                    * self.get_functional_contribution(serialized_state_perturbations)
                )

        self.from_numpy_array(serialized_state_perturbations, d_outputs)
        if self.options["integrated_quantities"]:
            self.add_functional_part_to_om_vec(functional_part_perturbations, d_outputs)

    def _compute_jacvec_product_rev(self, inputs, d_inputs, d_outputs):
        quadrature_rule_weights = self.options["quadrature_rule_weights"]
        delta_t = self.options["integration_control"].delta_t
        serialized_state = np.zeros(self.numpy_array_size)
        self.to_numpy_array(inputs, serialized_state)
        if not np.array_equal(self.cached_input, serialized_state) or self.revolver is None:
            outputs = self._vector_class("nonlinear", "output", self)
            self.compute(inputs, outputs)

        serialized_state_perturbations = np.zeros(self.numpy_array_size)
        self.to_numpy_array(d_outputs, serialized_state_perturbations)

        self.reverse_operator.serialized_state_perturbations = serialized_state_perturbations
        if self.options["integrated_quantities"]:
            functional_perturbations = np.zeros(self.numpy_array_size)
            self.functional_contribution_from_om_output_vec(d_outputs, functional_perturbations)
            self.reverse_operator.functional_perturbations = (
                quadrature_rule_weights[-1] * functional_perturbations
            )
            self.reverse_operator.original_functional_perturbations = functional_perturbations

        self.revolver.apply_reverse()

        self.from_numpy_array(self.reverse_operator.serialized_state_perturbations, d_inputs)
        if self.options["integrated_quantities"]:
            d_inputs_functional = self._vector_class("linear", "input", self)
            self.from_numpy_array(
                self.reverse_operator.functional_perturbations, d_inputs_functional
            )
            d_inputs.add_scal_vec(delta_t, d_inputs_functional)

        self.revolver = None

    def _run_step_jacvec_rev(
        self,
        step,
        serialized_state,
        stage_cache,
        accumulated_stages,
        serialized_state_perturbations,
        stage_perturbations_cache,
        functional_perturbations,
        functional_stage_perturbations_cache,
        original_functional_perturbations,
    ):
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        quadrature_rule_weights = self.options["quadrature_rule_weights"]
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        self.options["integration_control"].step = step
        time = initial_time + step * delta_t
        self.options["integration_control"].step_time_old = time
        self.options["integration_control"].step_time_new = time + delta_t
        for stage in range(butcher_tableau.number_of_stages()):
            self.options["integration_control"].stage = stage
            if stage != 0:
                accumulated_stages = self.runge_kutta_scheme.compute_accumulated_stages(
                    stage, stage_cache
                )
            else:
                accumulated_stages.fill(0.0)
            stage_cache[stage, :] = self.runge_kutta_scheme.compute_stage(
                stage, delta_t, time, serialized_state, accumulated_stages
            )
        new_serialized_state_perturbations = serialized_state_perturbations.copy()
        new_functional_perturbations = functional_perturbations.copy()
        for stage in range(butcher_tableau.number_of_stages() - 1, -1, -1):
            linearization_args = {}
            if stage != butcher_tableau.number_of_stages():
                linearization_args[
                    "numpy_acc_stage_vec"
                ] = self.runge_kutta_scheme.compute_accumulated_stages(stage, stage_cache)
            joined_perturbations = (
                self.runge_kutta_scheme.join_new_state_and_accumulated_stages_perturbations(
                    stage, serialized_state_perturbations, stage_perturbations_cache
                )
            )
            (
                wrt_old_state,
                stage_perturbations_cache[stage, :],
            ) = self.runge_kutta_scheme.compute_stage_transposed_jacvec(
                stage, delta_t, time, joined_perturbations, **linearization_args
            )
            new_serialized_state_perturbations += delta_t * wrt_old_state
            if self.options["integrated_quantities"]:
                functional_joined_perturbations = (
                    self.runge_kutta_scheme.join_new_state_and_accumulated_stages_perturbations(
                        stage, functional_perturbations, functional_stage_perturbations_cache
                    )
                )
                (
                    functional_wrt_old_state,
                    functional_stage_perturbations_cache[stage, :],
                ) = self.runge_kutta_scheme.compute_stage_transposed_jacvec(
                    stage,
                    delta_t,
                    time,
                    functional_joined_perturbations,
                )
                new_functional_perturbations += delta_t * functional_wrt_old_state

        return (
            new_serialized_state_perturbations,
            new_functional_perturbations
            + quadrature_rule_weights[step - 1] * original_functional_perturbations,
        )

    def _setup_variable_information(self):
        inner_problem: om.Problem = self.options["inner_problem"]
        self.numpy_array_size = 0
        old_step_input_vars = inner_problem.model.get_io_metadata(
            iotypes="input",
            metadata_keys=["tags"],
            tags=["step_input_var"],
            get_remote=False,
        )

        acc_stage_input_vars = inner_problem.model.get_io_metadata(
            iotypes="input",
            metadata_keys=["tags"],
            tags=["accumulated_stage_var"],
            get_remote=False,
        )

        stage_output_vars = inner_problem.model.get_io_metadata(
            iotypes="output",
            metadata_keys=["tags"],
            tags=["stage_output_var"],
            get_remote=False,
        )

        for quantity in self.options["quantity_tags"]:
            self._extract_quantity_metadata_from_inner_problem(
                quantity, old_step_input_vars, acc_stage_input_vars, stage_output_vars
            )
            if (
                (self.quantity_metadata[quantity]["step_input_var"] is None)
                and (self.quantity_metadata[quantity]["accumulated_stage_var"] is None)
                and (self.quantity_metadata[quantity]["stage_output_var"] is None)
            ):
                del self.quantity_metadata[quantity]
            else:
                self.quantity_metadata[quantity]["numpy_start_index"] = self.numpy_array_size
                self.numpy_array_size += np.prod(self.quantity_metadata[quantity]["shape"])
                if quantity in self.options["integrated_quantities"]:
                    self.quantity_metadata[quantity][
                        "numpy_functional_start_index"
                    ] = self.numpy_functional_size
                    self.numpy_functional_size += np.prod(self.quantity_metadata[quantity]["shape"])
                self._add_inputs_and_outputs(quantity)

    def _extract_quantity_metadata_from_inner_problem(
        self, quantity, old_step_input_vars, acc_stage_input_vars, stage_output_vars
    ):
        inner_problem: om.Problem = self.options["inner_problem"]

        quantity_inputs = inner_problem.model.get_io_metadata(
            iotypes="input",
            metadata_keys=["tags", "shape", "global_shape"],
            tags=[quantity],
            get_remote=False,
        )

        quantity_outputs = inner_problem.model.get_io_metadata(
            iotypes="output",
            metadata_keys=["tags", "shape", "global_shape"],
            tags=[quantity],
            get_remote=False,
        )

        self.quantity_metadata[quantity] = {
            "step_input_var": None,
            "accumulated_stage_var": None,
            "stage_output_var": None,
            "shape": None,
            "global_shape": None,
            "local_indices_start": None,
        }

        # TODO: use own error types instead of AssertionError, like in the DevRound

        for var, metadata in quantity_inputs.items():
            if var in old_step_input_vars:
                self.quantity_metadata[quantity]["step_input_var"] = var
                if self.quantity_metadata[quantity]["shape"] is None:
                    self.quantity_metadata[quantity]["shape"] = metadata["shape"]
                else:
                    assert metadata["shape"] == self.quantity_metadata[quantity]["shape"]
                if self.quantity_metadata[quantity]["global_shape"] is None:
                    self.quantity_metadata[quantity]["global_shape"] = metadata["global_shape"]
                else:
                    assert (
                        metadata["global_shape"] == self.quantity_metadata[quantity]["global_shape"]
                    )

            elif var in acc_stage_input_vars:
                self.quantity_metadata[quantity]["accumulated_stage_var"] = var
                if self.quantity_metadata[quantity]["shape"] is None:
                    self.quantity_metadata[quantity]["shape"] = metadata["shape"]
                else:
                    assert metadata["shape"] == self.quantity_metadata[quantity]["shape"]
                if self.quantity_metadata[quantity]["global_shape"] is None:
                    self.quantity_metadata[quantity]["global_shape"] = metadata["global_shape"]
                else:
                    assert (
                        metadata["global_shape"] == self.quantity_metadata[quantity]["global_shape"]
                    )

        for var, metadata in quantity_outputs.items():
            if var in stage_output_vars:
                self.quantity_metadata[quantity]["stage_output_var"] = var
            if self.quantity_metadata[quantity]["shape"] is None:
                self.quantity_metadata[quantity]["shape"] = metadata["shape"]
            else:
                assert metadata["shape"] == self.quantity_metadata[quantity]["shape"]
            if self.quantity_metadata[quantity]["global_shape"] is None:
                self.quantity_metadata[quantity]["global_shape"] = metadata["global_shape"]
            else:
                assert metadata["global_shape"] == self.quantity_metadata[quantity]["global_shape"]
            if (
                self.quantity_metadata[quantity]["shape"]
                != self.quantity_metadata[quantity]["global_shape"]
            ):
                sizes = inner_problem.model._var_sizes["output"][
                    :, inner_problem.model._var_allprocs_abs2idx[var]
                ]
                self.quantity_metadata[quantity]["local_indices_start"] = start = np.sum(
                    sizes[: self.comm.rank]
                )
                self.quantity_metadata[quantity]["local_indices_end"] = (
                    start + sizes[self.comm.rank]
                )
            else:
                self.quantity_metadata[quantity]["local_indices_start"] = 0
                self.quantity_metadata[quantity]["local_indices_end"] = np.prod(
                    self.quantity_metadata[quantity]["shape"]
                )

    def _add_inputs_and_outputs(self, quantity):
        self.add_input(
            quantity + "_initial",
            shape=self.quantity_metadata[quantity]["shape"],
            distributed=self.quantity_metadata[quantity]["shape"]
            != self.quantity_metadata[quantity]["global_shape"],
        )
        self.add_output(
            quantity + "_final",
            copy_shape=quantity + "_initial",
            distributed=self.quantity_metadata[quantity]["shape"]
            != self.quantity_metadata[quantity]["global_shape"],
        )
        if quantity in self.options["integrated_quantities"]:
            self.add_output(
                quantity + "_integrated",
                copy_shape=quantity + "_initial",
                distributed=self.quantity_metadata[quantity]["shape"]
                != self.quantity_metadata[quantity]["global_shape"],
            )

    def _setup_wrt_and_of_vars(self):
        for var_dict in self.quantity_metadata.values():
            self.of_vars.append(var_dict["stage_output_var"])
            self.wrt_vars.append(var_dict["step_input_var"])
            self.wrt_vars.append(var_dict["accumulated_stage_var"])

    def to_numpy_array(self, om_vector: om.DefaultVector, np_array: np.ndarray):
        if om_vector._kind == "input":
            name_suffix = "_initial"
        else:
            name_suffix = "_final"

        for quantity, metadata in self.quantity_metadata.items():
            start = metadata["numpy_start_index"]
            end = start + np.prod(metadata["shape"])
            np_array[start:end] = om_vector[quantity + name_suffix].flatten()

    def from_numpy_array(self, np_array: np.ndarray, om_vector: om.DefaultVector):
        if om_vector._kind == "input":
            name_suffix = "_initial"
        else:
            name_suffix = "_final"

        for quantity, metadata in self.quantity_metadata.items():
            start = metadata["numpy_start_index"]
            end = start + np.prod(metadata["shape"])
            om_vector[quantity + name_suffix] = np_array[start:end].reshape(metadata["shape"])

    def setup_runge_kutta_scheme(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        self.runge_kutta_scheme = RungeKuttaScheme(
            butcher_tableau,
            InnerProblemComputeFunctor(
                self.options["inner_problem"],
                self.options["integration_control"],
                self.quantity_metadata,
            ),
            InnerProblemComputeJacvecFunctor(
                self.options["inner_problem"],
                self.options["integration_control"],
                self.quantity_metadata,
                self.of_vars,
                self.wrt_vars,
            ),
            InnerProblemComputeTransposeJacvecFunctor(
                self.options["inner_problem"],
                self.options["integration_control"],
                self.quantity_metadata,
                self.of_vars,
                self.wrt_vars,
            ),
        )

    def get_functional_contribution(self, serialized_state: np.ndarray) -> np.ndarray:
        contribution = np.zeros(self.numpy_functional_size)
        for quantity in self.options["integrated_quantities"]:
            start_time_stepping = self.quantity_metadata[quantity]["numpy_start_index"]
            start_functional = self.quantity_metadata[quantity]["numpy_functional_start_index"]
            length = np.prod(self.quantity_metadata[quantity]["shape"])
            contribution[start_functional : start_functional + length] = serialized_state[
                start_time_stepping : start_time_stepping + length
            ]
        return contribution

    def functional_contribution_from_om_output_vec(
        self, output_vec: om.DefaultVector, array: np.ndarray
    ):
        for quantity in self.options["integrated_quantities"]:
            start = self.quantity_metadata[quantity]["numpy_start_index"]
            end = start + np.prod(self.quantity_metadata[quantity]["shape"])
            array[start:end] = output_vec[quantity + "_integrated"].flatten()

    def add_functional_part_to_om_vec(
        self, functional_numpy_array: np.ndarray, om_vector: om.DefaultVector
    ):
        for quantity in self.options["integrated_quantities"]:
            start = self.quantity_metadata[quantity]["numpy_functional_start_index"]
            end = start + np.prod(self.quantity_metadata[quantity]["shape"])
            om_vector[quantity + "_integrated"] += functional_numpy_array[start:end].reshape(
                self.quantity_metadata[quantity]["shape"]
            )
