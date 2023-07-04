import h5py
import numpy as np
import openmdao.api as om
from openmdao.vectors.vector import Vector as om_vector
import pyrevolve as pr

from .butcher_tableau import ButcherTableau
from .functional_coefficients import FunctionalCoefficients, EmptyFunctionalCoefficients
from .integration_control import IntegrationControl
from .runge_kutta_scheme import RungeKuttaScheme
from .time_stage_problem_computation_functors import (
    TimeStageProblemComputeFunctor,
    TimeStageProblemComputeJacvecFunctor,
    TimeStageProblemComputeTransposeJacvecFunctor,
)
from .postprocessing import Postprocessor
from .postprocessing_computation_functors import (
    PostprocessingProblemComputeFunctor,
    PostprocessingProblemComputeJacvecFunctor,
    PostprocessingProblemComputeTransposeJacvecFunctor,
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
    postprocessor: Postprocessor

    numpy_array_size: int
    numpy_functional_size: int

    cached_input: np.ndarray
    revolver: pr.BaseRevolver
    serialized_old_state_symbol: RungeKuttaIntegratorSymbol
    serialized_new_state_symbol: RungeKuttaIntegratorSymbol
    forward_operator: RungeKuttaForwardOperator
    reverse_operator: RungeKuttaReverseOperator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quantity_metadata = {}
        self.stage_cache = {}
        self.d_stage_cache = {}
        self.d_stage_cache_functional = {}
        self._time_cache = {}
        self.functional_quantities = []
        self.of_vars = []
        self.wrt_vars = []

        self.of_vars_postproc = []
        self.wrt_vars_postproc = []

        self.runge_kutta_scheme = None

        self.postprocessor = None

        self.numpy_array_size = 0
        self.numpy_postproc_size = 0
        self.numpy_functional_size = 0

        self.cached_input = None
        self.revolver = None
        self._revolver_class = None
        self.serialized_old_state_symbol: RungeKuttaIntegratorSymbol = None
        self.serialized_new_state_symbol: RungeKuttaIntegratorSymbol = None
        self.forward_operator: RungeKuttaForwardOperator = None
        self.reverse_operator: RungeKuttaReverseOperator = None

        self._disable_write_out = False

    def initialize(self):
        self.options.declare(
            "time_stage_problem",
            types=om.Problem,
            desc="""OpenMDAO problem used to model one Runge-Kutta stage""",
        )

        self.options.declare(
            "postprocessing_problem",
            types=(om.Problem, None),
            desc="""A problem used to calculate derived values based on the results of the time steps.""",
            default=None,
        )

        self.options.declare(
            "butcher_tableau",
            types=ButcherTableau,
            desc="The butcher tableau for the RK-scheme",
        )
        self.options.declare(
            "integration_control",
            types=IntegrationControl,
            desc="""Object used to exchange metadata between the inner and outer problems.""",
        )

        self.options.declare(
            "write_out_distance",
            types=int,
            default=0,
            desc="""If zero, no data is written out. Else, every ... time steps the data of the quantities are written 
            out to the write file.""",
        )

        self.options.declare(
            "write_file",
            types=str,
            default="data.h5",
            desc="The file where the results of each time steps are written.",
        )

        self.options.declare(
            "time_integration_quantities",
            types=list,
            desc="""List of tags that are used to find inputs and outputs in the time_stage_problem, and to find inputs
                in the post_processing_problem""",
        )

        self.options.declare(
            "post_processing_quantities",
            types=list,
            desc="""List of tags thar are used to find outputs in the postprocessing_problem""",
        )

        self.options.declare(
            "functional_coefficients",
            types=FunctionalCoefficients,
            default=EmptyFunctionalCoefficients(),
            desc="""A FunctionalCoefficients object that can return a list of quantities (both time_integration and 
                postprocessing ones) over which a functional is evaluated, as well as a coefficient given a time step 
                and a quantity""",
        )

        self.options.declare(
            "resets",
            types=(bool, list),
            default=False,
            desc="""If True, sets outputs of inner problem to zero before any call to run_solve_nonlinear. If a list, 
            then it only sets the variables in the list to zero. This can be useful due to the way the rtol option of
            nonlinear solvers in OpenMDAO works. After each time stage, the remaining values can cause the initial
            residual in the next stage to already be quite close to zero. While good in principle, this causes problems
            with the relative tolerance in OpenMDAO, since this compares to this initial residual.""",
        )

        self.options.declare(
            "revolver_type",
            values=["SingleLevel", "MultiLevel", "Memory", "Disk", "Base"],
            default="Memory",
            desc="""Chooses the type of revolver from pyrevolve. The default is the MemoryRevolver using numpy arrays as
            storage""",
        )

        self.options.declare(
            "revolver_options",
            types=dict,
            default={},
            desc="""Options that are given to the constructor of the revolver.""",
        )

    def _setup_revolver_class(self):
        revolver_type = self.options["revolver_type"]
        if revolver_type == "SingleLevel":
            self._revolver_class = pr.SingleLevelRevolver
        elif revolver_type == "MultiLevel":
            self._revolver_class = pr.MultiLevelRevolver
        elif revolver_type == "Disk":
            self._revolver_class = pr.DiskRevolver
        elif revolver_type == "Base":
            self._revolver_class = pr.BaseRevolver
        else:
            self._revolver_class = pr.MemoryRevolver

    def _check_num_checkpoints_in_revolver_options(self):
        if "n_checkpoints" not in self.options["revolver_options"].keys():
            self.options["revolver_options"]["n_checkpoints"] = (
                1 if self.options["integration_control"].num_steps == 1 else None
            )

    def _configure_write_out(self):
        self._disable_write_out = self.options["write_out_distance"] == 0

    def _setup_inner_problems(self):
        self.options["time_stage_problem"].setup()
        self.options["time_stage_problem"].final_setup()
        if self.options["postprocessing_problem"] is not None:
            self.options["postprocessing_problem"].setup()
            self.options["postprocessing_problem"].final_setup()

    def _setup_variables(self):
        self.functional_quantities = self.options[
            "functional_coefficients"
        ].list_quantities()

        self._setup_variable_information()

        self._setup_wrt_and_of_vars()

    def _setup_checkpointing(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        self._check_num_checkpoints_in_revolver_options()

        self.serialized_old_state_symbol = RungeKuttaIntegratorSymbol(
            self.numpy_array_size
        )
        self.serialized_new_state_symbol = RungeKuttaIntegratorSymbol(
            self.numpy_array_size
        )

        self.forward_operator = RungeKuttaForwardOperator(
            self.serialized_old_state_symbol,
            self.serialized_new_state_symbol,
            self.numpy_array_size,
            self.numpy_postproc_size,
            self.numpy_functional_size,
            butcher_tableau.number_of_stages(),
            self._run_step,
        )

        self.reverse_operator = RungeKuttaReverseOperator(
            self.serialized_old_state_symbol,
            self.numpy_array_size,
            self.numpy_postproc_size,
            butcher_tableau.number_of_stages(),
            self._run_step_jacvec_rev,
        )

    def setup(self):
        self._setup_inner_problems()

        self._setup_variables()

        self._setup_runge_kutta_scheme()

        self._setup_postprocessor()

        self._configure_write_out()

        self._setup_revolver_class()

        self._setup_checkpointing()

        # TODO: maybe add methods for time-independent in/outputs

    def _setup_postprocessor(self):
        postprocessing_problem: om.Problem = self.options["postprocessing_problem"]
        if postprocessing_problem is not None:
            self.postprocessor = Postprocessor(
                PostprocessingProblemComputeFunctor(
                    postprocessing_problem,
                    self.quantity_metadata,
                    self.numpy_array_size,
                    self.numpy_postproc_size,
                ),
                PostprocessingProblemComputeJacvecFunctor(
                    postprocessing_problem,
                    self.quantity_metadata,
                    self.numpy_array_size,
                    self.numpy_postproc_size,
                    self.of_vars_postproc,
                    self.wrt_vars_postproc,
                ),
                PostprocessingProblemComputeTransposeJacvecFunctor(
                    postprocessing_problem,
                    self.quantity_metadata,
                    self.numpy_array_size,
                    self.numpy_postproc_size,
                    self.of_vars_postproc,
                    self.wrt_vars_postproc,
                ),
            )

    def compute(self, inputs, outputs):  # pylint: disable = arguments-differ
        self._compute_preparation_phase(inputs)

        postprocessing_state = self._compute_postprocessing_phase()

        self._compute_initial_write_out_phase(postprocessing_state)

        functional_part = self._compute_functional_phase(postprocessing_state)

        self._compute_checkpointing_setup_phase(functional_part)

        self.revolver.apply_forward()

        self._compute_translate_to_om_vector_phase(outputs, functional_part)

    def _compute_preparation_phase(self, inputs: om_vector):
        self.to_numpy_array(inputs, self.serialized_new_state_symbol.data)
        self.cached_input = self.serialized_new_state_symbol.data.copy()
        self.options["integration_control"].reset()

    def _compute_postprocessing_phase(self):
        postprocessing_state = np.zeros(self.numpy_postproc_size)
        if self.options["postprocessing_problem"] is not None:
            postprocessing_state = self.postprocessor.postprocess(
                self.serialized_new_state_symbol.data
            )
        return postprocessing_state

    def _compute_initial_write_out_phase(self, postprocessing_state):
        initial_time = self.options["integration_control"].initial_time
        if not self._disable_write_out:
            self.write_out(
                0,
                initial_time,
                self.serialized_new_state_symbol.data,
                postprocessing_state,
                open_mode="w",
            )

    def _compute_functional_phase(self, postprocessing_state):
        functional_part = np.zeros(self.numpy_functional_size)
        if self.functional_quantities:
            functional_part += self.get_functional_contribution(
                self.serialized_new_state_symbol.data, postprocessing_state, 0
            )
        return functional_part

    def _compute_checkpointing_setup_phase(self, functional_part):
        num_steps = self.options["integration_control"].num_steps
        checkpoint_dict = {
            "serialized_old_state": self.serialized_old_state_symbol,
            "serialized_new_state": self.serialized_new_state_symbol,
        }

        checkpoint = RungeKuttaCheckpoint(checkpoint_dict)

        self.forward_operator.functional_part = functional_part

        self.revolver = self._revolver_class(
            checkpoint=checkpoint,
            fwd_operator=self.forward_operator,
            rev_operator=self.reverse_operator,
            n_timesteps=num_steps,
            **self.options["revolver_options"],
        )

    def _compute_translate_to_om_vector_phase(
        self, outputs: om_vector, functional_part: np.ndarray
    ):
        self.from_numpy_array(self.serialized_new_state_symbol.data, outputs)

        if self.options["postprocessing_problem"] is not None:
            self.from_numpy_array_postprocessing(
                self.forward_operator.postproc_state, outputs
            )

        if self.functional_quantities:
            self.add_functional_part_to_om_vec(functional_part, outputs)

    def _run_step(
        self, step, serialized_state, functional_part, stage_cache, accumulated_stages
    ):
        self._run_step_preparation_phase(step)

        serialized_state = self._run_step_time_integration_phase(
            serialized_state, stage_cache, accumulated_stages
        )

        postprocessing_state = self._run_step_postprocessing_phase(serialized_state)

        functional_part = self._run_step_functional_phase(
            serialized_state, postprocessing_state, functional_part
        )

        self._run_step_write_out_phase(serialized_state, postprocessing_state)

        return serialized_state, postprocessing_state, functional_part

    def _run_step_preparation_phase(self, step):
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        self.options["integration_control"].step = step
        time = initial_time + step * delta_t
        self.options["integration_control"].step_time_old = time
        self.options["integration_control"].step_time_new = time + delta_t

    def _run_step_time_integration_phase(
        self, serialized_state, stage_cache, accumulated_stages
    ):
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        time = self.options["integration_control"].step_time_old
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
        return serialized_state

    def _run_step_postprocessing_phase(self, serialized_state):
        postprocessing_state = np.zeros(self.numpy_postproc_size)
        if self.options["postprocessing_problem"] is not None:
            postprocessing_state = self.postprocessor.postprocess(serialized_state)
        return postprocessing_state

    def _run_step_functional_phase(
        self, serialized_state, postprocessing_state, functional_part
    ):
        step = self.options["integration_control"].step
        if self.functional_quantities:
            functional_part += self.get_functional_contribution(
                serialized_state, postprocessing_state, step
            )
        return functional_part

    def _run_step_write_out_phase(self, serialized_state, postprocessing_state):
        step = self.options["integration_control"].step
        time = self.options["integration_control"].step_time_old
        if not self._disable_write_out and (
            step % self.options["write_out_distance"] == 0
            or step == self.options["integration_control"].num_steps
        ):
            self.write_out(
                step, time, serialized_state, postprocessing_state, open_mode="r+"
            )

    def write_out(
        self,
        step: int,
        time: float,
        serialized_state: np.ndarray,
        postprocessing_state=None,
        open_mode="r+",
    ):
        with h5py.File(self.options["write_file"], mode=open_mode) as f:
            for quantity, metadata in self.quantity_metadata.items():
                if metadata["type"] == "time_integration":
                    start = metadata["numpy_start_index"]
                    end = metadata["numpy_end_index"]
                    dataset = f.create_dataset(
                        quantity + "/" + str(step),
                        data=serialized_state[start:end].reshape(metadata["shape"]),
                    )
                    dataset.attrs["time"] = time
                elif metadata["type"] == "postprocessing":
                    start = metadata["numpy_postproc_start_index"]
                    end = metadata["numpy_postproc_end_index"]
                    dataset = f.create_dataset(
                        quantity + "/" + str(step),
                        data=postprocessing_state[start:end].reshape(metadata["shape"]),
                    )

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode
    ):  # pylint: disable = arguments-differ
        if mode == "fwd":
            print("starting fwd_mode_jacvec")
            self._compute_jacvec_product_fwd(inputs, d_inputs, d_outputs)
            print("finished fwd_mode_jacvec")
        elif mode == "rev":
            print("starting rev_mode_jacvec")
            self._compute_jacvec_product_rev(inputs, d_inputs, d_outputs)
            print("finished rev_mode_jacvec")

    def _compute_jacvec_product_fwd(self, inputs, d_inputs, d_outputs):
        (
            serialized_state,
            serialized_state_perturbations,
        ) = self._compute_jacvec_fwd_preparation_phase(inputs, d_inputs)

        postprocessing_state_perturbations = (
            self._compute_jacvec_fwd_postprocessing_phase(
                serialized_state, serialized_state_perturbations
            )
        )

        functional_part_perturbations = self._compute_jacvec_fwd_functional_phase(
            serialized_state_perturbations, postprocessing_state_perturbations
        )

        (
            serialized_state_perturbations,
            postprocessing_state_perturbations,
            functional_part_perturbations,
        ) = self._compute_jacvec_fwd_run_steps(
            serialized_state,
            serialized_state_perturbations,
            functional_part_perturbations,
        )

        self._compute_jacvec_fwd_translate_to_om_vector_phase(
            d_outputs,
            serialized_state_perturbations,
            postprocessing_state_perturbations,
            functional_part_perturbations,
        )

    def _compute_jacvec_fwd_preparation_phase(self, inputs, d_inputs):
        serialized_state = np.zeros(self.numpy_array_size)
        self.to_numpy_array(inputs, serialized_state)
        serialized_state_perturbations = np.zeros(self.numpy_array_size)
        self.to_numpy_array(d_inputs, serialized_state_perturbations)
        self.options["integration_control"].reset()
        return serialized_state, serialized_state_perturbations

    def _compute_jacvec_fwd_postprocessing_phase(
        self, serialized_state, serialized_state_perturbations
    ):
        postprocessing_state_perturbations = np.zeros(self.numpy_postproc_size)
        if self.options["postprocessing_problem"] is not None:
            self.postprocessor.postprocess(serialized_state)
            postprocessing_state_perturbations = self.postprocessor.postprocess_jacvec(
                serialized_state_perturbations
            )
        return postprocessing_state_perturbations

    def _compute_jacvec_fwd_functional_phase(
        self, serialized_state_perturbations, postprocessing_state_perturbations
    ):
        functional_part_perturbations = np.zeros(self.numpy_functional_size)
        if self.functional_quantities:
            functional_part_perturbations += self.get_functional_contribution(
                serialized_state_perturbations, postprocessing_state_perturbations, 0
            )
        return functional_part_perturbations

    def _compute_jacvec_fwd_run_steps(
        self,
        serialized_state,
        serialized_state_perturbations,
        functional_part_perturbations,
    ):
        num_steps = self.options["integration_control"].num_steps

        postprocessing_state_perturbations = np.zeros(self.numpy_postproc_size)
        for step in range(1, num_steps + 1):
            self._compute_jacvec_fwd_run_steps_preparation_phase(step)
            (
                serialized_state,
                serialized_state_perturbations,
            ) = self._compute_jacvec_fwd_run_steps_time_integration_phase(
                serialized_state, serialized_state_perturbations
            )

            postprocessing_state_perturbations = (
                self._compute_jacvec_fwd_run_steps_postprocessing_phase(
                    serialized_state, serialized_state_perturbations
                )
            )

            functional_part_perturbations = (
                self._compute_jacvec_fwd_run_steps_functional_phase(
                    serialized_state_perturbations,
                    postprocessing_state_perturbations,
                    functional_part_perturbations,
                )
            )

        return (
            serialized_state_perturbations,
            postprocessing_state_perturbations,
            functional_part_perturbations,
        )

    def _compute_jacvec_fwd_run_steps_preparation_phase(self, step):
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        self.options["integration_control"].step = step
        time = initial_time + step * delta_t
        self.options["integration_control"].step_time_old = time
        self.options["integration_control"].step_time_new = time + delta_t

    def _compute_jacvec_fwd_run_steps_time_integration_phase(
        self, serialized_state, serialized_state_perturbations
    ):
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        time = self.options["integration_control"].step_time_old
        stage_cache = np.zeros(
            (butcher_tableau.number_of_stages(), serialized_state.size)
        )
        stage_perturbations_cache = np.zeros(
            (butcher_tableau.number_of_stages(), serialized_state.size)
        )
        accumulated_stages = np.zeros_like(serialized_state)
        accumulated_stage_perturbations = np.zeros_like(serialized_state)
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
            stage_perturbations_cache[
                stage, :
            ] = self.runge_kutta_scheme.compute_stage_jacvec(
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
        return serialized_state, serialized_state_perturbations

    def _compute_jacvec_fwd_run_steps_postprocessing_phase(
        self, serialized_state, serialized_state_perturbations
    ):
        postprocessing_state_perturbations = np.zeros(self.numpy_postproc_size)
        if self.options["postprocessing_problem"] is not None:
            self.postprocessor.postprocess(serialized_state)
            postprocessing_state_perturbations = self.postprocessor.postprocess_jacvec(
                serialized_state_perturbations
            )
        return postprocessing_state_perturbations

    def _compute_jacvec_fwd_run_steps_functional_phase(
        self,
        serialized_state_perturbations,
        postprocessing_state_perturbations,
        functional_part_perturbations,
    ):
        step = self.options["integration_control"].step
        if self.functional_quantities:
            functional_part_perturbations += self.get_functional_contribution(
                serialized_state_perturbations,
                postprocessing_state_perturbations,
                step,
            )
        return functional_part_perturbations

    def _compute_jacvec_fwd_translate_to_om_vector_phase(
        self,
        d_outputs: om_vector,
        serialized_state_perturbations: np.ndarray,
        postprocessing_state_perturbations: np.ndarray,
        functional_part_perturbations: np.ndarray,
    ):
        self.from_numpy_array(serialized_state_perturbations, d_outputs)

        if self.options["postprocessing_problem"] is not None:
            self.from_numpy_array_postprocessing(
                postprocessing_state_perturbations, d_outputs
            )

        if self.functional_quantities:
            self.add_functional_part_to_om_vec(functional_part_perturbations, d_outputs)

    def _compute_jacvec_product_rev(self, inputs, d_inputs, d_outputs):
        self._disable_write_out = True
        delta_t = self.options["integration_control"].delta_t
        serialized_state = np.zeros(self.numpy_array_size)
        self.to_numpy_array(inputs, serialized_state)
        if (
            not np.array_equal(self.cached_input, serialized_state)
            or self.revolver is None
        ):
            outputs = self._vector_class("nonlinear", "output", self)
            self.compute(inputs, outputs)

        serialized_state_perturbations = np.zeros(self.numpy_array_size)
        self.to_numpy_array(d_outputs, serialized_state_perturbations)

        postprocessing_state_perturbations = np.zeros(self.numpy_postproc_size)
        self.to_numpy_array_postprocessing(
            d_outputs, postprocessing_state_perturbations
        )

        if self.options["postprocessing_problem"] is not None:
            # due to the above check that we called compute with the current inputs
            # we know that the postprocessor is currently linearized at the right point
            serialized_state_perturbations += (
                self.postprocessor.postprocess_jacvec_transposed(
                    postprocessing_state_perturbations
                )
            )

        self.reverse_operator.serialized_state_perturbations = (
            serialized_state_perturbations
        )
        if self.functional_quantities:
            functional_time_integration_perturbations = np.zeros(self.numpy_array_size)
            self.functional_time_integration_contribution_from_om_output_vec(
                d_outputs, functional_time_integration_perturbations
            )
            self.reverse_operator.original_time_integration_functional_perturbations = (
                functional_time_integration_perturbations
            )
            functional_postprocessing_perturbations = np.zeros(self.numpy_postproc_size)
            if self.options["postprocessing_problem"] is not None:
                self.functional_postprocessing_contribution_from_om_output_vec(
                    d_outputs, functional_postprocessing_perturbations
                )
                self.reverse_operator.original_postprocessing_functional_perturbations = (
                    functional_postprocessing_perturbations
                )

            self.reverse_operator.functional_perturbations = (
                self.join_time_integration_and_postprocessing_functional(
                    self.options["integration_control"].num_steps,
                    functional_time_integration_perturbations,
                    functional_postprocessing_perturbations,
                )
            )

        self.revolver.apply_reverse()

        self.from_numpy_array(
            self.reverse_operator.serialized_state_perturbations, d_inputs
        )
        if self.functional_quantities:
            d_inputs_functional = self._vector_class("linear", "input", self)
            self.from_numpy_array(
                self.reverse_operator.functional_perturbations, d_inputs_functional
            )
            d_inputs.add_scal_vec(1.0, d_inputs_functional)

        self.revolver = None
        self._configure_write_out()

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
        original_time_integration_functional_perturbations,
        original_postprocessing_functional_perturbations,
    ):
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        self.options["integration_control"].step = step
        time = initial_time + step * delta_t
        self.options["integration_control"].step_time_old = time
        self.options["integration_control"].step_time_new = time + delta_t
        inputs_cache = {}
        outputs_cache = {}
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
            inputs_cache[stage] = self.options["time_stage_problem"].model._inputs
            outputs_cache[stage] = self.options["time_stage_problem"].model._outputs

        future_serialized_state = self.runge_kutta_scheme.compute_step(
            delta_t, serialized_state, stage_cache
        )
        postprocessing_state = np.zeros(self.numpy_postproc_size)
        if self.options["postprocessing_problem"] is not None:
            postprocessing_state = self.postprocessor.postprocess(
                future_serialized_state
            )

        new_serialized_state_perturbations = serialized_state_perturbations.copy()
        if self.functional_quantities:
            new_functional_perturbations = functional_perturbations.copy()
        for stage in range(butcher_tableau.number_of_stages() - 1, -1, -1):
            linearization_args = {
                "inputs": inputs_cache[stage],
                "outputs": outputs_cache[stage],
            }
            joined_perturbations = self.runge_kutta_scheme.join_new_state_and_accumulated_stages_perturbations(
                stage, serialized_state_perturbations, stage_perturbations_cache
            )
            (
                wrt_old_state,
                stage_perturbations_cache[stage, :],
            ) = self.runge_kutta_scheme.compute_stage_transposed_jacvec(
                stage, delta_t, time, joined_perturbations, **linearization_args
            )

            new_serialized_state_perturbations += delta_t * wrt_old_state
            if self.functional_quantities:
                functional_joined_perturbations = self.runge_kutta_scheme.join_new_state_and_accumulated_stages_perturbations(
                    stage,
                    functional_perturbations,
                    functional_stage_perturbations_cache,
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
            + self.join_time_integration_and_postprocessing_functional(
                step - 1,
                original_time_integration_functional_perturbations,
                original_postprocessing_functional_perturbations,
            )
            if self.functional_quantities
            else functional_perturbations,
        )

    def _setup_variable_information(self):
        time_stage_problem: om.Problem = self.options["time_stage_problem"]
        postprocessing_problem: om.Problem = self.options["postprocessing_problem"]
        self.numpy_array_size = 0
        self.numpy_postproc_size = 0
        self.numpy_functional_size = 0
        old_step_input_vars = time_stage_problem.model.get_io_metadata(
            iotypes="input",
            metadata_keys=["tags"],
            tags=["step_input_var"],
            get_remote=False,
        )

        acc_stage_input_vars = time_stage_problem.model.get_io_metadata(
            iotypes="input",
            metadata_keys=["tags"],
            tags=["accumulated_stage_var"],
            get_remote=False,
        )

        stage_output_vars = time_stage_problem.model.get_io_metadata(
            iotypes="output",
            metadata_keys=["tags"],
            tags=["stage_output_var"],
            get_remote=False,
        )
        if postprocessing_problem is not None:
            postproc_input_vars = postprocessing_problem.model.get_io_metadata(
                iotypes="input",
                metadata_keys=["tags"],
                tags=["postproc_input_var"],
                get_remote=False,
            )

            postproc_output_vars = postprocessing_problem.model.get_io_metadata(
                iotypes="output",
                mtadata_keys=["tags"],
                tags=["postproc_outputs_var"],
                get_remote=False,
            )

        for quantity in self.options["time_integration_quantities"]:
            self._extract_quantity_metadata_from_time_stage_problem(
                quantity, old_step_input_vars, acc_stage_input_vars, stage_output_vars
            )
            if postprocessing_problem is not None:
                self._extract_time_integration_quantity_metadata_from_postprocessing_problem(
                    quantity, postproc_input_vars
                )
            if not (
                (
                    (self.quantity_metadata[quantity]["step_input_var"] is not None)
                    and (
                        self.quantity_metadata[quantity]["accumulated_stage_var"]
                        is not None
                    )
                    and (
                        self.quantity_metadata[quantity]["stage_output_var"] is not None
                    )
                )
                or (
                    (self.quantity_metadata[quantity]["step_input_var"] is None)
                    and (
                        self.quantity_metadata[quantity]["accumulated_stage_var"]
                        is None
                    )
                    and self.quantity_metadata[quantity]["stage_output_var"] is not None
                )
            ):
                # TODO this probably should raise an error and not just be a message
                print(
                    f"""Warning! The time integration problem contains a nonworking combination of variables for
                    quantity {quantity}. Check that there is a stage_output var and either both or none of
                    step_input_var and accumulated_stage_var. {quantity} is ignored for this time integration."""
                )
                del self.quantity_metadata[quantity]
            else:
                self.quantity_metadata[quantity][
                    "numpy_start_index"
                ] = self.numpy_array_size
                self.numpy_array_size += np.prod(
                    self.quantity_metadata[quantity]["shape"]
                )
                self.quantity_metadata[quantity][
                    "numpy_end_index"
                ] = self.numpy_array_size
                if quantity in self.functional_quantities:
                    self.quantity_metadata[quantity]["integrated"] = True
                    self.quantity_metadata[quantity][
                        "numpy_functional_start_index"
                    ] = self.numpy_functional_size
                    self.numpy_functional_size += np.prod(
                        self.quantity_metadata[quantity]["shape"]
                    )
                    self.quantity_metadata[quantity][
                        "numpy_functional_end_index"
                    ] = self.numpy_functional_size
                self._add_time_integration_inputs_and_outputs(quantity)
        if postprocessing_problem is not None:
            for quantity in self.options["postprocessing_quantities"]:
                self._extract_postprocessing_quantity_metadata_from_postprocessing_problem(
                    quantity, postproc_output_vars
                )
                if quantity in self.functional_quantities:
                    self.quantity_metadata[quantity]["integrated"] = True
                    self.quantity_metadata[quantity][
                        "numpy_functional_start_index"
                    ] = self.numpy_functional_size
                    self.numpy_functional_size += np.prod(
                        self.quantity_metadata[quantity]["shape"]
                    )
                    self.quantity_metadata[quantity][
                        "numpy_functional_end_index"
                    ] = self.numpy_functional_size

                self._add_postprocessing_outputs(quantity)

    def _add_postprocessing_outputs(self, quantity):
        self.add_output(
            quantity + "_final",
            shape=self.quantity_metadata[quantity]["shape"],
            distributed=self.quantity_metadata[quantity]["shape"]
            != self.quantity_metadata[quantity]["global_shape"],
        )
        if self.quantity_metadata[quantity]["integrated"]:
            self.add_output(
                quantity + "_integrated",
                copy_shape=quantity + "_final",
                distributed=self.quantity_metadata[quantity]["shape"]
                != self.quantity_metadata[quantity]["global_shape"],
            )

    def _extract_postprocessing_quantity_metadata_from_postprocessing_problem(
        self, quantity, postproc_output_vars
    ):
        postprocessing_problem: om.Problem = self.options["postprocessing_problem"]
        quantity_outputs = postprocessing_problem.model.get_io_metadata(
            iotypes="output",
            metadata_keys=["tags", "shape", "global_shape"],
            tags=[quantity],
            get_remote=False,
        )

        for var, metadata in quantity_outputs.items():
            if var in postproc_output_vars:
                self.quantity_metadata[quantity]["integrated"] = False
                self.quantity_metadata[quantity]["postproc_output_var"] = var
                self.quantity_metadata[quantity]["type"] = "postprocessing"
                self.quantity_metadata[quantity]["shape"] = metadata["shape"]
                self.quantity_metadata[quantity]["global_shape"] = metadata[
                    "global_shape"
                ]
                self.quantity_metadata[
                    "numpy_postproc_start_index"
                ] = self.numpy_postproc_size
                self.numpy_postproc_size += np.prod(
                    self.quantity_metadata[quantity]["shape"]
                )
                self.quantity_metadata[
                    "numpy_postproc_end_index"
                ] = self.numpy_postproc_size

    def _extract_time_integration_quantity_metadata_from_postprocessing_problem(
        self, quantity, postproc_input_vars
    ):
        postprocessing_problem: om.Problem = self.options["postprocessing_problem"]
        quantity_inputs = postprocessing_problem.model.get_io_metadata(
            iotypes="input",
            metadata_keys=["tags", "shape", "global_shape"],
            tags=[quantity],
            get_remote=False,
        )

        for var, metadata in quantity_inputs.items():
            if var in postproc_input_vars:
                self.quantity_metadata[quantity]["postproc_input_var"] = var
                assert metadata["shape"] == self.quantity_metadata[quantity]["shape"]
                assert (
                    metadata["global_shape"]
                    == self.quantity_metadata[quantity]["global_shape"]
                )

    def _extract_quantity_metadata_from_time_stage_problem(
        self, quantity, old_step_input_vars, acc_stage_input_vars, stage_output_vars
    ):
        time_stage_problem: om.Problem = self.options["time_stage_problem"]

        quantity_inputs = time_stage_problem.model.get_io_metadata(
            iotypes="input",
            metadata_keys=["tags", "shape", "global_shape"],
            tags=[quantity],
            get_remote=False,
        )

        quantity_outputs = time_stage_problem.model.get_io_metadata(
            iotypes="output",
            metadata_keys=["tags", "shape", "global_shape"],
            tags=[quantity],
            get_remote=False,
        )

        self.quantity_metadata[quantity] = {
            "type": "time_integration",
            "integrated": False,
            "step_input_var": None,
            "accumulated_stage_var": None,
            "stage_output_var": None,
            "shape": None,
            "global_shape": None,
        }

        for var, metadata in quantity_inputs.items():
            if var in old_step_input_vars:
                self.quantity_metadata[quantity]["step_input_var"] = var
                if self.quantity_metadata[quantity]["shape"] is None:
                    self.quantity_metadata[quantity]["shape"] = metadata["shape"]
                else:
                    assert (
                        metadata["shape"] == self.quantity_metadata[quantity]["shape"]
                    )
                if self.quantity_metadata[quantity]["global_shape"] is None:
                    self.quantity_metadata[quantity]["global_shape"] = metadata[
                        "global_shape"
                    ]
                else:
                    assert (
                        metadata["global_shape"]
                        == self.quantity_metadata[quantity]["global_shape"]
                    )

            elif var in acc_stage_input_vars:
                self.quantity_metadata[quantity]["accumulated_stage_var"] = var
                if self.quantity_metadata[quantity]["shape"] is None:
                    self.quantity_metadata[quantity]["shape"] = metadata["shape"]
                else:
                    assert (
                        metadata["shape"] == self.quantity_metadata[quantity]["shape"]
                    )
                if self.quantity_metadata[quantity]["global_shape"] is None:
                    self.quantity_metadata[quantity]["global_shape"] = metadata[
                        "global_shape"
                    ]
                else:
                    assert (
                        metadata["global_shape"]
                        == self.quantity_metadata[quantity]["global_shape"]
                    )

        for var, metadata in quantity_outputs.items():
            if var in stage_output_vars:
                self.quantity_metadata[quantity]["stage_output_var"] = var
            if self.quantity_metadata[quantity]["shape"] is None:
                self.quantity_metadata[quantity]["shape"] = metadata["shape"]
            else:
                assert metadata["shape"] == self.quantity_metadata[quantity]["shape"]
            if self.quantity_metadata[quantity]["global_shape"] is None:
                self.quantity_metadata[quantity]["global_shape"] = metadata[
                    "global_shape"
                ]
            else:
                assert (
                    metadata["global_shape"]
                    == self.quantity_metadata[quantity]["global_shape"]
                )
            if (
                self.quantity_metadata[quantity]["shape"]
                != self.quantity_metadata[quantity]["global_shape"]
            ):
                sizes = time_stage_problem.model._var_sizes["output"][
                    :, time_stage_problem.model._var_allprocs_abs2idx[var]
                ]
                self.quantity_metadata[quantity][
                    "local_indices_start"
                ] = start = np.sum(sizes[: self.comm.rank])
                self.quantity_metadata[quantity]["local_indices_end"] = (
                    start + sizes[self.comm.rank]
                )
            else:
                self.quantity_metadata[quantity]["local_indices_start"] = 0
                self.quantity_metadata[quantity]["local_indices_end"] = np.prod(
                    self.quantity_metadata[quantity]["shape"]
                )

    def _add_time_integration_inputs_and_outputs(self, quantity):
        time_stage_problem: om.Problem = self.options["time_stage_problem"]
        self.add_input(
            quantity + "_initial",
            shape=self.quantity_metadata[quantity]["shape"],
            val=time_stage_problem.get_val(
                self.quantity_metadata[quantity]["step_input_var"]
            ),
            distributed=self.quantity_metadata[quantity]["shape"]
            != self.quantity_metadata[quantity]["global_shape"],
        )
        self.add_output(
            quantity + "_final",
            copy_shape=quantity + "_initial",
            distributed=self.quantity_metadata[quantity]["shape"]
            != self.quantity_metadata[quantity]["global_shape"],
        )
        if self.quantity_metadata[quantity]["integrated"]:
            self.add_output(
                quantity + "_integrated",
                copy_shape=quantity + "_initial",
                distributed=self.quantity_metadata[quantity]["shape"]
                != self.quantity_metadata[quantity]["global_shape"],
            )

    def _setup_wrt_and_of_vars(self):
        for quantity, metadata in self.quantity_metadata.items():
            if metadata["type"] == "time_integration":
                self.of_vars.append(metadata["stage_output_var"])
                self.wrt_vars.append(metadata["step_input_var"])
                self.wrt_vars.append(metadata["accumulated_stage_var"])
                if "postproc_input_var" in metadata:
                    self.wrt_vars_postproc.append("postproc_input_var")
            elif metadata["type"] == "postprocessing":
                self.of_vars_postproc.append(metadata["postproc_output_var"])

    # TODO: Make work with postprocessing
    def to_numpy_array(self, om_vector: om_vector, np_array: np.ndarray):
        if om_vector._kind == "input":
            name_suffix = "_initial"
        else:
            name_suffix = "_final"

        for quantity, metadata in self.quantity_metadata.items():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = metadata["numpy_end_index"]
                np_array[start:end] = om_vector[quantity + name_suffix].flatten()

    def to_numpy_array_postprocessing(
        self, om_vector: om_vector, np_postproc_array: np.ndarray
    ):
        name_suffix = "_final"
        for quantity, metadata in self.quantity_metadata.items():
            if metadata["type"] == "postprocessing":
                start = metadata["numpy_postproc_start_index"]
                end = metadata["numpy_postproc_end_index"]
                np_postproc_array[start:end] = om_vector[quantity + name_suffix].flatten

    def from_numpy_array(self, np_array: np.ndarray, om_vector: om_vector):
        if om_vector._kind == "input":
            name_suffix = "_initial"
        else:
            name_suffix = "_final"

        for quantity, metadata in self.quantity_metadata.items():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = metadata["numpy_end_index"]
                om_vector[quantity + name_suffix] = np_array[start:end].reshape(
                    metadata["shape"]
                )

    def from_numpy_array_postprocessing(
        self, np_postproc_array: np.ndarray, om_vector: om_vector
    ):
        name_suffix = "_final"  # postprocessing variables are only in output, not input
        for quantity, metadata in self.quantity_metadata.items():
            if metadata["type"] == "postprocessing":
                start = metadata["numpy_postproc_start_index"]
                end = metadata["numpy_postproc_end_index"]
                om_vector[quantity + name_suffix] = np_postproc_array[
                    start:end
                ].reshape(metadata["shape"])

    def _setup_runge_kutta_scheme(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        self.runge_kutta_scheme = RungeKuttaScheme(
            butcher_tableau,
            TimeStageProblemComputeFunctor(
                self.options["time_stage_problem"],
                self.options["integration_control"],
                self.quantity_metadata,
                self.options["resets"],
            ),
            TimeStageProblemComputeJacvecFunctor(
                self.options["time_stage_problem"],
                self.options["integration_control"],
                self.quantity_metadata,
                self.of_vars,
                self.wrt_vars,
            ),
            TimeStageProblemComputeTransposeJacvecFunctor(
                self.options["time_stage_problem"],
                self.options["integration_control"],
                self.quantity_metadata,
                self.of_vars,
                self.wrt_vars,
            ),
        )

    # TODO: change since integrated quantities is no more
    def get_functional_contribution(
        self,
        serialized_state: np.ndarray,
        postprocessing_state: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        functional_coefficients: FunctionalCoefficients = self.options[
            "functional_coefficients"
        ]
        contribution = np.zeros(self.numpy_functional_size)
        for quantity in self.functional_quantities:
            start_functional = self.quantity_metadata[quantity][
                "numpy_functional_start_index"
            ]
            end_functional = self.quantity_metadata[quantity][
                "numpy_functional_end_index"
            ]
            if self.quantity_metadata[quantity]["type"] == "time_integration":
                start_time_stepping = self.quantity_metadata[quantity][
                    "numpy_start_index"
                ]
                end_time_stepping = self.quantity_metadata[quantity]["numpy_end_index"]
                contribution[start_functional:end_functional] = (
                    functional_coefficients.get_coefficient(timestep, quantity)
                    * serialized_state[start_time_stepping:end_time_stepping]
                )
            elif self.quantity_metadata[quantity]["type"] == "postprocessing":
                start_postprocessing = self.quantity_metadata[quantity][
                    "numpy_postproc_start_index"
                ]
                end_postprocessing = self.quantity_metadata[quantity][
                    "numpy_postproc_end_index"
                ]
                contribution[start_functional:end_functional] = (
                    functional_coefficients.get_coefficient(timestep, quantity)
                    * postprocessing_state[start_postprocessing:end_postprocessing]
                )
        return contribution

    # TODO: change since integrated quantities is no more
    def functional_time_integration_contribution_from_om_output_vec(
        self, output_vec: om_vector, array: np.ndarray
    ):
        for quantity in self.functional_quantities:
            if self.quantity_metadata[quantity]["type"] == "time_integration":
                start = self.quantity_metadata[quantity]["numpy_start_index"]
                end = self.quantity_metadata[quantity]["numpy_end_index"]
                array[start:end] = output_vec[quantity + "_integrated"].flatten()

    def functional_postprocessing_contribution_from_om_output_vec(
        self, output_vec: om_vector, array: np.ndarray
    ):
        for quantity in self.functional_quantities:
            if self.quantity_metadata[quantity]["type"] == "postprocessing":
                start = self.quantity_metadata[quantity]["numpy_postproc_start_index"]
                end = self.quantity_metadata[quantity]["numpy_postproc_end_index"]
                array[start:end] = output_vec[quantity + "integrated"].flatten()

    def join_time_integration_and_postprocessing_functional(
        self, step: int, time_integration: np.ndarray, postprocessing: np.ndarray
    ) -> np.ndarray:
        functional_coefficents: FunctionalCoefficients = self.options[
            "functional_coefficients"
        ]
        result = np.zeros(self.numpy_array_size)
        postproc_copy = postprocessing.copy()
        for quantity in self.functional_quantities:
            if self.quantity_metadata[quantity]["type"] == "time_integration":
                start = self.quantity_metadata[quantity]["numpy_start_index"]
                end = self.quantity_metadata[quantity]["numpy_end_index"]
                result[start:end] += (
                    functional_coefficents.get_coefficient(step, quantity)
                    * time_integration[start:end]
                )
            elif self.quantity_metadata["type"] == "postprocessing":
                start = self.quantity_metadata[quantity]["numpy_postproc_start_index"]
                end = self.quantity_metadata[quantity]["numpy_postproc_end_index"]
                postproc_copy[start:end] *= functional_coefficents.get_coefficient(
                    step, quantity
                )
        result += (
            self.postprocessor.postprocessing_computation_functor_jacvec_transposed(
                postproc_copy
            )
        )
        return result

    # TODO: change since integrated quantities is no more
    def add_functional_part_to_om_vec(
        self, functional_numpy_array: np.ndarray, om_vector: om_vector
    ):
        for quantity in self.functional_numpy_array:
            start = self.quantity_metadata[quantity]["numpy_functional_start_index"]
            end = self.quantity_metadata[quantity]["numpy_functional_end_index"]
            om_vector[quantity + "_integrated"] += functional_numpy_array[
                start:end
            ].reshape(self.quantity_metadata[quantity]["shape"])
