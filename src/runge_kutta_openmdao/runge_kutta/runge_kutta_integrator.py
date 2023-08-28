import warnings

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
    Outer component for solving time-dependent problems with explicit or diagonally implicit Runge-Kutta schemes. One
    stage of the scheme is modelled by an inner openMDAO-problem. Optionally, time-step postprocessing and calculation
    of linear combinations of quantities can be done.
    OpenMDAO inputs: - initial values of the quantities for the time integration
    OpenMDAO output: - final values of the quantities for the time integration
                     - (optional) postprocessed final values
                     - (optional) linear combinations of quantities over time
    """

    _quantity_metadata: dict
    _d_stage_cache: dict
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
        self._quantity_metadata = {}
        self.functional_quantities = []
        self.of_vars = []
        self.wrt_vars = []
        self.of_vars_postproc = []
        self.wrt_vars_postproc = []

        self.runge_kutta_scheme = None
        self._serialized_state = np.zeros(0)
        self._accumulated_stages = np.zeros(0)
        self._stage_cache = np.zeros(0)
        self._serialized_state_perturbations = np.zeros(0)
        self._serialized_state_perturbations_from_functional = np.zeros(0)
        self._accumulated_stage_perturbations = np.zeros(0)
        self._stage_perturbations_cache = np.zeros(0)

        self.postprocessor = None
        self._postprocessing_state = np.zeros(0)
        self._postprocessing_state_perturbations = np.zeros(0)

        self._functional_part = np.zeros(0)
        self._functional_part_perturbations = np.zeros(0)

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
            "postprocessing_quantities",
            types=list,
            default=[],
            desc="""List of tags that are used to find outputs in the postprocessing_problem""",
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
            storage. 
            Warning: MultiLevelRevolver currently has problems where certain numbers of checkpoints work and others don't 
            (without an obvious reason why). Use with care.""",
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
            warnings.warn(
                """MultiLevelRevolver currently has problems where certain numbers of checkpoints work and others don't 
                (without an obvious reason why). Use with care."""
            )
        elif revolver_type == "Disk":
            self._revolver_class = pr.DiskRevolver
        elif revolver_type == "Base":
            self._revolver_class = pr.BaseRevolver
        else:
            self._revolver_class = pr.MemoryRevolver

    def _check_num_checkpoints_in_revolver_options(self):
        if "n_checkpoints" not in self.options[
            "revolver_options"
        ].keys() and self.options["revolver_type"] not in ["MultiLevel", "Base"]:
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

    def _setup_arrays(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        self._serialized_state.resize(self.numpy_array_size)
        self._accumulated_stages.resize(self.numpy_array_size)
        self._stage_cache.resize(
            (butcher_tableau.number_of_stages(), self.numpy_array_size)
        )
        self._serialized_state_perturbations.resize(self.numpy_array_size)
        self._serialized_state_perturbations_from_functional.resize(
            self.numpy_array_size
        )
        self._accumulated_stage_perturbations.resize(self.numpy_array_size)
        self._stage_perturbations_cache.resize(
            (butcher_tableau.number_of_stages(), self.numpy_array_size)
        )

        self._postprocessing_state.resize(self.numpy_postproc_size)
        self._postprocessing_state_perturbations.resize(self.numpy_postproc_size)

        self._functional_part.resize(self.numpy_functional_size)
        self._functional_part_perturbations.resize(self.numpy_functional_size)

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
            self._run_step_jacvec_rev,
        )

    def setup(self):
        self._setup_inner_problems()

        self._setup_variables()

        self._setup_arrays()

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
                    self._quantity_metadata,
                    self.numpy_array_size,
                    self.numpy_postproc_size,
                ),
                PostprocessingProblemComputeJacvecFunctor(
                    postprocessing_problem,
                    self._quantity_metadata,
                    self.numpy_array_size,
                    self.numpy_postproc_size,
                    self.of_vars_postproc,
                    self.wrt_vars_postproc,
                ),
                PostprocessingProblemComputeTransposeJacvecFunctor(
                    postprocessing_problem,
                    self._quantity_metadata,
                    self.numpy_array_size,
                    self.numpy_postproc_size,
                    self.of_vars_postproc,
                    self.wrt_vars_postproc,
                ),
            )

    def compute(self, inputs, outputs):  # pylint: disable = arguments-differ
        self._compute_preparation_phase(inputs)

        self._compute_postprocessing_phase()

        self._compute_initial_write_out_phase()

        self._compute_functional_phase()

        self._compute_checkpointing_setup_phase()

        self.revolver.apply_forward()

        self._compute_translate_to_om_vector_phase(outputs)

    def _compute_preparation_phase(self, inputs: om_vector):
        self.to_numpy_array(inputs, self.serialized_new_state_symbol.data)
        self.cached_input = self.serialized_new_state_symbol.data.copy()
        self.options["integration_control"].reset()

    def _compute_postprocessing_phase(self):
        if self.options["postprocessing_problem"] is not None:
            self._postprocessing_state = self.postprocessor.postprocess(
                self.serialized_new_state_symbol.data
            )

    def _compute_initial_write_out_phase(self):
        initial_time = self.options["integration_control"].initial_time
        if not self._disable_write_out:
            self.write_out(
                0,
                initial_time,
                self.serialized_new_state_symbol.data,
                self._postprocessing_state,
                open_mode="w",
            )

    def _compute_functional_phase(self):
        if self.functional_quantities:
            self._functional_part = self.get_functional_contribution(
                self.serialized_new_state_symbol.data, self._postprocessing_state, 0
            )

    def _compute_checkpointing_setup_phase(self):
        num_steps = self.options["integration_control"].num_steps
        checkpoint_dict = {
            "serialized_old_state": self.serialized_old_state_symbol,
            "serialized_new_state": self.serialized_new_state_symbol,
        }

        checkpoint = RungeKuttaCheckpoint(checkpoint_dict)

        revolver_options = {}

        for key, value in self.options["revolver_options"].items():
            if self.options["revolver_type"] == "MultiLevel" and key == "storage_list":
                storage_list = []
                for storage_type, options in value.items():
                    if storage_type == "Numpy":
                        storage_list.append(pr.NumpyStorage(checkpoint.size, **options))
                    elif storage_type == "Disk":
                        storage_list.append(pr.DiskStorage(checkpoint.size, **options))
                    elif storage_type == "Bytes":
                        storage_list.append(pr.BytesStorage(checkpoint.size, **options))
                revolver_options[key] = storage_list
            else:
                revolver_options[key] = value
        revolver_options["checkpoint"] = checkpoint
        revolver_options["fwd_operator"] = self.forward_operator
        revolver_options["rev_operator"] = self.reverse_operator
        revolver_options["n_timesteps"] = num_steps

        self.revolver = self._revolver_class(
            **revolver_options,
        )

    def _compute_translate_to_om_vector_phase(self, outputs: om_vector):
        self.from_numpy_array(self.serialized_new_state_symbol.data, outputs)

        if self.options["postprocessing_problem"] is not None:
            self.from_numpy_array_postprocessing(self._postprocessing_state, outputs)

        if self.functional_quantities:
            self.add_functional_part_to_om_vec(self._functional_part, outputs)

    def _run_step(self, step, serialized_state):
        self._run_step_preparation_phase(step, serialized_state)

        self._run_step_time_integration_phase()

        self._run_step_postprocessing_phase()

        self._run_step_functional_phase()

        self._run_step_write_out_phase()

        return self._serialized_state

    def _run_step_preparation_phase(self, step, serialized_state):
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        self.options["integration_control"].step = step
        time = initial_time + step * delta_t
        self.options["integration_control"].step_time_old = time
        self.options["integration_control"].step_time_new = time + delta_t
        self._serialized_state = serialized_state

    def _run_step_time_integration_phase(self):
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        time = self.options["integration_control"].step_time_old
        for stage in range(butcher_tableau.number_of_stages()):
            self.options["integration_control"].stage = stage
            if stage != 0:
                self._accumulated_stages = (
                    self.runge_kutta_scheme.compute_accumulated_stages(
                        stage, self._stage_cache
                    )
                )
            else:
                self._accumulated_stages.fill(0.0)
            self._stage_cache[stage, :] = self.runge_kutta_scheme.compute_stage(
                stage, delta_t, time, self._serialized_state, self._accumulated_stages
            )
        self._serialized_state = self.runge_kutta_scheme.compute_step(
            delta_t, self._serialized_state, self._stage_cache
        )

    def _run_step_postprocessing_phase(self):
        if self.options["postprocessing_problem"] is not None:
            self._postprocessing_state = self.postprocessor.postprocess(
                self._serialized_state
            )

    def _run_step_functional_phase(self):
        step = self.options["integration_control"].step
        if self.functional_quantities:
            self._functional_part += self.get_functional_contribution(
                self._serialized_state, self._postprocessing_state, step
            )

    def _run_step_write_out_phase(self):
        step = self.options["integration_control"].step
        time = self.options["integration_control"].step_time_old
        if not self._disable_write_out and (
            step % self.options["write_out_distance"] == 0
            or step == self.options["integration_control"].num_steps
        ):
            self.write_out(
                step,
                time,
                self._serialized_state,
                self._postprocessing_state,
                open_mode="r+",
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
            for quantity, metadata in self._quantity_metadata.items():
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
        self._compute_jacvec_fwd_preparation_phase(inputs, d_inputs)

        self._compute_jacvec_fwd_postprocessing_phase()

        self._compute_jacvec_fwd_functional_phase()

        self._compute_jacvec_fwd_run_steps()

        self._compute_jacvec_fwd_translate_to_om_vector_phase(d_outputs)

    def _compute_jacvec_fwd_preparation_phase(self, inputs, d_inputs):
        self.to_numpy_array(inputs, self._serialized_state)
        self.to_numpy_array(d_inputs, self._serialized_state_perturbations)
        self.options["integration_control"].reset()

    def _compute_jacvec_fwd_postprocessing_phase(self):
        if self.options["postprocessing_problem"] is not None:
            self.postprocessor.postprocess(self._serialized_state)
            self._postprocessing_state_perturbations = (
                self.postprocessor.postprocess_jacvec(
                    self._serialized_state_perturbations
                )
            )

    def _compute_jacvec_fwd_functional_phase(self):
        if self.functional_quantities:
            self._functional_part_perturbations = self.get_functional_contribution(
                self._serialized_state_perturbations,
                self._postprocessing_state_perturbations,
                0,
            )

    def _compute_jacvec_fwd_run_steps(self):
        num_steps = self.options["integration_control"].num_steps
        for step in range(1, num_steps + 1):
            self._compute_jacvec_fwd_run_steps_preparation_phase(step)
            self._compute_jacvec_fwd_run_steps_time_integration_phase()

            self._compute_jacvec_fwd_run_steps_postprocessing_phase()

            self._compute_jacvec_fwd_run_steps_functional_phase()

    def _compute_jacvec_fwd_run_steps_preparation_phase(self, step):
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        self.options["integration_control"].step = step
        time = initial_time + step * delta_t
        self.options["integration_control"].step_time_old = time
        self.options["integration_control"].step_time_new = time + delta_t

    def _compute_jacvec_fwd_run_steps_time_integration_phase(self):
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        time = self.options["integration_control"].step_time_old
        for stage in range(butcher_tableau.number_of_stages()):
            if stage != 0:
                self._accumulated_stages = (
                    self.runge_kutta_scheme.compute_accumulated_stages(
                        stage, self._stage_cache
                    )
                )
                self._accumulated_stage_perturbations = (
                    self.runge_kutta_scheme.compute_accumulated_stage_perturbations(
                        stage, self._stage_perturbations_cache
                    )
                )
            else:
                self._accumulated_stages.fill(0.0)
                self._accumulated_stage_perturbations.fill(0.0)
            self._stage_cache[stage, :] = self.runge_kutta_scheme.compute_stage(
                stage, delta_t, time, self._serialized_state, self._accumulated_stages
            )
            self._stage_perturbations_cache[
                stage, :
            ] = self.runge_kutta_scheme.compute_stage_jacvec(
                stage,
                delta_t,
                time,
                self._serialized_state_perturbations,
                self._accumulated_stage_perturbations,
            )
        self._serialized_state = self.runge_kutta_scheme.compute_step(
            delta_t, self._serialized_state, self._stage_cache
        )

        self._serialized_state_perturbations = (
            self.runge_kutta_scheme.compute_step_jacvec(
                delta_t,
                self._serialized_state_perturbations,
                self._stage_perturbations_cache,
            )
        )

    def _compute_jacvec_fwd_run_steps_postprocessing_phase(self):
        if self.options["postprocessing_problem"] is not None:
            self.postprocessor.postprocess(self._serialized_state)
            self._postprocessing_state_perturbations = (
                self.postprocessor.postprocess_jacvec(
                    self._serialized_state_perturbations
                )
            )

    def _compute_jacvec_fwd_run_steps_functional_phase(self):
        step = self.options["integration_control"].step
        if self.functional_quantities:
            self._functional_part_perturbations += self.get_functional_contribution(
                self._serialized_state_perturbations,
                self._postprocessing_state_perturbations,
                step,
            )

    def _compute_jacvec_fwd_translate_to_om_vector_phase(
        self,
        d_outputs: om_vector,
    ):
        self.from_numpy_array(self._serialized_state_perturbations, d_outputs)

        if self.options["postprocessing_problem"] is not None:
            self.from_numpy_array_postprocessing(
                self._postprocessing_state_perturbations, d_outputs
            )

        if self.functional_quantities:
            self.add_functional_part_to_om_vec(
                self._functional_part_perturbations, d_outputs
            )

    def _compute_jacvec_product_rev(self, inputs, d_inputs, d_outputs):
        self._disable_write_out = True
        self.to_numpy_array(inputs, self._serialized_state)
        if (
            not np.array_equal(self.cached_input, self._serialized_state)
            or self.revolver is None
        ):
            outputs = self._vector_class("nonlinear", "output", self)
            self.compute(inputs, outputs)

        self.to_numpy_array(d_outputs, self._serialized_state_perturbations)
        self.to_numpy_array_postprocessing(
            d_outputs, self._postprocessing_state_perturbations
        )
        if self.options["postprocessing_problem"] is not None:
            self._serialized_state_perturbations += (
                self.postprocessor.postprocess_jacvec_transposed(
                    self._postprocessing_state_perturbations
                )
            )

        if self.functional_quantities:
            self._get_functional_contribution_from_om_output_vec(d_outputs)
            self._add_functional_perturbations_to_state_perturbations(
                self.options["integration_control"].num_steps
            )

        self.reverse_operator.serialized_state_perturbations = (
            self._serialized_state_perturbations
        )

        self.revolver.apply_reverse()

        self.from_numpy_array(
            self.reverse_operator.serialized_state_perturbations, d_inputs
        )

        self.revolver = None
        self._configure_write_out()

    def _run_step_jacvec_rev(
        self, step, serialized_state, serialized_state_perturbations
    ):
        self._serialized_state = serialized_state
        self._serialized_state_perturbations = serialized_state_perturbations
        new_serialized_state_perturbations = serialized_state_perturbations.copy()
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        self.options["integration_control"].step = step
        time = initial_time + step * delta_t
        self.options["integration_control"].step_time_old = time
        self.options["integration_control"].step_time_new = time + delta_t

        inputs_cache = {}
        outputs_cache = {}
        # forward iteration
        for stage in range(butcher_tableau.number_of_stages()):
            self.options["integration_control"].stage = stage
            if stage != 0:
                self._accumulated_stages = (
                    self.runge_kutta_scheme.compute_accumulated_stages(
                        stage, self._stage_cache
                    )
                )
            else:
                self._accumulated_stages.fill(0.0)
            self._stage_cache[stage, :] = self.runge_kutta_scheme.compute_stage(
                stage, delta_t, time, self._serialized_state, self._accumulated_stages
            )
            inputs_cache[stage] = self.options[
                "time_stage_problem"
            ].model._vector_class(
                "nonlinear", "input", self.options["time_stage_problem"].model
            )
            inputs_cache[stage].set_vec(
                self.options["time_stage_problem"].model._inputs
            )
            outputs_cache[stage] = self.options[
                "time_stage_problem"
            ].model._vector_class(
                "nonlinear", "output", self.options["time_stage_problem"].model
            )
            outputs_cache[stage].set_vec(
                self.options["time_stage_problem"].model._outputs
            )
        # backward iteration

        for stage in range(butcher_tableau.number_of_stages() - 1, -1, -1):
            linearization_args = {
                "inputs": inputs_cache[stage],
                "outputs": outputs_cache[stage],
            }
            joined_perturbations = self.runge_kutta_scheme.join_new_state_and_accumulated_stages_perturbations(
                stage,
                self._serialized_state_perturbations,
                self._stage_perturbations_cache,
            )
            (
                wrt_old_state,
                self._stage_perturbations_cache[stage, :],
            ) = self.runge_kutta_scheme.compute_stage_transposed_jacvec(
                stage, delta_t, time, joined_perturbations, **linearization_args
            )
            new_serialized_state_perturbations += delta_t * wrt_old_state

        self._serialized_state_perturbations = new_serialized_state_perturbations
        if self.options["postprocessing_problem"] is not None:
            self.postprocessor.postprocess(self._serialized_state)

        self._add_functional_perturbations_to_state_perturbations(step - 1)

        return self._serialized_state_perturbations

    def _add_functional_perturbations_to_state_perturbations(
        self, step, postprocessor_linearization_args={}
    ):
        functional_coefficients: FunctionalCoefficients = self.options[
            "functional_coefficients"
        ]
        postprocessing_functional_perturbations = np.zeros(self.numpy_postproc_size)
        for quantity in self.functional_quantities:
            start_functional = self._quantity_metadata[quantity][
                "numpy_functional_start_index"
            ]
            end_functional = self._quantity_metadata[quantity][
                "numpy_functional_end_index"
            ]
            if self._quantity_metadata[quantity]["type"] == "time_integration":
                start = self._quantity_metadata[quantity]["numpy_start_index"]
                end = self._quantity_metadata[quantity]["numpy_end_index"]
                self._serialized_state_perturbations[start:end] += (
                    functional_coefficients.get_coefficient(step, quantity)
                ) * self._functional_part_perturbations[start_functional:end_functional]
            elif self._quantity_metadata[quantity]["type"] == "postprocessing":
                start = self._quantity_metadata[quantity]["numpy_postproc_start_index"]
                end = self._quantity_metadata[quantity]["numpy_postproc_end_index"]
                postprocessing_functional_perturbations[start:end] += (
                    functional_coefficients.get_coefficient(step, quantity)
                    * self._functional_part_perturbations[
                        start_functional:end_functional
                    ]
                )
        if self.options["postprocessing_problem"]:
            self._serialized_state_perturbations += (
                self.postprocessor.postprocess_jacvec_transposed(
                    postprocessing_functional_perturbations,
                    **postprocessor_linearization_args,
                )
            )

    def _get_functional_contribution_from_om_output_vec(self, d_outputs: om_vector):
        for quantity in self.functional_quantities:
            start = self._quantity_metadata[quantity]["numpy_functional_start_index"]
            end = self._quantity_metadata[quantity]["numpy_functional_end_index"]
            self._functional_part_perturbations[start:end] = d_outputs[
                quantity + "_functional"
            ]

    # def _compute_jacvec_rev_preparation_and_check_revolver_integrity(
    #     self, inputs: om_vector, d_outputs: om_vector
    # ):
    #     self.to_numpy_array(inputs, self._serialized_state)
    #     if (
    #         not np.array_equal(self.cached_input, self._serialized_state)
    #         or self.revolver is None
    #     ):
    #         outputs = self._vector_class("nonlinear", "output", self)
    #         self.compute(inputs, outputs)
    #     self.to_numpy_array(d_outputs, self._serialized_state_perturbations)
    #
    # def _compute_jacvec_rev_postprocessing_phase(self, d_outputs: om_vector):
    #     self.to_numpy_array_postprocessing(
    #         d_outputs, self._postprocessing_state_perturbations
    #     )
    #
    #     if self.options["postprocessing_problem"] is not None:
    #         # due to the above check that we called compute with the current inputs
    #         # we know that the postprocessor is currently linearized at the right point
    #         self._serialized_state_perturbations += (
    #             self.postprocessor.postprocess_jacvec_transposed(
    #                 self._postprocessing_state_perturbations
    #             )
    #         )
    #
    # def _compute_jacvec_rev_functional_phase(self, d_outputs: om_vector):
    #     if self.functional_quantities:
    #         functional_time_integration_perturbations = np.zeros(self.numpy_array_size)
    #         self.functional_time_integration_contribution_from_om_output_vec(
    #             d_outputs, functional_time_integration_perturbations
    #         )
    #         self.reverse_operator.original_time_integration_functional_perturbations = (
    #             functional_time_integration_perturbations
    #         )
    #         functional_postprocessing_perturbations = np.zeros(self.numpy_postproc_size)
    #         if self.options["postprocessing_problem"] is not None:
    #             self.functional_postprocessing_contribution_from_om_output_vec(
    #                 d_outputs, functional_postprocessing_perturbations
    #             )
    #             self.reverse_operator.original_postprocessing_functional_perturbations = (
    #                 functional_postprocessing_perturbations
    #             )
    #
    #         self.reverse_operator.functional_perturbations = (
    #             self.join_time_integration_and_postprocessing_functional(
    #                 self.options["integration_control"].num_steps,
    #                 functional_time_integration_perturbations,
    #                 functional_postprocessing_perturbations,
    #             )
    #         )
    #
    # def _compute_jacvec_rev_translate_to_om_vector_phase(self, d_inputs: om_vector):
    #     self.from_numpy_array(
    #         self.reverse_operator.serialized_state_perturbations, d_inputs
    #     )
    #     if self.functional_quantities:
    #         d_inputs_functional = self._vector_class("linear", "input", self)
    #         self.from_numpy_array(
    #             self.reverse_operator.functional_perturbations, d_inputs_functional
    #         )
    #         d_inputs.add_scal_vec(1.0, d_inputs_functional)
    #
    # def _compute_jacvec_product_rev(self, inputs, d_inputs, d_outputs):
    #     self._disable_write_out = True
    #
    #     self._compute_jacvec_rev_preparation_and_check_revolver_integrity(
    #         inputs, d_outputs
    #     )
    #
    #     self._compute_jacvec_rev_postprocessing_phase(
    #         serialized_state_perturbations, d_outputs
    #     )
    #
    #     self._compute_jacvec_rev_functional_phase(d_outputs)
    #
    #     self.revolver.apply_reverse()
    #
    #     self._compute_jacvec_rev_translate_to_om_vector_phase(d_inputs)
    #
    #     self.revolver = None
    #     self._configure_write_out()
    #
    # def _run_step_jacvec_rev_preparation_phase(self, step):
    #     initial_time = self.options["integration_control"].initial_time
    #     delta_t = self.options["integration_control"].delta_t
    #     self.options["integration_control"].step = step
    #     time = initial_time + step * delta_t
    #     self.options["integration_control"].step_time_old = time
    #     self.options["integration_control"].step_time_new = time + delta_t
    #
    # def _run_step_jacvec_rev_time_integration_forward_phase(self):
    #     delta_t = self.options["integration_control"].delta_t
    #     butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
    #     time = self.options["integration_control"].step_time_old
    #     inputs_cache = {}
    #     outputs_cache = {}
    #     for stage in range(butcher_tableau.number_of_stages()):
    #         self.options["integration_control"].stage = stage
    #         if stage != 0:
    #             self._accumulated_stages = (
    #                 self.runge_kutta_scheme.compute_accumulated_stages(
    #                     stage, self._stage_cache
    #                 )
    #             )
    #         else:
    #             self._accumulated_stages.fill(0.0)
    #         self._stage_cache[stage, :] = self.runge_kutta_scheme.compute_stage(
    #             stage, delta_t, time, self._serialized_state, self._accumulated_stages
    #         )
    #         inputs_cache[stage] = self.options["time_stage_problem"].model._inputs
    #         outputs_cache[stage] = self.options["time_stage_problem"].model._outputs
    #         # inputs_cache[stage] = self.options[
    #         #     "time_stage_problem"
    #         # ].model._vector_class(
    #         #     "nonlinear", "input", self.options["time_stage_problem"].model
    #         # )
    #         # inputs_cache[stage].add_scal_vec(
    #         #     1.0, self.options["time_stage_problem"].model._inputs
    #         # )
    #         # outputs_cache[stage] = self.options[
    #         #     "time_stage_problem"
    #         # ].model._vector_class(
    #         #     "nonlinear", "output", self.options["time_stage_problem"].model
    #         # )
    #         # outputs_cache[stage].add_scal_vec(
    #         #     1.0, self.options["time_stage_problem"].model._outputs
    #         # )
    #
    #     future_serialized_state = self.runge_kutta_scheme.compute_step(
    #         delta_t, self._serialized_state, self._stage_cache
    #     )
    #     return future_serialized_state, inputs_cache, outputs_cache
    #
    # def _run_step_jacvec_rev_postprocessing_forward_phase(
    #     self, future_serialized_state
    # ):
    #     if self.options["postprocessing_problem"] is not None:
    #         self.postprocessor.postprocess(future_serialized_state)
    #
    # def _run_step_jacvec_rev_backward_phase(self, inputs_cache, outputs_cache):
    #     delta_t = self.options["integration_control"].delta_t
    #     butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
    #     time = self.options["integration_control"].step_time_old
    #     new_serialized_state_perturbations = self._serialized_state_perturbations.copy()
    #     for stage in range(butcher_tableau.number_of_stages() - 1, -1, -1):
    #         linearization_args = {
    #             "inputs": inputs_cache[stage],
    #             "outputs": outputs_cache[stage],
    #         }
    #         joined_perturbations = self.runge_kutta_scheme.join_new_state_and_accumulated_stages_perturbations(
    #             stage,
    #             self._serialized_state_perturbations,
    #             self._stage_perturbations_cache,
    #         )
    #         (
    #             wrt_old_state,
    #             self._stage_perturbations_cache[stage, :],
    #         ) = self.runge_kutta_scheme.compute_stage_transposed_jacvec(
    #             stage, delta_t, time, joined_perturbations, **linearization_args
    #         )
    #
    #         new_serialized_state_perturbations += delta_t * wrt_old_state
    #
    # def _run_step_jacvec_rev_backward_stage_time_integration_phase(
    #     self,
    #     linearization_args,
    # ):
    #     delta_t = self.options["integration_control"].delta_t
    #     time = self.options["integration_control"].step_time_old
    #     stage = self.options["integration_control"].stage
    #     joined_perturbations = (
    #         self.runge_kutta_scheme.join_new_state_and_accumulated_stages_perturbations(
    #             stage, self._serialized_state_perturbations, self._stage_perturbations_cache
    #         )
    #     )
    #     (
    #         wrt_old_state,
    #         self._stage_perturbations_cache[stage, :],
    #     ) = self.runge_kutta_scheme.compute_stage_transposed_jacvec(
    #         stage, delta_t, time, joined_perturbations, **linearization_args
    #     )
    #
    #     return delta_t * wrt_old_state
    #
    # def _run_step_jacvec_rev(
    #     self,
    #     step,
    #     serialized_state,
    #     serialized_state_perturbations,
    # ):
    #     self._serialized_state = serialized_state
    #     self._serialized_state_perturbations = serialized_state_perturbations
    #     # initial_time = self.options["integration_control"].initial_time
    #     # delta_t = self.options["integration_control"].delta_t
    #     butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
    #     # self.options["integration_control"].step = step
    #     # time = initial_time + step * delta_t
    #     # self.options["integration_control"].step_time_old = time
    #     # self.options["integration_control"].step_time_new = time + delta_t
    #     self._run_step_jacvec_rev_preparation_phase(step)
    #
    #     (
    #         future_serialized_state,
    #         inputs_cache,
    #         outputs_cache,
    #     ) = self._run_step_jacvec_rev_time_integration_forward_phase()
    #     self._run_step_jacvec_rev_postprocessing_forward_phase(future_serialized_state)
    #
    #     # new_serialized_state_perturbations = serialized_state_perturbations.copy()
    #     # if self.functional_quantities:
    #     #     new_functional_perturbations = functional_perturbations.copy()
    #     # for stage in range(butcher_tableau.number_of_stages() - 1, -1, -1):
    #     #     linearization_args = {
    #     #         "inputs": inputs_cache[stage],
    #     #         "outputs": outputs_cache[stage],
    #     #     }
    #     #     joined_perturbations = self.runge_kutta_scheme.join_new_state_and_accumulated_stages_perturbations(
    #     #         stage, serialized_state_perturbations, stage_perturbations_cache
    #     #     )
    #     #     (
    #     #         wrt_old_state,
    #     #         stage_perturbations_cache[stage, :],
    #     #     ) = self.runge_kutta_scheme.compute_stage_transposed_jacvec(
    #     #         stage, delta_t, time, joined_perturbations, **linearization_args
    #     #     )
    #     #
    #     #     new_serialized_state_perturbations += delta_t * wrt_old_state
    #     #     if self.functional_quantities:
    #     #         functional_joined_perturbations = self.runge_kutta_scheme.join_new_state_and_accumulated_stages_perturbations(
    #     #             stage,
    #     #             functional_perturbations,
    #     #             functional_stage_perturbations_cache,
    #     #         )
    #     #         (
    #     #             functional_wrt_old_state,
    #     #             functional_stage_perturbations_cache[stage, :],
    #     #         ) = self.runge_kutta_scheme.compute_stage_transposed_jacvec(
    #     #             stage,
    #     #             delta_t,
    #     #             time,
    #     #             functional_joined_perturbations,
    #     #         )
    #     #         new_functional_perturbations += delta_t * functional_wrt_old_state
    #
    #     self._run_step_jacvec_rev_backward_phase(inputs_cache, outputs_cache)
    #     return self._serialized_state_perturbations

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
                metadata_keys=["tags"],
                tags=["postproc_output_var"],
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
                    (self._quantity_metadata[quantity]["step_input_var"] is not None)
                    and (
                        self._quantity_metadata[quantity]["accumulated_stage_var"]
                        is not None
                    )
                    and (
                        self._quantity_metadata[quantity]["stage_output_var"]
                        is not None
                    )
                )
                or (
                    (self._quantity_metadata[quantity]["step_input_var"] is None)
                    and (
                        self._quantity_metadata[quantity]["accumulated_stage_var"]
                        is None
                    )
                    and self._quantity_metadata[quantity]["stage_output_var"]
                    is not None
                )
            ):
                # TODO this probably should raise an error and not just be a message
                print(
                    f"""Warning! The time integration problem contains a nonworking combination of variables for
                    quantity {quantity}. Check that there is a stage_output var and either both or none of
                    step_input_var and accumulated_stage_var. {quantity} is ignored for this time integration."""
                )
                del self._quantity_metadata[quantity]
            else:
                self._quantity_metadata[quantity][
                    "numpy_start_index"
                ] = self.numpy_array_size
                self.numpy_array_size += np.prod(
                    self._quantity_metadata[quantity]["shape"]
                )
                self._quantity_metadata[quantity][
                    "numpy_end_index"
                ] = self.numpy_array_size
                if quantity in self.functional_quantities:
                    self._quantity_metadata[quantity]["integrated"] = True
                    self._quantity_metadata[quantity][
                        "numpy_functional_start_index"
                    ] = self.numpy_functional_size
                    self.numpy_functional_size += np.prod(
                        self._quantity_metadata[quantity]["shape"]
                    )
                    self._quantity_metadata[quantity][
                        "numpy_functional_end_index"
                    ] = self.numpy_functional_size
                self._add_time_integration_inputs_and_outputs(quantity)
        if postprocessing_problem is not None:
            for quantity in self.options["postprocessing_quantities"]:
                self._extract_postprocessing_quantity_metadata_from_postprocessing_problem(
                    quantity, postproc_output_vars
                )
                if quantity in self.functional_quantities:
                    self._quantity_metadata[quantity]["integrated"] = True
                    self._quantity_metadata[quantity][
                        "numpy_functional_start_index"
                    ] = self.numpy_functional_size
                    self.numpy_functional_size += np.prod(
                        self._quantity_metadata[quantity]["shape"]
                    )
                    self._quantity_metadata[quantity][
                        "numpy_functional_end_index"
                    ] = self.numpy_functional_size
                self._add_postprocessing_outputs(quantity)

    def _add_postprocessing_outputs(self, quantity):
        self.add_output(
            quantity + "_final",
            shape=self._quantity_metadata[quantity]["shape"],
            distributed=self._quantity_metadata[quantity]["shape"]
            != self._quantity_metadata[quantity]["global_shape"],
        )
        if self._quantity_metadata[quantity]["integrated"]:
            self.add_output(
                quantity + "_functional",
                copy_shape=quantity + "_final",
                distributed=self._quantity_metadata[quantity]["shape"]
                != self._quantity_metadata[quantity]["global_shape"],
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
                self._quantity_metadata[quantity] = {}
                self._quantity_metadata[quantity]["integrated"] = False
                self._quantity_metadata[quantity]["postproc_output_var"] = var
                self._quantity_metadata[quantity]["type"] = "postprocessing"
                self._quantity_metadata[quantity]["shape"] = metadata["shape"]
                self._quantity_metadata[quantity]["global_shape"] = metadata[
                    "global_shape"
                ]
                self._quantity_metadata[quantity][
                    "numpy_postproc_start_index"
                ] = self.numpy_postproc_size
                self.numpy_postproc_size += np.prod(
                    self._quantity_metadata[quantity]["shape"]
                )
                self._quantity_metadata[quantity][
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
                self._quantity_metadata[quantity]["postproc_input_var"] = var
                assert metadata["shape"] == self._quantity_metadata[quantity]["shape"]
                assert (
                    metadata["global_shape"]
                    == self._quantity_metadata[quantity]["global_shape"]
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

        self._quantity_metadata[quantity] = {
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
                self._quantity_metadata[quantity]["step_input_var"] = var
                if self._quantity_metadata[quantity]["shape"] is None:
                    self._quantity_metadata[quantity]["shape"] = metadata["shape"]
                else:
                    assert (
                        metadata["shape"] == self._quantity_metadata[quantity]["shape"]
                    )
                if self._quantity_metadata[quantity]["global_shape"] is None:
                    self._quantity_metadata[quantity]["global_shape"] = metadata[
                        "global_shape"
                    ]
                else:
                    assert (
                        metadata["global_shape"]
                        == self._quantity_metadata[quantity]["global_shape"]
                    )

            elif var in acc_stage_input_vars:
                self._quantity_metadata[quantity]["accumulated_stage_var"] = var
                if self._quantity_metadata[quantity]["shape"] is None:
                    self._quantity_metadata[quantity]["shape"] = metadata["shape"]
                else:
                    assert (
                        metadata["shape"] == self._quantity_metadata[quantity]["shape"]
                    )
                if self._quantity_metadata[quantity]["global_shape"] is None:
                    self._quantity_metadata[quantity]["global_shape"] = metadata[
                        "global_shape"
                    ]
                else:
                    assert (
                        metadata["global_shape"]
                        == self._quantity_metadata[quantity]["global_shape"]
                    )

        for var, metadata in quantity_outputs.items():
            if var in stage_output_vars:
                self._quantity_metadata[quantity]["stage_output_var"] = var
            if self._quantity_metadata[quantity]["shape"] is None:
                self._quantity_metadata[quantity]["shape"] = metadata["shape"]
            else:
                assert metadata["shape"] == self._quantity_metadata[quantity]["shape"]
            if self._quantity_metadata[quantity]["global_shape"] is None:
                self._quantity_metadata[quantity]["global_shape"] = metadata[
                    "global_shape"
                ]
            else:
                assert (
                    metadata["global_shape"]
                    == self._quantity_metadata[quantity]["global_shape"]
                )
            if (
                self._quantity_metadata[quantity]["shape"]
                != self._quantity_metadata[quantity]["global_shape"]
            ):
                sizes = time_stage_problem.model._var_sizes["output"][
                    :, time_stage_problem.model._var_allprocs_abs2idx[var]
                ]
                self._quantity_metadata[quantity][
                    "local_indices_start"
                ] = start = np.sum(sizes[: self.comm.rank])
                self._quantity_metadata[quantity]["local_indices_end"] = (
                    start + sizes[self.comm.rank]
                )
            else:
                self._quantity_metadata[quantity]["local_indices_start"] = 0
                self._quantity_metadata[quantity]["local_indices_end"] = np.prod(
                    self._quantity_metadata[quantity]["shape"]
                )

    def _add_time_integration_inputs_and_outputs(self, quantity):
        time_stage_problem: om.Problem = self.options["time_stage_problem"]
        self.add_input(
            quantity + "_initial",
            shape=self._quantity_metadata[quantity]["shape"],
            val=time_stage_problem.get_val(
                self._quantity_metadata[quantity]["step_input_var"]
            )
            if self._quantity_metadata[quantity]["step_input_var"] is not None
            else np.zeros(self._quantity_metadata[quantity]["shape"]),
            distributed=self._quantity_metadata[quantity]["shape"]
            != self._quantity_metadata[quantity]["global_shape"],
        )
        self.add_output(
            quantity + "_final",
            copy_shape=quantity + "_initial",
            distributed=self._quantity_metadata[quantity]["shape"]
            != self._quantity_metadata[quantity]["global_shape"],
        )
        if self._quantity_metadata[quantity]["integrated"]:
            self.add_output(
                quantity + "_functional",
                copy_shape=quantity + "_initial",
                distributed=self._quantity_metadata[quantity]["shape"]
                != self._quantity_metadata[quantity]["global_shape"],
            )

    def _setup_wrt_and_of_vars(self):
        for quantity, metadata in self._quantity_metadata.items():
            if metadata["type"] == "time_integration":
                self.of_vars.append(metadata["stage_output_var"])
                if metadata["step_input_var"] is not None:
                    self.wrt_vars.append(metadata["step_input_var"])
                    self.wrt_vars.append(metadata["accumulated_stage_var"])
                if "postproc_input_var" in metadata:
                    self.wrt_vars_postproc.append(metadata["postproc_input_var"])
            elif metadata["type"] == "postprocessing":
                self.of_vars_postproc.append(metadata["postproc_output_var"])

    # TODO: Make work with postprocessing
    def to_numpy_array(self, om_vector: om_vector, np_array: np.ndarray):
        if om_vector._kind == "input":
            name_suffix = "_initial"
        else:
            name_suffix = "_final"

        for quantity, metadata in self._quantity_metadata.items():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = metadata["numpy_end_index"]
                np_array[start:end] = om_vector[quantity + name_suffix].flatten()

    def to_numpy_array_postprocessing(
        self, om_vector: om_vector, np_postproc_array: np.ndarray
    ):
        name_suffix = "_final"
        for quantity, metadata in self._quantity_metadata.items():
            if metadata["type"] == "postprocessing":
                start = metadata["numpy_postproc_start_index"]
                end = metadata["numpy_postproc_end_index"]
                np_postproc_array[start:end] = om_vector[
                    quantity + name_suffix
                ].flatten()

    def from_numpy_array(self, np_array: np.ndarray, om_vector: om_vector):
        if om_vector._kind == "input":
            name_suffix = "_initial"
        else:
            name_suffix = "_final"

        for quantity, metadata in self._quantity_metadata.items():
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
        for quantity, metadata in self._quantity_metadata.items():
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
                self._quantity_metadata,
                self.options["resets"],
            ),
            TimeStageProblemComputeJacvecFunctor(
                self.options["time_stage_problem"],
                self.options["integration_control"],
                self._quantity_metadata,
                self.of_vars,
                self.wrt_vars,
            ),
            TimeStageProblemComputeTransposeJacvecFunctor(
                self.options["time_stage_problem"],
                self.options["integration_control"],
                self._quantity_metadata,
                self.of_vars,
                self.wrt_vars,
            ),
        )

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
            start_functional = self._quantity_metadata[quantity][
                "numpy_functional_start_index"
            ]
            end_functional = self._quantity_metadata[quantity][
                "numpy_functional_end_index"
            ]
            if self._quantity_metadata[quantity]["type"] == "time_integration":
                start_time_stepping = self._quantity_metadata[quantity][
                    "numpy_start_index"
                ]
                end_time_stepping = self._quantity_metadata[quantity]["numpy_end_index"]
                contribution[start_functional:end_functional] = (
                    functional_coefficients.get_coefficient(timestep, quantity)
                    * serialized_state[start_time_stepping:end_time_stepping]
                )
            elif self._quantity_metadata[quantity]["type"] == "postprocessing":
                start_postprocessing = self._quantity_metadata[quantity][
                    "numpy_postproc_start_index"
                ]
                end_postprocessing = self._quantity_metadata[quantity][
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
            if self._quantity_metadata[quantity]["type"] == "time_integration":
                start = self._quantity_metadata[quantity]["numpy_start_index"]
                end = self._quantity_metadata[quantity]["numpy_end_index"]
                array[start:end] = output_vec[quantity + "_functional"].flatten()

    def functional_postprocessing_contribution_from_om_output_vec(
        self, output_vec: om_vector, array: np.ndarray
    ):
        for quantity in self.functional_quantities:
            if self._quantity_metadata[quantity]["type"] == "postprocessing":
                start = self._quantity_metadata[quantity]["numpy_postproc_start_index"]
                end = self._quantity_metadata[quantity]["numpy_postproc_end_index"]
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
            if self._quantity_metadata[quantity]["type"] == "time_integration":
                start = self._quantity_metadata[quantity]["numpy_start_index"]
                end = self._quantity_metadata[quantity]["numpy_end_index"]
                result[start:end] += (
                    functional_coefficents.get_coefficient(step, quantity)
                    * time_integration[start:end]
                )
            elif self._quantity_metadata["type"] == "postprocessing":
                start = self._quantity_metadata[quantity]["numpy_postproc_start_index"]
                end = self._quantity_metadata[quantity]["numpy_postproc_end_index"]
                postproc_copy[start:end] *= functional_coefficents.get_coefficient(
                    step, quantity
                )
        result += (
            self.postprocessor.postprocessing_computation_functor_jacvec_transposed(
                postproc_copy
            )
        )
        return result

    def add_functional_part_to_om_vec(
        self, functional_numpy_array: np.ndarray, om_vector: om_vector
    ):
        for quantity in self.functional_quantities:
            start = self._quantity_metadata[quantity]["numpy_functional_start_index"]
            end = self._quantity_metadata[quantity]["numpy_functional_end_index"]
            om_vector[quantity + "_functional"] = functional_numpy_array[
                start:end
            ].reshape(self._quantity_metadata[quantity]["shape"])
