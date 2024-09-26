# pylint: disable=missing-module-docstring, protected-access

from typing import Optional
import h5py
import numpy as np
import openmdao.api as om
from openmdao.vectors.vector import Vector as OMVector


from .butcher_tableau import ButcherTableau
from .functional_coefficients import FunctionalCoefficients, EmptyFunctionalCoefficients
from .integration_control import IntegrationControl
from .runge_kutta_scheme import RungeKuttaScheme, EmbeddedRungeKuttaScheme
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
from .checkpoint_interface.checkpoint_interface import CheckpointInterface
from .checkpoint_interface.no_checkpointer import NoCheckpointer
from .errors import SetupError


class RungeKuttaIntegrator(om.ExplicitComponent):
    """Outer component for solving time-dependent problems with explicit or diagonally
    implicit Runge-Kutta schemes. One stage of the scheme is modelled by an inner
    OpenMDAO-problem. Optionally, time-step postprocessing and calculationof linear
    combinations of quantities can be done.
    OpenMDAO inputs: - initial values of the quantities for the time integration
    OpenMDAO output: - final values of the quantities for the time integration
                     - (optional) postprocessed final values
                     - (optional) linear combinations of quantities over time
    """

    _quantity_metadata: dict
    _functional_quantities: list
    _of_vars: list
    _wrt_vars: list
    _of_vars_postproc: list
    _wrt_vars_postproc: list

    _runge_kutta_scheme: Optional[RungeKuttaScheme]
    _serialized_state: Optional[np.ndarray]
    _accumulated_stages: Optional[np.ndarray]
    _stage_cache: Optional[np.ndarray]
    _serialized_state_perturbations: Optional[np.ndarray]
    _serialized_state_perturbations_from_functional: Optional[np.ndarray]
    _accumulated_stage_perturbations: Optional[np.ndarray]
    _stage_perturbations_cache: Optional[np.ndarray]

    _postprocessor: Optional[Postprocessor]
    _postprocessing_state: Optional[np.ndarray]
    _postprocessing_state_perturbations: Optional[np.ndarray]

    _functional_part: Optional[np.ndarray]
    _functional_part_perturbations: Optional[np.ndarray]

    _numpy_array_size: int
    _numpy_postproc_size: int
    _numpy_functional_size: int

    _cached_input: np.ndarray

    _disable_write_out: bool

    _checkpointer: Optional[CheckpointInterface]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._quantity_metadata = {}
        self._functional_quantities = []
        self._of_vars = []
        self._wrt_vars = []
        self._of_vars_postproc = []
        self._wrt_vars_postproc = []

        self._runge_kutta_scheme = None
        self._serialized_state = None
        self._accumulated_stages = None
        self._stage_cache = None
        self._serialized_state_perturbations = None
        self._serialized_state_perturbations_from_functional = None
        self._accumulated_stage_perturbations = None
        self._stage_perturbations_cache = None

        self._postprocessor = None
        self._postprocessing_state = None
        self._postprocessing_state_perturbations = None

        self._functional_part = None
        self._functional_part_perturbations = None

        self._numpy_array_size = 0
        self._numpy_postproc_size = 0
        self._numpy_functional_size = 0

        self._cached_input = np.full(0, np.nan)
        self._disable_write_out = False

        self._checkpointer = None

    def initialize(self):
        self.options.declare(
            "time_stage_problem",
            types=om.Problem,
            desc="""openMDAO problem used to model one Runge-Kutta stage. All variables
            relevant to the time-integration need to have two tags. One to declare to
            which quantity the variable belongs, and a second to differentiate between
            their roles in the time integration. For inputs, there is the tag
            'step_input_var' specifying the state at the start of the time step, and
            there is the tag 'accumulated_stage_var' used for specifying a variable that
            contains the weighted sum (according to the butcher matrix) of the stage
            updates up to the previous stage. For outputs, there is only the tag
            'stage_output_var', which marks the variable whose value is used at the end
            of the time step to calculate the new state. Per quantity, there needs to be
            either three variables which cover all of the time integration tags, or just
            one output variable with the 'stage_output_var' tag (this should be used for
            quantities that only depend on time and the state of other quantities,
            but not its own previous state).""",
        )

        self.options.declare(
            "postprocessing_problem",
            types=(om.Problem, None),
            desc="""An optional openMDAO problem used to calculate derived quantities
            based on the results of the time steps. All variables relevant to the
            outside need to have two tags, one as name for the own quantity, and one to 
            declare their role in the time stepping. For inputs, the quantity tag needs
            to be the same as one quantity of the time_stage_problem, and the second tag
            needs to be 'postproc_input_var'. For outputs, the quantity tags can be
            chosen freely to best describe the quantity (but shouldn't be one already
            used in the time integration), while the other tag needs to be
            'postproc_output_var'.""",
            default=None,
        )

        self.options.declare(
            "butcher_tableau",
            types=ButcherTableau,
            desc="""The butcher tableau for the RK-scheme. Needs to be explicit or
            diagonally implicit (i.e all zeros in the upper triangle of the butcher
            matrix).""",
        )
        self.options.declare(
            "integration_control",
            types=IntegrationControl,
            desc="""Object used to exchange (meta)data between the inner and outer
            problems. In particular, this class modifies the (meta)data like the current
            diagonal element of the butcher tableau, or the current time step and stage,
            which then can be read by anyone else holding the same instance (like e.g.
            components in the time_stage_problem).""",
        )

        self.options.declare(
            "write_out_distance",
            types=int,
            default=0,
            desc="""Toggles the write out of data of the quantities. If
            write_out_distance == 0, no data is written out. Else, every
            "write_out_distance"th time step the data of the quantities are written out
            to the write_file, starting with the initial values. The data at the end of
            the last time step is written out even if the last time step is not a
            "write_out_distance"th time step.""",
        )

        self.options.declare(
            "write_file",
            types=str,
            default="data.h5",
            desc="""The file where the results of each time steps are written if
            write_out_distance != 0.""",
        )

        self.options.declare(
            "time_integration_quantities",
            types=list,
            desc="""List of tags used to describe the quantities that are time
            integrated by the RK integrator. These tags fulfil multiple roles, and need
            to be set at different points in the inner problems:
                1. Based on these tags, inputs and outputs are added to the
                RungeKuttaIntegrator. Per tag there will be *tag*_initial as input and
                *tag*_final as output.
                2. Inputs and outputs with these tags need to be present in the
                time_stage_problem. In detail, per tag there need to be either both
                inputs tagged with [*tag*,"step_input_var", ...] and 
                [*tag*, "accumulated_stage_var", ...] as well as an output with
                [*tag*, "stage_output_var", ...], or just an output with 
                [*tag*, "stage_output_var"] (in case where there is no dependence on the
                previous state of the quantity in the ODE)
                3. If you use postprocessing, and if the postprocessing should use the
                quantity, than there needs to be an input with the tags
                [*tag*, "postproc_input_var", ...] somewhere in
                the postprocessing_problem.
                """,
        )

        self.options.declare(
            "postprocessing_quantities",
            types=list,
            default=[],
            desc="""List of tags used to describe the quantities computed by the
            postprocessing. These tags fulfil multiple roles:
                1. Based on these tags, outputs are added to the RUngeKuttaIntegrator.
                Per tag there will be *tag*_final as output, containing the
                postprocessing of the final state of the time integration.
                2. Outputs with these tags need to be present in the
                postprocessing_problem. In detail, per tag there needs to be an output
                with the tags [*tag*, "postproc_output_var].""",
        )

        self.options.declare(
            "functional_coefficients",
            types=FunctionalCoefficients,
            default=EmptyFunctionalCoefficients(),  # By default, don't compute any
            # linear combination.
            desc="""A FunctionalCoefficients object that can return a list of quantities
            (which needs to be a subset of the time integration and postprocessing ones)
            over which linear combinations are evaluated, as well as a coefficient given
            a time step and a quantity. Per quantity returned by the object, an output
            is added, named *quantity*_functional.""",
        )

        self.options.declare(
            "checkpointing_type",
            check_valid=self.check_checkpointing_type,
            default=NoCheckpointer,
            desc="""Type of checkpointing used. Must be a subclass of
            CheckpointInterface""",
        )

        self.options.declare(
            "checkpoint_options",
            types=dict,
            default={},
            desc="""Additional options passed the the checkpointer. Valid options depend
            on the used checkpointing_type""",
        )

    @staticmethod
    def check_checkpointing_type(name, value):
        """Checks whether the passed checkpointing type for the options is an actual
        subclass of CheckpointInterface"""
        # pylint: disable=unused-argument
        # OpenMDAO needs that specific interface, even if we don't need it fully.
        if not issubclass(value, CheckpointInterface):
            raise TypeError(f"{value} is not a subclass of CheckpointInterface")

    def setup(self):
        self._setup_inner_problems()
        self._setup_variables()
        self._setup_arrays()
        self._setup_runge_kutta_scheme()
        self._setup_postprocessor()
        self._configure_write_out()
        self._setup_write_file()
        self._setup_checkpointing()
        # TODO: maybe add methods for time-independent in/outputs

    def _setup_inner_problems(self):
        self.options["time_stage_problem"].setup()
        self.options["time_stage_problem"].final_setup()
        if self.options["postprocessing_problem"] is not None:
            self.options["postprocessing_problem"].setup()
            self.options["postprocessing_problem"].final_setup()

    def _setup_variables(self):
        self._functional_quantities = self.options[
            "functional_coefficients"
        ].list_quantities()

        self._setup_variable_information()
        self._setup_wrt_and_of_vars()

    def _setup_variable_information(self):
        time_stage_problem: om.Problem = self.options["time_stage_problem"]
        postprocessing_problem: om.Problem = self.options["postprocessing_problem"]
        self._numpy_array_size = 0
        self._numpy_postproc_size = 0
        self._numpy_functional_size = 0
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
                self._extract_time_integration_metadata_from_postprocessing_problem(
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
                raise SetupError(
                    f"""Warning! The time integration problem contains a nonworking
                    combination of variables for quantity {quantity} on rank
                    {self.comm.rank}. Check that there is an output with the tags
                    'stage_output_var' and {quantity} and either both or no inputs with 
                    tags=['step_input_var', {quantity}] and 
                    tags=['accumulated_stage_var', {quantity}]."""
                )
            self._quantity_metadata[quantity][
                "numpy_start_index"
            ] = self._numpy_array_size
            self._numpy_array_size += np.prod(
                self._quantity_metadata[quantity]["shape"]
            )
            self._quantity_metadata[quantity][
                "numpy_end_index"
            ] = self._numpy_array_size
            self._add_functional_quantity_metadata(quantity)
            self._add_time_integration_inputs_and_outputs(quantity)
        if postprocessing_problem is not None:
            for quantity in self.options["postprocessing_quantities"]:
                self._extract_postprocessing_metadata_from_postprocessing_problem(
                    quantity, postproc_output_vars
                )
                self._add_functional_quantity_metadata(quantity)
                self._add_postprocessing_outputs(quantity)

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
                self._extract_quantity_metadata_or_check_for_consistency(
                    quantity, metadata
                )
            elif var in acc_stage_input_vars:
                self._quantity_metadata[quantity]["accumulated_stage_var"] = var
                self._extract_quantity_metadata_or_check_for_consistency(
                    quantity, metadata
                )

        for var, metadata in quantity_outputs.items():
            if var in stage_output_vars:
                self._quantity_metadata[quantity]["stage_output_var"] = var
            self._extract_quantity_metadata_or_check_for_consistency(quantity, metadata)
            if (
                self._quantity_metadata[quantity]["shape"]
                != self._quantity_metadata[quantity]["global_shape"]
            ):
                try:
                    sizes = time_stage_problem.model._var_sizes["output"][
                        :, time_stage_problem.model._var_allprocs_abs2idx[var]
                    ]
                except KeyError:  # Have to deal with an older openMDAO version
                    sizes = time_stage_problem.model._var_sizes["nonlinear"]["output"][
                        :,
                        time_stage_problem.model._var_allprocs_abs2idx["nonlinear"][
                            var
                        ],
                    ]
                self._quantity_metadata[quantity]["local_indices_start"] = start = (
                    np.sum(sizes[: self.comm.rank])
                )
                self._quantity_metadata[quantity]["local_indices_end"] = (
                    start + sizes[self.comm.rank]
                )
            else:
                self._quantity_metadata[quantity]["local_indices_start"] = 0
                self._quantity_metadata[quantity]["local_indices_end"] = np.prod(
                    self._quantity_metadata[quantity]["shape"]
                )

    def _extract_quantity_metadata_or_check_for_consistency(
        self, quantity: str, metadata: dict
    ):
        if self._quantity_metadata[quantity]["shape"] is None:
            self._quantity_metadata[quantity]["shape"] = metadata["shape"]
        else:
            assert metadata["shape"] == self._quantity_metadata[quantity]["shape"]
        if self._quantity_metadata[quantity]["global_shape"] is None:
            self._quantity_metadata[quantity]["global_shape"] = metadata["global_shape"]
        else:
            assert (
                metadata["global_shape"]
                == self._quantity_metadata[quantity]["global_shape"]
            )

    def _extract_time_integration_metadata_from_postprocessing_problem(
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

    def _extract_postprocessing_metadata_from_postprocessing_problem(
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
                if (
                    self._quantity_metadata[quantity]["shape"]
                    != self._quantity_metadata[quantity]["global_shape"]
                ):
                    sizes = postprocessing_problem.model._var_sizes["nonlinear"][
                        "output"
                    ][
                        :,
                        postprocessing_problem.model._var_allprocs_abs2idx["nonlinear"][
                            var
                        ],
                    ]
                    self._quantity_metadata[quantity]["local_indices_start"] = start = (
                        np.sum(sizes[: self.comm.rank])
                    )
                    self._quantity_metadata[quantity]["local_indices_end"] = (
                        start + sizes[self.comm.rank]
                    )
                else:
                    self._quantity_metadata[quantity]["local_indices_start"] = 0
                    self._quantity_metadata[quantity]["local_indices_end"] = np.prod(
                        self._quantity_metadata[quantity]["shape"]
                    )
                self._quantity_metadata[quantity][
                    "numpy_postproc_start_index"
                ] = self._numpy_postproc_size
                self._numpy_postproc_size += np.prod(
                    self._quantity_metadata[quantity]["shape"]
                )
                self._quantity_metadata[quantity][
                    "numpy_postproc_end_index"
                ] = self._numpy_postproc_size

    def _add_functional_quantity_metadata(self, quantity):
        if quantity in self._functional_quantities:
            self._quantity_metadata[quantity]["integrated"] = True
            self._quantity_metadata[quantity][
                "numpy_functional_start_index"
            ] = self._numpy_functional_size
            self._numpy_functional_size += np.prod(
                self._quantity_metadata[quantity]["shape"]
            )
            self._quantity_metadata[quantity][
                "numpy_functional_end_index"
            ] = self._numpy_functional_size

    def _add_time_integration_inputs_and_outputs(self, quantity):
        time_stage_problem: om.Problem = self.options["time_stage_problem"]
        # time_stage_problem.comm.Barrier()
        self.add_input(
            quantity + "_initial",
            shape=self._quantity_metadata[quantity]["shape"],
            val=(
                time_stage_problem.get_val(
                    self._quantity_metadata[quantity][
                        "step_input_var"
                    ],  # get_remote=False
                )
                if self._quantity_metadata[quantity]["step_input_var"] is not None
                else np.zeros(self._quantity_metadata[quantity]["shape"])
            ),
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

    def _setup_wrt_and_of_vars(self):
        self._of_vars = []
        self._wrt_vars = []
        for metadata in self._quantity_metadata.values():
            if metadata["type"] == "time_integration":
                self._of_vars.append(metadata["stage_output_var"])
                if metadata["step_input_var"] is not None:
                    self._wrt_vars.append(metadata["step_input_var"])
                    self._wrt_vars.append(metadata["accumulated_stage_var"])
                if "postproc_input_var" in metadata:
                    self._wrt_vars_postproc.append(metadata["postproc_input_var"])
            elif metadata["type"] == "postprocessing":
                self._of_vars_postproc.append(metadata["postproc_output_var"])

    def _setup_arrays(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        self._serialized_state = np.zeros(self._numpy_array_size)
        self._accumulated_stages = np.zeros(self._numpy_array_size)
        self._stage_cache = np.zeros(
            (butcher_tableau.number_of_stages(), self._numpy_array_size)
        )
        self._serialized_state_perturbations = np.zeros(self._numpy_array_size)
        self._serialized_state_perturbations_from_functional = np.zeros(
            self._numpy_array_size
        )
        self._accumulated_stage_perturbations = np.zeros(self._numpy_array_size)
        self._stage_perturbations_cache = np.zeros(
            (butcher_tableau.number_of_stages(), self._numpy_array_size)
        )
        self._postprocessing_state = np.zeros(self._numpy_postproc_size)
        self._postprocessing_state_perturbations = np.zeros(self._numpy_postproc_size)
        self._functional_part = np.zeros(self._numpy_functional_size)
        self._functional_part_perturbations = np.zeros(self._numpy_functional_size)

    def _setup_runge_kutta_scheme(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        if not butcher_tableau.is_embedded:
            self._runge_kutta_scheme = RungeKuttaScheme(
                butcher_tableau,
                TimeStageProblemComputeFunctor(
                    self.options["time_stage_problem"],
                    self.options["integration_control"],
                    self._quantity_metadata,
                ),
                TimeStageProblemComputeJacvecFunctor(
                    self.options["time_stage_problem"],
                    self.options["integration_control"],
                    self._quantity_metadata,
                    self._of_vars,
                    self._wrt_vars,
                ),
                TimeStageProblemComputeTransposeJacvecFunctor(
                    self.options["time_stage_problem"],
                    self.options["integration_control"],
                    self._quantity_metadata,
                    self._of_vars,
                    self._wrt_vars,
                ),
            )
        else:
            self._runge_kutta_scheme = EmbeddedRungeKuttaScheme(
                butcher_tableau,
                TimeStageProblemComputeFunctor(
                    self.options["time_stage_problem"],
                    self.options["integration_control"],
                    self._quantity_metadata,
                ),
                TimeStageProblemComputeJacvecFunctor(
                    self.options["time_stage_problem"],
                    self.options["integration_control"],
                    self._quantity_metadata,
                    self._of_vars,
                    self._wrt_vars,
                ),
                TimeStageProblemComputeTransposeJacvecFunctor(
                    self.options["time_stage_problem"],
                    self.options["integration_control"],
                    self._quantity_metadata,
                    self._of_vars,
                    self._wrt_vars,
                ),
            )


    def _setup_postprocessor(self):
        postprocessing_problem: om.Problem = self.options["postprocessing_problem"]
        if postprocessing_problem is not None:
            self._postprocessor = Postprocessor(
                PostprocessingProblemComputeFunctor(
                    postprocessing_problem,
                    self._quantity_metadata,
                    self._numpy_array_size,
                    self._numpy_postproc_size,
                ),
                PostprocessingProblemComputeJacvecFunctor(
                    postprocessing_problem,
                    self._quantity_metadata,
                    self._numpy_array_size,
                    self._numpy_postproc_size,
                    self._of_vars_postproc,
                    self._wrt_vars_postproc,
                ),
                PostprocessingProblemComputeTransposeJacvecFunctor(
                    postprocessing_problem,
                    self._quantity_metadata,
                    self._numpy_array_size,
                    self._numpy_postproc_size,
                    self._of_vars_postproc,
                    self._wrt_vars_postproc,
                ),
            )

    def _configure_write_out(self):
        self._disable_write_out = self.options["write_out_distance"] == 0

    def _setup_write_file(self):
        if not self._disable_write_out:
            delta_t = self.options["integration_control"].delta_t
            initial_time = self.options["integration_control"].initial_time
            num_steps = self.options["integration_control"].num_steps
            for i in range(self.comm.size):
                if i == self.comm.rank:
                    with h5py.File(
                        self.options["write_file"], mode="w" if i == 0 else "r+"
                    ) as f:
                        for quantity, metadata in self._quantity_metadata.items():
                            for j in range(
                                0,
                                num_steps,
                                self.options["write_out_distance"],
                            ):
                                # check whether dataset was already created by other
                                # process
                                path = quantity + "/" + str(j)
                                if path not in f:
                                    dataset = f.create_dataset(
                                        path,
                                        data=np.full(metadata["global_shape"], np.nan),
                                    )
                                    dataset.attrs["time"] = j * delta_t + initial_time
                            path = quantity + "/" + str(num_steps)
                            if path not in f:
                                dataset = f.create_dataset(
                                    path,
                                    data=np.full(metadata["global_shape"], np.nan),
                                )
                                dataset.attrs["time"] = (
                                    num_steps * delta_t + initial_time
                                )
                if self.comm.size > 1:
                    self.comm.Barrier()

    def _setup_checkpointing(self):
        self._checkpointer = self.options["checkpointing_type"](
            array_size=self._numpy_array_size,
            num_steps=self.options["integration_control"].num_steps,
            run_step_func=self._run_step,
            run_step_jacvec_rev_func=self._run_step_jacvec_rev,
            **self.options["checkpoint_options"],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.comm.rank == 0:
            print("\n\nStarting compute\n\n")
        self._compute_preparation_phase(inputs)
        self._compute_postprocessing_phase()
        self._compute_initial_write_out_phase()
        self._compute_functional_phase()
        self._compute_checkpointing_setup_phase()
        self._checkpointer.iterate_forward(self._serialized_state)
        self._compute_translate_to_om_vector_phase(outputs)
        if self.comm.rank == 0:
            print("\n\nFinishing compute\n\n")

    def _compute_preparation_phase(self, inputs: OMVector):
        self._to_numpy_array(inputs, self._serialized_state)
        self._cached_input = self._serialized_state.copy()
        self.options["integration_control"].reset()

    def _to_numpy_array(self, om_vector: OMVector, np_array: np.ndarray):
        if om_vector._kind == "input":
            name_suffix = "_initial"
        else:
            name_suffix = "_final"
        for quantity, metadata in self._quantity_metadata.items():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = metadata["numpy_end_index"]
                np_array[start:end] = om_vector[quantity + name_suffix].flatten()

    def _compute_postprocessing_phase(self):
        if self.options["postprocessing_problem"] is not None:
            if self.comm.rank == 0:
                print("Starting postprocessing.")
            self._postprocessing_state = self._postprocessor.postprocess(
                self._serialized_state
            )
            if self.comm.rank == 0:
                print("Finished postprocessing.")

    def _compute_initial_write_out_phase(self):
        if not self._disable_write_out:
            print(self.comm.rank, "-----------------------------------------------")
            self._write_out(
                0,
                self._serialized_state,
                self._postprocessing_state,
                open_mode="r+",
            )

    def _write_out(
        self,
        step: int,
        serialized_state: np.ndarray,
        postprocessing_state=None,
        open_mode="r+",
    ):
        further_args = {}
        if self.comm.size > 1:
            further_args["driver"] = "mpio"
            further_args["comm"] = self.comm
        with h5py.File(self.options["write_file"], mode=open_mode, **further_args) as f:
            if self.comm.size > 1:
                f.atomic = True
            for quantity, metadata in self._quantity_metadata.items():
                if metadata["type"] == "time_integration":
                    start = metadata["numpy_start_index"]
                    end = metadata["numpy_end_index"]
                    # check whether data has already been written on dataset by checking
                    # first element
                    if np.isnan(
                        f[quantity][str(step)][
                            np.unravel_index(
                                metadata["local_indices_start"],
                                metadata["global_shape"],
                            )
                        ]
                    ):
                        start_tuple = np.unravel_index(
                            metadata["local_indices_start"],
                            metadata["global_shape"],
                        )
                        end_tuple = np.unravel_index(
                            metadata["local_indices_end"] - 1,
                            metadata["global_shape"],
                        )
                        access_list = []
                        for start_index, end_index in zip(start_tuple, end_tuple):
                            access_list.append(slice(start_index, end_index + 1))
                        f[quantity][str(step)][tuple(access_list)] = serialized_state[
                            start:end
                        ].reshape(metadata["shape"])
                elif metadata["type"] == "postprocessing":
                    start = metadata["numpy_postproc_start_index"]
                    end = metadata["numpy_postproc_end_index"]
                    if np.isnan(
                        f[quantity][str(step)][
                            np.unravel_index(
                                metadata["local_indices_start"],
                                metadata["global_shape"],
                            )
                        ]
                    ):
                        start_tuple = np.unravel_index(
                            metadata["local_indices_start"],
                            metadata["global_shape"],
                        )
                        end_tuple = np.unravel_index(
                            metadata["local_indices_end"] - 1,
                            metadata["global_shape"],
                        )
                        access_list = []
                        for start_index, end_index in zip(start_tuple, end_tuple):
                            access_list.append(slice(start_index, end_index + 1))
                        f[quantity][str(step)][tuple(access_list)] = (
                            postprocessing_state[start:end].reshape(metadata["shape"])
                        )

    def _compute_functional_phase(self):
        if self._functional_quantities:
            if self.comm.rank == 0:
                print("Starting computation of functional contribution.")
            self._functional_part = self._get_functional_contribution(
                self._serialized_state, self._postprocessing_state, 0
            )
            if self.comm.rank == 0:
                print("Finished computation of functional contribution.")

    def _get_functional_contribution(
        self,
        serialized_state: np.ndarray,
        postprocessing_state: np.ndarray,
        timestep: int,
    ) -> np.ndarray:
        functional_coefficients: FunctionalCoefficients = self.options[
            "functional_coefficients"
        ]
        contribution = np.zeros(self._numpy_functional_size)
        for quantity in self._functional_quantities:
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

    def _compute_checkpointing_setup_phase(self):
        self._checkpointer.create_checkpointer()

    def _compute_translate_to_om_vector_phase(self, outputs: OMVector):
        self._from_numpy_array(self._serialized_state, outputs)
        if self.options["postprocessing_problem"] is not None:
            self._from_numpy_array_postprocessing(self._postprocessing_state, outputs)
        if self._functional_quantities:
            self._add_functional_part_to_om_vec(self._functional_part, outputs)

    def _from_numpy_array(self, np_array: np.ndarray, om_vector: OMVector):
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

    def _from_numpy_array_postprocessing(
        self, np_postproc_array: np.ndarray, om_vector: OMVector
    ):
        name_suffix = "_final"  # postprocessing variables are only in output, not input
        for quantity, metadata in self._quantity_metadata.items():
            if metadata["type"] == "postprocessing":
                start = metadata["numpy_postproc_start_index"]
                end = metadata["numpy_postproc_end_index"]
                om_vector[quantity + name_suffix] = np_postproc_array[
                    start:end
                ].reshape(metadata["shape"])

    def _add_functional_part_to_om_vec(
        self, functional_numpy_array: np.ndarray, om_vector: OMVector
    ):
        for quantity in self._functional_quantities:
            start = self._quantity_metadata[quantity]["numpy_functional_start_index"]
            end = self._quantity_metadata[quantity]["numpy_functional_end_index"]
            om_vector[quantity + "_functional"] = functional_numpy_array[
                start:end
            ].reshape(self._quantity_metadata[quantity]["shape"])

    def _run_step(self, step, serialized_state):
        if self.comm.rank == 0:
            print(f"\nStarting step {step} of compute.\n")
        self._run_step_preparation_phase(step, serialized_state)
        self._run_step_time_integration_phase()
        self._run_step_postprocessing_phase()
        self._run_step_functional_phase()
        self._run_step_write_out_phase()
        if self.comm.rank == 0:
            print(f"\nFinishing step {step} of compute.\n")
        return self._serialized_state

    def _run_step_preparation_phase(self, step, serialized_state):
        self._update_integration_control_step(step)
        self._serialized_state = serialized_state

    def _update_integration_control_step(self, step):
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        self.options["integration_control"].step = step
        time = initial_time + (step - 1) * delta_t
        self.options["integration_control"].step_time_old = time
        self.options["integration_control"].step_time_new = time + delta_t

    def _run_step_time_integration_phase(self):
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        for stage in range(butcher_tableau.number_of_stages()):
            self._stage_computation(stage)
        self._serialized_state = self._runge_kutta_scheme.compute_step(
            delta_t, self._serialized_state, self._stage_cache
        )

    def _stage_computation(self, stage):
        delta_t = self.options["integration_control"].delta_t
        time = self.options["integration_control"].step_time_old
        step = self.options["integration_control"].step
        if self.comm.rank == 0:
            print(f"Starting stage {stage} of compute in step {step}.")
        self.options["integration_control"].stage = stage
        if stage != 0:
            self._accumulated_stages = (
                self._runge_kutta_scheme.compute_accumulated_stages(
                    stage, self._stage_cache
                )
            )
        else:
            self._accumulated_stages.fill(0.0)
        self._stage_cache[stage, :] = self._runge_kutta_scheme.compute_stage(
            stage, delta_t, time, self._serialized_state, self._accumulated_stages
        )
        if self.comm.rank == 0:
            print(f"Finished stage {stage} of compute in step {step}.")

    def _run_step_postprocessing_phase(self):
        if self.options["postprocessing_problem"] is not None:
            if self.comm.rank == 0:
                print("Starting postprocessing.")
            self._postprocessing_state = self._postprocessor.postprocess(
                self._serialized_state
            )
            if self.comm.rank == 0:
                print("Finished postprocessing.")

    def _run_step_functional_phase(self):
        step = self.options["integration_control"].step
        if self._functional_quantities:
            if self.comm.rank == 0:
                print("Starting computation of functional contribution.")
            self._functional_part += self._get_functional_contribution(
                self._serialized_state, self._postprocessing_state, step
            )
            if self.comm.rank == 0:
                print("Finished computation of functional contribution.")

    def _run_step_write_out_phase(self):
        step = self.options["integration_control"].step
        if not self._disable_write_out and (
            step % self.options["write_out_distance"] == 0
            or step == self.options["integration_control"].num_steps
        ):
            self._write_out(
                step,
                self._serialized_state,
                self._postprocessing_state,
                open_mode="r+",
            )

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):  # pylint: disable = arguments-differ
        if mode == "fwd":
            self._compute_jacvec_product_fwd(inputs, d_inputs, d_outputs)
        elif mode == "rev":
            self._compute_jacvec_product_rev(inputs, d_inputs, d_outputs)

    def _compute_jacvec_product_fwd(self, inputs, d_inputs, d_outputs):
        if self.comm.rank == 0:
            print("\n\nStarting fwd-mode jacvec product\n\n")
        self._compute_jacvec_fwd_preparation_phase(inputs, d_inputs)
        self._compute_jacvec_fwd_postprocessing_phase()
        self._compute_jacvec_fwd_functional_phase()
        self._compute_jacvec_fwd_run_steps()
        self._compute_jacvec_fwd_translate_to_om_vector_phase(d_outputs)
        if self.comm.rank == 0:
            print("\n\nFinished fwd-mode jacvec product\n\n")

    def _compute_jacvec_fwd_preparation_phase(self, inputs, d_inputs):
        self._to_numpy_array(inputs, self._serialized_state)
        self._to_numpy_array(d_inputs, self._serialized_state_perturbations)
        self.options["integration_control"].reset()

    def _compute_jacvec_fwd_postprocessing_phase(self):
        if self.options["postprocessing_problem"] is not None:
            if self.comm.rank == 0:
                print("Starting postprocessing.")
            self._postprocessor.postprocess(self._serialized_state)
            self._postprocessing_state_perturbations = (
                self._postprocessor.postprocess_jacvec(
                    self._serialized_state_perturbations
                )
            )
            if self.comm.rank == 0:
                print("Finished postprocessing.")

    def _compute_jacvec_fwd_functional_phase(self):
        if self._functional_quantities:
            if self.comm.rank == 0:
                print("Starting computation of functional contribution.")
            self._functional_part_perturbations = self._get_functional_contribution(
                self._serialized_state_perturbations,
                self._postprocessing_state_perturbations,
                0,
            )
            if self.comm.rank == 0:
                print("Finished computation of functional contribution.")

    def _compute_jacvec_fwd_run_steps(self):
        num_steps = self.options["integration_control"].num_steps
        for step in range(1, num_steps + 1):
            if self.comm.rank == 0:
                print(f"\nStarting step {step} of fwd-mode jacvec product.\n")
            self._compute_jacvec_fwd_run_steps_preparation_phase(step)
            self._compute_jacvec_fwd_run_steps_time_integration_phase()
            self._compute_jacvec_fwd_run_steps_postprocessing_phase()
            self._compute_jacvec_fwd_run_steps_functional_phase()
            if self.comm.rank == 0:
                print(f"\nFinished step {step} of fwd-mode jacvec product.\n")

    def _compute_jacvec_fwd_run_steps_preparation_phase(self, step):
        self._update_integration_control_step(step)

    def _compute_jacvec_fwd_run_steps_time_integration_phase(self):
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        time = self.options["integration_control"].step_time_old
        step = self.options["integration_control"].step
        for stage in range(butcher_tableau.number_of_stages()):
            if self.comm.rank == 0:
                print(
                    f"Starting stage {stage} of fwd-mode jacvec productin step {step}."
                )
            if stage != 0:
                self._accumulated_stages = (
                    self._runge_kutta_scheme.compute_accumulated_stages(
                        stage, self._stage_cache
                    )
                )
                self._accumulated_stage_perturbations = (
                    self._runge_kutta_scheme.compute_accumulated_stage_perturbations(
                        stage, self._stage_perturbations_cache
                    )
                )
            else:
                self._accumulated_stages.fill(0.0)
                self._accumulated_stage_perturbations.fill(0.0)
            self._stage_cache[stage, :] = self._runge_kutta_scheme.compute_stage(
                stage, delta_t, time, self._serialized_state, self._accumulated_stages
            )
            self._stage_perturbations_cache[stage, :] = (
                self._runge_kutta_scheme.compute_stage_jacvec(
                    stage,
                    delta_t,
                    time,
                    self._serialized_state_perturbations,
                    self._accumulated_stage_perturbations,
                )
            )
            if self.comm.rank == 0:
                print(
                    f"Finished stage {stage} of fwd-mode jacvec product in step {step}."
                )
        self._serialized_state = self._runge_kutta_scheme.compute_step(
            delta_t, self._serialized_state, self._stage_cache
        )

        self._serialized_state_perturbations = (
            self._runge_kutta_scheme.compute_step_jacvec(
                delta_t,
                self._serialized_state_perturbations,
                self._stage_perturbations_cache,
            )
        )

    def _compute_jacvec_fwd_run_steps_postprocessing_phase(self):
        if self.options["postprocessing_problem"] is not None:
            if self.comm.rank == 0:
                print("Starting postprocessing.")
            self._postprocessor.postprocess(self._serialized_state)
            self._postprocessing_state_perturbations = (
                self._postprocessor.postprocess_jacvec(
                    self._serialized_state_perturbations
                )
            )
            if self.comm.rank == 0:
                print("Finished postprocessing.")

    def _compute_jacvec_fwd_run_steps_functional_phase(self):
        step = self.options["integration_control"].step
        if self._functional_quantities:
            if self.comm.rank == 0:
                print("Starting computation of functional contribution.")
            self._functional_part_perturbations += self._get_functional_contribution(
                self._serialized_state_perturbations,
                self._postprocessing_state_perturbations,
                step,
            )
            if self.comm.rank == 0:
                print("Finished computation of functional contribution.")

    def _compute_jacvec_fwd_translate_to_om_vector_phase(
        self,
        d_outputs: OMVector,
    ):
        self._from_numpy_array(self._serialized_state_perturbations, d_outputs)
        if self.options["postprocessing_problem"] is not None:
            self._from_numpy_array_postprocessing(
                self._postprocessing_state_perturbations, d_outputs
            )
        if self._functional_quantities:
            self._add_functional_part_to_om_vec(
                self._functional_part_perturbations, d_outputs
            )

    def _compute_jacvec_product_rev(self, inputs, d_inputs, d_outputs):
        if self.comm.rank == 0:
            print("\n\nStarting rev-mode jacvec product\n\n")
        self._disable_write_out = True
        self._to_numpy_array(inputs, self._serialized_state)
        if not np.array_equal(self._cached_input, self._serialized_state):
            if self.comm.rank == 0:
                print("Preparing checkpoints by calling compute()")
            outputs = self._vector_class("nonlinear", "output", self)
            self.compute(inputs, outputs)
        self._to_numpy_array(d_outputs, self._serialized_state_perturbations)
        self._to_numpy_array_postprocessing(
            d_outputs, self._postprocessing_state_perturbations
        )
        if self.options["postprocessing_problem"] is not None:
            if self.comm.rank == 0:
                print("Starting postprocessing.")
            self._serialized_state_perturbations += (
                self._postprocessor.postprocess_jacvec_transposed(
                    self._postprocessing_state_perturbations
                )
            )
            if self.comm.rank == 0:
                print("Finished postprocessing.")
        if self._functional_quantities:
            if self.comm.rank == 0:
                print("Finished computation of functional contribution.")
            self._get_functional_contribution_from_om_output_vec(d_outputs)
            self._add_functional_perturbations_to_state_perturbations(
                self.options["integration_control"].num_steps
            )
            if self.comm.rank == 0:
                print("Finished computation of functional contribution.")
        self._checkpointer.iterate_reverse(self._serialized_state_perturbations)

        self._from_numpy_array(self._serialized_state_perturbations, d_inputs)
        self._configure_write_out()

        # using the checkpoints invalidates the checkpointer in general
        self._cached_input = np.full(0, np.nan)

        if self.comm.rank == 0:
            print("\n\nFinished rev-mode jacvec product\n\n")

    def _to_numpy_array_postprocessing(
        self, om_vector: OMVector, np_postproc_array: np.ndarray
    ):
        name_suffix = "_final"
        for quantity, metadata in self._quantity_metadata.items():
            if metadata["type"] == "postprocessing":
                start = metadata["numpy_postproc_start_index"]
                end = metadata["numpy_postproc_end_index"]
                np_postproc_array[start:end] = om_vector[
                    quantity + name_suffix
                ].flatten()

    def _get_functional_contribution_from_om_output_vec(self, d_outputs: OMVector):
        for quantity in self._functional_quantities:
            start = self._quantity_metadata[quantity]["numpy_functional_start_index"]
            end = self._quantity_metadata[quantity]["numpy_functional_end_index"]
            self._functional_part_perturbations[start:end] = d_outputs[
                quantity + "_functional"
            ]

    def _add_functional_perturbations_to_state_perturbations(
        self, step, postprocessor_linearization_args=None
    ):
        if postprocessor_linearization_args is None:
            postprocessor_linearization_args = {}
        functional_coefficients: FunctionalCoefficients = self.options[
            "functional_coefficients"
        ]
        postprocessing_functional_perturbations = np.zeros(self._numpy_postproc_size)
        if self._functional_quantities:
            if self.comm.rank == 0:
                print("Starting computation of functional contribution.")
            for quantity in self._functional_quantities:
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
                    ) * self._functional_part_perturbations[
                        start_functional:end_functional
                    ]
                elif self._quantity_metadata[quantity]["type"] == "postprocessing":
                    start = self._quantity_metadata[quantity][
                        "numpy_postproc_start_index"
                    ]
                    end = self._quantity_metadata[quantity]["numpy_postproc_end_index"]
                    postprocessing_functional_perturbations[start:end] += (
                        functional_coefficients.get_coefficient(step, quantity)
                        * self._functional_part_perturbations[
                            start_functional:end_functional
                        ]
                    )
            if self.comm.rank == 0:
                print("Finished computation of functional contribution.")
        if self.options["postprocessing_problem"]:
            if self.comm.rank == 0:
                print("Starting postprocessing.")
            self._serialized_state_perturbations += (
                self._postprocessor.postprocess_jacvec_transposed(
                    postprocessing_functional_perturbations,
                    **postprocessor_linearization_args,
                )
            )
            if self.comm.rank == 0:
                print("Finished postprocessing.")

    def _run_step_jacvec_rev(
        self, step, serialized_state, serialized_state_perturbations
    ):
        if self.comm.rank == 0:
            print(f"\nStarting step {step} of rev-mode jacvec product.\n")
        self._serialized_state = serialized_state
        self._serialized_state_perturbations = serialized_state_perturbations
        new_serialized_state_perturbations = serialized_state_perturbations.copy()
        initial_time = self.options["integration_control"].initial_time
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        self.options["integration_control"].step = step
        time = initial_time + (step - 1) * delta_t
        self.options["integration_control"].step_time_old = time
        self.options["integration_control"].step_time_new = time + delta_t

        inputs_cache = {}
        outputs_cache = {}
        # forward iteration
        for stage in range(butcher_tableau.number_of_stages()):
            prob_inputs, prob_outputs, _ = self.options[
                "time_stage_problem"
            ].model.get_nonlinear_vectors()
            if self.comm.rank == 0:
                print(
                    f"Starting stage {stage} of the fwd iteration of rev-mode jvp in\n"
                    f"step {step}."
                )
            self._stage_computation(stage)
            inputs_cache[stage] = prob_inputs.asarray(copy=True)
            outputs_cache[stage] = prob_outputs.asarray(copy=True)
            if self.comm.rank == 0:
                print(
                    f"Finished stage {stage} of the fwd iteration of rev-mode jvp in\n"
                    f"step {step}."
                )
        # backward iteration

        for stage in reversed(range(butcher_tableau.number_of_stages())):
            if self.comm.rank == 0:
                print(
                    f"Starting stage {stage} of the rev iteration of rev-mode jvp in\n"
                    f"step {step}."
                )
            linearization_args = {
                "inputs": inputs_cache[stage],
                "outputs": outputs_cache[stage],
            }
            joined_perturbations = self._runge_kutta_scheme.join_perturbations(
                stage,
                self._serialized_state_perturbations,
                self._stage_perturbations_cache,
            )
            (
                wrt_old_state,
                self._stage_perturbations_cache[stage, :],
            ) = self._runge_kutta_scheme.compute_stage_transposed_jacvec(
                stage, delta_t, time, joined_perturbations, **linearization_args
            )
            new_serialized_state_perturbations += delta_t * wrt_old_state
            if self.comm.rank == 0:
                print(
                    f"Finished stage {stage} of the rev iteration of reve-mode jvp in\n"
                    f"step {step}."
                )
        self._serialized_state_perturbations = new_serialized_state_perturbations
        if self.options["postprocessing_problem"] is not None:
            self._postprocessor.postprocess(self._serialized_state)

        self._add_functional_perturbations_to_state_perturbations(step - 1)

        if self.comm.rank == 0:
            print(f"\nFinishing step {step} of rev-mode jacvec product.\n")
        return self._serialized_state_perturbations
