# pylint: disable=missing-module-docstring, protected-access

from __future__ import annotations
from itertools import chain

import numpy as np
import openmdao.api as om
from openmdao.vectors.vector import Vector as OMVector

from .butcher_tableau import ButcherTableau
from .discretized_ode.openmdao_ode import OpenMDAOODE
from .functional_coefficients import FunctionalCoefficients, EmptyFunctionalCoefficients
from .integration_control import IntegrationControl
from .runge_kutta_scheme import RungeKuttaScheme
from .error_controller import ErrorController
from .error_controllers import integral
from .error_estimator import ErrorEstimator, SimpleErrorEstimator
from .postprocessing import Postprocessor
from .postprocessing_computation_functors import (
    PostprocessingProblemComputeFunctor,
    PostprocessingProblemComputeJacvecFunctor,
    PostprocessingProblemComputeTransposeJacvecFunctor,
)
from .checkpoint_interface.checkpoint_interface import CheckpointInterface
from .checkpoint_interface.no_checkpointer import NoCheckpointer
from .metadata_extractor import (
    add_postprocessing_metadata,
    add_functional_metadata,
    Quantity,
    TimeIntegrationMetadata,
)

from .file_writer import FileWriterInterface, Hdf5FileWriter


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

    _functional_quantities: list
    _of_vars: list
    _wrt_vars: list
    _of_vars_postproc: list
    _wrt_vars_postproc: list

    _ode: OpenMDAOODE

    _runge_kutta_scheme: RungeKuttaScheme | None
    _serialized_state: np.ndarray | None
    _accumulated_stages: np.ndarray | None
    _stage_cache: np.ndarray | None
    _serialized_state_perturbations: np.ndarray | None
    _serialized_state_perturbations_from_functional: np.ndarray | None
    _accumulated_stage_perturbations: np.ndarray | None
    _stage_perturbations_cache: np.ndarray | None

    _error_estimator: ErrorEstimator | None
    _error_controller: ErrorController | None

    _independent_input_array: np.ndarray | None
    _independent_input_perturbations: np.ndarray | None

    _postprocessor: Postprocessor | None
    _postprocessing_state: np.ndarray | None
    _postprocessing_state_perturbations: np.ndarray | None

    _functional_part: np.ndarray | None
    _functional_part_perturbations: np.ndarray | None

    _time_integration_metadata: TimeIntegrationMetadata | None

    _cached_input: tuple[np.ndarray, np.ndarray]

    _disable_write_out: bool
    _file_writer: FileWriterInterface | None

    _checkpointer: CheckpointInterface | None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
        self._error_controller = None
        self._independent_input_array = None
        self._independent_input_perturbations = None

        self._postprocessor = None
        self._postprocessing_state = None
        self._postprocessing_state_perturbations = None

        self._functional_part = None
        self._functional_part_perturbations = None

        self._time_integration_metadata = None

        self._cached_input = (np.full(0, np.nan), np.full(0, np.nan))

        self._disable_write_out = False
        self._file_writer = None

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
            "file_writing_implementation",
            default=Hdf5FileWriter,
            check_valid=self.check_file_writer_type,
            desc="Defines what kind of file writing should be used.",
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
                [*tag*, "stage_output_var"] (in case where there is no dependence on
                the previous state of the quantity in the ODE)
                3. If you use postprocessing, and if the postprocessing should use the
                quantity, than there needs to be an input with the tags
                [*tag*, "postproc_input_var", ...] somewhere in
                the postprocessing_problem.
                """,
        )
        self.options.declare(
            "time_independent_input_quantities",
            types=list,
            default=[],
            desc="""List of tags used to describe quantities of independent inputs in
            the time_stage_problem. These tags fulfil the following roles:
                1. Based on these tags, an input will be added to the
                RungeKuttaIntegrator. Per tag there will be the input *tag*.
                2. Inputs with these tags need to be present in the time_stage_problem.
                In detail, per tag there needs to be exactly one input tagged with
                [*tag*, 'time_independent_input_var'].
                """,
        )

        self.options.declare(
            "postprocessing_quantities",
            types=list,
            default=[],
            desc="""List of tags used to describe the quantities computed by the
            postprocessing. These tags fulfil multiple roles:
                1. Based on these tags, outputs are added to the RungeKuttaIntegrator.
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
            desc="""A FunctionalCoefficients object that can return a list of
            quantities (which needs to be a subset of the time integration and
            postprocessing ones) over which linear combinations are evaluated,
            as well as a coefficient given a time step and a quantity. Per
            quantity returned by the object, an output is added, named
            *quantity*_functional.""",
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
            desc="""Additional options passed to the checkpointer. Valid options depend
            on the used checkpointing_type""",
        )

        self.options.declare(
            "error_estimator_type",
            check_valid=self.check_error_estimator_type,
            default=SimpleErrorEstimator,
            desc="""Type of ErrorEstimator used. Must be a subclass of
            ErrorEstimator""",
        )

        self.options.declare(
            "error_estimator_options",
            types=dict,
            default={},
            desc="""Additional options passed to the ErrorController.
            Valid options depend on the error_estimator_type""",
        )

        self.options.declare(
            "error_controller",
            default=[integral],
            check_valid=self.check_error_controller,
            desc="""List of Error controllers for adaptive time stepping of
            the Runge-Kutta Scheme, where the first is the outer most controller
            and the other decorate on it.""",
        )

        self.options.declare(
            "error_controller_options",
            types=dict,
            default={},
            desc="""Options for the error controller class. Valid options depend of
            the error controller type.""",
        )

        self.options.declare(
            "adaptive_time_stepping",
            types=bool,
            default=False,
            desc="A flag that indicates whether to use the adaptive scheme.",
        )

    @staticmethod
    def check_checkpointing_type(name, value):
        """Checks whether the passed checkpointing type for the options is an actual
        subclass of CheckpointInterface"""
        # pylint: disable=unused-argument
        # OpenMDAO needs that specific interface, even if we don't need it fully.
        if not issubclass(value, CheckpointInterface):
            raise TypeError(f"{value} is not a subclass of CheckpointInterface")

    @staticmethod
    def check_file_writer_type(name, value):
        """Checks whether the passed file writing type for the options is an actual
        subclass of FileWriterInterface"""
        # pylint: disable=unused-argument
        # OpenMDAO needs that specific interface, even if we don't need it fully.
        if not issubclass(value, FileWriterInterface):
            raise TypeError(f"{value} is not a subclass of FileWriterInterface")

    @staticmethod
    def check_error_estimator_type(name, value):
        """Checks whether the passed checkpointing type for the options is an actual
        subclass of CheckpointInterface"""
        # pylint: disable=unused-argument
        # OpenMDAO needs that specific interface, even if we don't need it fully.
        if not issubclass(value, ErrorEstimator):
            raise TypeError(f"{value} is not a subclass of CheckpointInterface")

    @staticmethod
    def check_error_controller(name, value):
        """Checks whether the passed error_controller type for the options is an actual
        a callable and has the right parameters"""
        # pylint: disable=unused-argument
        # OpenMDAO needs that specific interface, even if we don't need it fully.
        for method in value:
            if not callable(method):
                raise TypeError(f"{method} is not a callable.")
            temp = method(p=1, error_estimator=SimpleErrorEstimator)
            if not isinstance(temp, ErrorController):
                raise TypeError(
                    f"{method} does not instantiate an instance of ErrorController"
                )

    def setup(self):
        if self.comm.rank == 0:
            print("\n" + "=" * 33 + " setup starts " + "=" * 33 + "\n")
        self._setup_inner_problems()
        self._setup_variables()
        self._setup_arrays()
        self._setup_error_estimator()
        self._setup_error_controller()
        self._setup_runge_kutta_scheme()
        self._setup_postprocessor()
        self._configure_write_out()
        self._setup_checkpointing()
        if self.comm.rank == 0:
            print("\n" + "=" * 34 + " setup ends " + "=" * 34 + "\n")

    def _setup_error_controller(self):
        p = self.options["butcher_tableau"].min_p_order()
        self._error_controller = None
        # Sets initial delta_t to be at boundary if given wrongly by mistake
        for controller in self.options["error_controller"]:
            self._error_controller = controller(
                p=p,
                **self.options["error_controller_options"],
                error_estimator=self._error_estimator,
                base=self._error_controller,
            )
        if self.comm.rank == 0:
            print(f"\n{self._error_controller}")

    def _setup_inner_problems(self):
        self.options["time_stage_problem"].setup()
        self.options["time_stage_problem"].final_setup()
        if self.options["postprocessing_problem"] is not None:
            self.options["postprocessing_problem"].setup()
            self.options["postprocessing_problem"].final_setup()
        self._ode = OpenMDAOODE(
            self.options["time_stage_problem"],
            self.options["integration_control"],
            self.options["time_integration_quantities"],
            self.options["time_independent_input_quantities"],
        )

    def _setup_variables(self):
        self._functional_quantities = self.options[
            "functional_coefficients"
        ].list_quantities()
        self._time_integration_metadata = self._ode.time_integration_metadata
        if self.options["postprocessing_problem"] is not None:
            add_postprocessing_metadata(
                self.options["postprocessing_problem"],
                self.options["time_integration_quantities"],
                self.options["postprocessing_quantities"],
                self._time_integration_metadata,
            )

        add_functional_metadata(
            self._functional_quantities, self._time_integration_metadata
        )

        self._setup_wrt_and_of_vars()
        self._add_inputs_and_outputs()

    def _add_inputs_and_outputs(self):
        self._add_time_integration_inputs_and_outputs()
        self._add_time_independent_inputs()
        self._add_postprocessing_outputs()
        self._add_functional_outputs()

    def _add_time_integration_inputs_and_outputs(self):
        time_stage_problem: om.Problem = self.options["time_stage_problem"]
        for quantity in self._time_integration_metadata.time_integration_quantity_list:
            if quantity.array_metadata.local:
                self.add_input(
                    quantity.name + "_initial",
                    shape=quantity.array_metadata.shape,
                    val=(
                        time_stage_problem.get_val(
                            quantity.translation_metadata.step_input_var,
                        )
                        if quantity.translation_metadata.step_input_var is not None
                        else np.zeros(quantity.array_metadata.shape)
                    ),
                    distributed=quantity.array_metadata.distributed,
                )
            else:
                self.add_input(
                    quantity.name + "_initial",
                    shape=quantity.array_metadata.shape,
                    distributed=quantity.array_metadata.distributed,
                )
            self.add_output(
                quantity.name + "_final",
                copy_shape=quantity.name + "_initial",
                distributed=quantity.array_metadata.distributed,
            )

    def _add_time_independent_inputs(self):
        time_stage_problem: om.Problem = self.options["time_stage_problem"]
        for (
            quantity
        ) in self._time_integration_metadata.time_independent_input_quantity_list:
            self.add_input(
                quantity.name,
                shape=quantity.array_metadata.shape,
                val=time_stage_problem.get_val(
                    quantity.translation_metadata.time_independent_input_var
                ),
                distributed=quantity.array_metadata.distributed,
            )

    def _add_postprocessing_outputs(self):
        for quantity in self._time_integration_metadata.postprocessing_quantity_list:
            self.add_output(
                quantity.name + "_final",
                shape=quantity.array_metadata.shape,
                distributed=quantity.array_metadata.distributed,
            )

    def _add_functional_outputs(self):
        for quantity in chain(
            self._time_integration_metadata.time_integration_quantity_list,
            self._time_integration_metadata.postprocessing_quantity_list,
        ):
            if quantity.array_metadata.functionally_integrated:
                self.add_output(
                    quantity.name + "_functional",
                    copy_shape=quantity.name + "_final",
                    distributed=quantity.array_metadata.distributed,
                )

    def _setup_wrt_and_of_vars(self):
        self._add_time_integration_derivative_info()
        self._add_time_independent_variable_derivative_info()
        self._add_postprocessing_derivative_info()

    def _add_time_integration_derivative_info(self):
        for quantity in self._time_integration_metadata.time_integration_quantity_list:
            self._of_vars.append(quantity.translation_metadata.stage_output_var)
            if quantity.translation_metadata.step_input_var is not None:
                self._wrt_vars.append(quantity.translation_metadata.step_input_var)
                self._wrt_vars.append(
                    quantity.translation_metadata.accumulated_stage_var
                )
            if quantity.translation_metadata.postproc_input_var is not None:
                self._wrt_vars_postproc.append(
                    quantity.translation_metadata.postproc_input_var
                )

    def _add_time_independent_variable_derivative_info(self):
        for (
            quantity
        ) in self._time_integration_metadata.time_independent_input_quantity_list:
            self._wrt_vars.append(
                quantity.translation_metadata.time_independent_input_var
            )

    def _add_postprocessing_derivative_info(self):
        for quantity in self._time_integration_metadata.postprocessing_quantity_list:
            self._of_vars_postproc.append(
                quantity.translation_metadata.postproc_output_var
            )

    def _setup_arrays(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        self._serialized_state = np.zeros(
            self._time_integration_metadata.time_integration_array_size
        )
        self._accumulated_stages = np.zeros(
            self._time_integration_metadata.time_integration_array_size
        )
        self._stage_cache = np.zeros(
            (
                butcher_tableau.number_of_stages(),
                self._time_integration_metadata.time_integration_array_size,
            )
        )
        self._serialized_state_perturbations = np.zeros(
            self._time_integration_metadata.time_integration_array_size
        )
        self._serialized_state_perturbations_from_functional = np.zeros(
            self._time_integration_metadata.time_integration_array_size
        )
        self._accumulated_stage_perturbations = np.zeros(
            self._time_integration_metadata.time_integration_array_size
        )
        self._stage_perturbations_cache = np.zeros(
            (
                butcher_tableau.number_of_stages(),
                self._time_integration_metadata.time_integration_array_size,
            )
        )
        self._independent_input_array = np.zeros(
            self._time_integration_metadata.time_independent_input_size
        )
        self._independent_input_perturbations = np.zeros(
            self._time_integration_metadata.time_independent_input_size
        )
        self._postprocessing_state = np.zeros(
            self._time_integration_metadata.postprocessing_array_size
        )
        self._postprocessing_state_perturbations = np.zeros(
            self._time_integration_metadata.postprocessing_array_size
        )
        self._functional_part = np.zeros(
            self._time_integration_metadata.functional_array_size
        )
        self._functional_part_perturbations = np.zeros(
            self._time_integration_metadata.functional_array_size
        )

    def _setup_runge_kutta_scheme(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        self._runge_kutta_scheme = RungeKuttaScheme(
            butcher_tableau,
            self._ode,
            self.options["adaptive_time_stepping"],
            self._error_controller,
        )
        if self.comm.rank == 0:
            print(f"\n{self._runge_kutta_scheme.butcher_tableau}\n")

    def _setup_postprocessor(self):
        postprocessing_problem: om.Problem = self.options["postprocessing_problem"]
        if postprocessing_problem is not None:
            self._postprocessor = Postprocessor(
                PostprocessingProblemComputeFunctor(
                    postprocessing_problem,
                    self._time_integration_metadata,
                ),
                PostprocessingProblemComputeJacvecFunctor(
                    postprocessing_problem,
                    self._time_integration_metadata,
                    self._of_vars_postproc,
                    self._wrt_vars_postproc,
                ),
                PostprocessingProblemComputeTransposeJacvecFunctor(
                    postprocessing_problem,
                    self._time_integration_metadata,
                    self._of_vars_postproc,
                    self._wrt_vars_postproc,
                ),
            )

    def _configure_write_out(self):
        self._disable_write_out = self.options["write_out_distance"] == 0
        if not self._disable_write_out:
            self._file_writer = self.options["file_writing_implementation"](
                self.options["write_file"], self._time_integration_metadata, self.comm
            )

    def _setup_checkpointing(self):
        self._checkpointer = self.options["checkpointing_type"](
            array_size=self._time_integration_metadata.time_integration_array_size,
            integration_control=self.options["integration_control"],
            run_step_func=self._run_step,
            run_step_jacvec_rev_func=self._run_step_jacvec_rev,
            **self.options["checkpoint_options"],
        )

    def _setup_error_estimator(self):
        self._error_estimator = self.options["error_estimator_type"](
            **self.options["error_estimator_options"],
            comm=self.comm,
            quantity_metadata=self._time_integration_metadata,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.comm.rank == 0:
            print("\n" + "=" * 32 + " compute starts " + "=" * 32 + "\n\n")
        self._compute_preparation_phase(inputs)
        self._compute_postprocessing_phase()
        self._compute_initial_write_out_phase()
        self._compute_functional_phase()
        self._compute_checkpointing_setup_phase()
        self._checkpointer.iterate_forward(self._serialized_state)
        self._compute_translate_to_om_vector_phase(outputs)
        if self.comm.rank == 0:
            print("\n" + "=" * 33 + " compute ends " + "=" * 33 + "\n")

    def _compute_preparation_phase(self, inputs: OMVector):
        self._to_numpy_array_time_integration(inputs, self._serialized_state)
        self._to_numpy_array_independent_inputs(inputs, self._independent_input_array)
        self._cached_input = (
            self._serialized_state.copy(),
            self._independent_input_array.copy(),
        )
        self.options["integration_control"].reset()
        self._update_error_estimator()

    def _to_numpy_array_time_integration(
        self, om_vector: OMVector, np_array: np.ndarray
    ):
        if om_vector._kind == "input":
            name_suffix = "_initial"
        else:
            name_suffix = "_final"
        for quantity in self._time_integration_metadata.time_integration_quantity_list:
            start = quantity.array_metadata.start_index
            end = quantity.array_metadata.end_index
            np_array[start:end] = om_vector[quantity.name + name_suffix].flatten()

    def _to_numpy_array_independent_inputs(
        self, om_vector: OMVector, np_array: np.ndarray
    ):
        if om_vector._kind != "input":
            raise TypeError(
                "Can't extract independent input data from an output vector"
            )
        else:
            for (
                quantity
            ) in self._time_integration_metadata.time_independent_input_quantity_list:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                np_array[start:end] = om_vector[quantity.name].flatten()

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
                self.options["integration_control"].initial_time,
                self._serialized_state,
                self._postprocessing_state,
            )

    def _write_out(
        self,
        step: int,
        time: float,
        serialized_state: np.ndarray,
        postprocessing_state=None,
    ):
        if self._file_writer is not None:
            if self.options["adaptive_time_stepping"]:
                self._file_writer.write_step(
                    step,
                    time,
                    serialized_state,
                    postprocessing_state,
                    self._error_controller.get_last_step_norm(),
                )
            else:
                self._file_writer.write_step(
                    step,
                    time,
                    serialized_state,
                    postprocessing_state,
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
        contribution = np.zeros(self._time_integration_metadata.functional_array_size)
        self._add_contribution_of_quantities_from_vector(
            contribution,
            timestep,
            self._time_integration_metadata.time_integration_quantity_list,
            serialized_state,
        )
        self._add_contribution_of_quantities_from_vector(
            contribution,
            timestep,
            self._time_integration_metadata.postprocessing_quantity_list,
            postprocessing_state,
        )
        return contribution

    def _add_contribution_of_quantities_from_vector(
        self,
        contribution: np.ndarray,
        step: int,
        quantity_list: list[Quantity],
        vector: np.ndarray,
    ):
        functional_coefficients: FunctionalCoefficients = self.options[
            "functional_coefficients"
        ]
        for quantity in quantity_list:
            start_functional = quantity.array_metadata.functional_start_index
            end_functional = quantity.array_metadata.functional_end_index
            start_quantity = quantity.array_metadata.start_index
            end_quantity = quantity.array_metadata.end_index
            contribution[start_functional:end_functional] = (
                functional_coefficients.get_coefficient(step, quantity.name)
                * vector[start_quantity:end_quantity]
            )

    def _compute_checkpointing_setup_phase(self):
        self._checkpointer.create_checkpointer()

    def _compute_translate_to_om_vector_phase(self, outputs: OMVector):
        self._from_numpy_array_time_integration(self._serialized_state, outputs)
        if self.options["postprocessing_problem"] is not None:
            self._from_numpy_array_postprocessing(self._postprocessing_state, outputs)
        if self._functional_quantities:
            self._add_functional_part_to_om_vec(self._functional_part, outputs)

    def _from_numpy_array_time_integration(
        self, np_array: np.ndarray, om_vector: OMVector
    ):
        if om_vector._kind == "input":
            name_suffix = "_initial"
        else:
            name_suffix = "_final"
        for quantity in self._time_integration_metadata.time_integration_quantity_list:
            start = quantity.array_metadata.start_index
            end = quantity.array_metadata.end_index
            om_vector[quantity.name + name_suffix] = np_array[start:end].reshape(
                quantity.array_metadata.shape
            )

    def _from_numpy_array_independent_input(
        self, np_array: np.ndarray, om_vector: OMVector
    ):
        if om_vector._kind != "input":
            raise TypeError("Can't export independent input data to an output vector")
        else:
            for (
                quantity
            ) in self._time_integration_metadata.time_independent_input_quantity_list:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                om_vector[quantity.name] = np_array[start:end].reshape(
                    quantity.array_metadata.shape
                )

    def _from_numpy_array_postprocessing(
        self, np_postproc_array: np.ndarray, om_vector: OMVector
    ):
        name_suffix = "_final"  # postprocessing variables are only in output, not input
        for quantity in self._time_integration_metadata.postprocessing_quantity_list:
            start = quantity.array_metadata.start_index
            end = quantity.array_metadata.end_index
            om_vector[quantity.name + name_suffix] = np_postproc_array[
                start:end
            ].reshape(quantity.array_metadata.shape)

    def _add_functional_part_to_om_vec(
        self, functional_numpy_array: np.ndarray, om_vector: OMVector
    ):
        for quantity in chain(
            self._time_integration_metadata.time_integration_quantity_list,
            self._time_integration_metadata.postprocessing_quantity_list,
        ):
            if quantity.array_metadata.functionally_integrated:
                start = quantity.array_metadata.functional_start_index
                end = quantity.array_metadata.functional_end_index
                om_vector[quantity.name + "_functional"] = functional_numpy_array[
                    start:end
                ].reshape(quantity.array_metadata.shape)

    def _run_step(self, serialized_state):
        step = self.options["integration_control"].step
        if self.comm.rank == 0:
            print(
                f"\nStarting step <{step}> of compute "
                f"at {self.options['integration_control'].step_time:.5f}.\n"
            )
        self._serialized_state = serialized_state
        self._run_step_time_integration_phase()
        self._update_integration_control_step()
        self._run_step_postprocessing_phase()
        self._run_step_functional_phase()
        self._run_step_write_out_phase()
        if self.comm.rank == 0:
            print(f"\nFinishing step <{step}> of compute.\n")
        return (
            self._serialized_state,
            self._error_controller.local_data.delta_time_steps,
            self._error_controller.local_data.local_error_norms,
        )

    def _update_integration_control_step(self):
        self.options["integration_control"].step_time += self.options[
            "integration_control"
        ].delta_t

    def _update_error_estimator(self):
        if (
            self.options["integration_control"].step
            == self.options["integration_control"].initial_step
        ):
            self._error_controller.reset()
        if self.comm.rank == 0:
            print("" * 29 + "\n| Resetting error estimator |\n" + "-" * 29 + "\n")

    def _iterate_on_step(self, delta_t_suggestion, stage_computation_func, i=0):
        """Iterates on the a step until a time step that
        complies with the norm's tolerance"""
        self.options["integration_control"].delta_t = delta_t_suggestion
        if self.options["adaptive_time_stepping"]:
            if self.comm.rank == 0:
                print(f"Start Iteration {i}: Trying with {delta_t_suggestion}")

        for stage in range(self.options["butcher_tableau"].number_of_stages()):
            stage_computation_func(stage)

        temp_serialized_state, delta_t_suggestion, accepted = (
            self._runge_kutta_scheme.compute_step(
                self.options["integration_control"].delta_t,
                self._serialized_state,
                self._stage_cache,
                self.options["integration_control"].remaining_time(),
            )
        )
        if self.options["adaptive_time_stepping"]:
            if self.comm.rank == 0:
                nl = "\n"
                print(
                    f"""End Iteration {i}: {
                    self.options['integration_control'].delta_t} {f'succeeded. {nl}'
                    'Estimation for next step is:' 
                    if accepted else f'failed. {nl}' 
                    'retrying with:'
                    } {delta_t_suggestion}
                    """
                )

        if not accepted:
            return self._iterate_on_step(
                delta_t_suggestion, stage_computation_func, i + 1
            )
        return temp_serialized_state, delta_t_suggestion

    def _run_step_time_integration_phase(self):
        delta_t_suggestion = self.options["integration_control"].delta_t_suggestion
        temp_serialized_state, delta_t_suggestion = self._iterate_on_step(
            delta_t_suggestion, stage_computation_func=self._stage_computation
        )
        self.options["integration_control"].delta_t_suggestion = (
            delta_t_suggestion  # Suggestion for next timestep
        )
        self._serialized_state = temp_serialized_state

    def _stage_computation(self, stage, out=True):
        delta_t = self.options["integration_control"].delta_t
        time = self.options["integration_control"].step_time
        step = self.options["integration_control"].step
        if self.comm.rank == 0 and out:
            print(f"Starting stage [{stage + 1}] of compute in step <{step}>.")
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
            stage,
            delta_t,
            time,
            self._serialized_state,
            self._accumulated_stages,
            self._independent_input_array,
        )
        if self.comm.rank == 0 and out:
            print(f"Finished stage [{stage+1}] of compute in step <{step}>.")

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
        time = self.options["integration_control"].step_time
        if not self._disable_write_out and (
            step % self.options["write_out_distance"] == 0
            or self.options["integration_control"].is_last_time_step()
        ):
            self._write_out(
                step,
                time,
                self._serialized_state,
                self._postprocessing_state,
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
            print(
                "\n" + "=" * 24 + " fwd-mode jacvec product starts " + "=" * 24 + "\n\n"
            )
        self._compute_jacvec_fwd_preparation_phase(inputs, d_inputs)
        self._compute_jacvec_fwd_postprocessing_phase()
        self._compute_jacvec_fwd_functional_phase()
        self._compute_jacvec_fwd_run_steps()
        self._compute_jacvec_fwd_translate_to_om_vector_phase(d_outputs)
        if self.comm.rank == 0:
            print(
                "\n\n" + "=" * 25 + " fwd-mode jacvec product ends " + "=" * 25 + "\n"
            )

    def _compute_jacvec_fwd_preparation_phase(self, inputs, d_inputs):
        self._to_numpy_array_time_integration(inputs, self._serialized_state)
        self._to_numpy_array_time_integration(
            d_inputs, self._serialized_state_perturbations
        )
        self._to_numpy_array_independent_inputs(inputs, self._independent_input_array)
        self._to_numpy_array_independent_inputs(
            d_inputs, self._independent_input_perturbations
        )
        self.options["integration_control"].reset()
        self._update_error_estimator()

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
        while self.options["integration_control"].termination_condition_status():
            if self.comm.rank == 0:
                print(
                    f"\nStarting step {self.options['integration_control'].step} "
                    f"of fwd-mode jacvec product."
                )
            self._compute_jacvec_fwd_run_steps_time_integration_phase()
            self._compute_jacvec_fwd_run_steps_preparation_phase()
            self._compute_jacvec_fwd_run_steps_postprocessing_phase()
            self._compute_jacvec_fwd_run_steps_functional_phase()
            if self.comm.rank == 0:
                print(
                    f"\nFinished step {self.options['integration_control'].step} "
                    f"of fwd-mode jacvec product.\n"
                )

    def _compute_jacvec_fwd_run_steps_preparation_phase(self):
        self._update_integration_control_step()

    def _compute_stage_perturbations(self, stage):
        step = self.options["integration_control"].step
        time = self.options["integration_control"].step_time
        if self.comm.rank == 0:
            print(
                f"Starting stage [{stage + 1}] of fwd-mode jacvec product "
                f"in step <{step}>."
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
            stage,
            self.options["integration_control"].delta_t,
            time,
            self._serialized_state,
            self._accumulated_stages,
            self._independent_input_array,
        )
        self._stage_perturbations_cache[stage, :] = (
            self._runge_kutta_scheme.compute_stage_jacvec(
                stage,
                self.options["integration_control"].delta_t,
                self._serialized_state_perturbations,
                self._accumulated_stage_perturbations,
                self._independent_input_perturbations,
            )
        )
        if self.comm.rank == 0:
            print(
                f"Finished stage [{stage+1}] of fwd-mode jacvec product "
                f"in step <{step}>."
            )

    def _compute_jacvec_fwd_run_steps_time_integration_phase(self):
        delta_t_suggestion = self.options["integration_control"].delta_t_suggestion
        temp_serialized_state, delta_t_suggestion = self._iterate_on_step(
            delta_t_suggestion, stage_computation_func=self._compute_stage_perturbations
        )
        self.options["integration_control"].delta_t_suggestion = (
            delta_t_suggestion  # Suggestion for next timestep
        )
        self._serialized_state = temp_serialized_state

        self._serialized_state_perturbations = (
            self._runge_kutta_scheme.compute_step_jacvec(
                self.options["integration_control"].delta_t,
                self._serialized_state_perturbations,
                self._stage_perturbations_cache,
            )
        )

    def _compute_jacvec_fwd_run_steps_postprocessing_phase(self):
        self._compute_jacvec_fwd_postprocessing_phase()

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
        self._from_numpy_array_time_integration(
            self._serialized_state_perturbations, d_outputs
        )
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
            print(
                "\n" + "=" * 24 + " rev-mode jacvec product starts " + "=" * 24 + "\n\n"
            )
        self._disable_write_out = True
        self._to_numpy_array_time_integration(inputs, self._serialized_state)
        self._to_numpy_array_independent_inputs(inputs, self._independent_input_array)
        self._to_numpy_array_independent_inputs(
            d_inputs, self._independent_input_perturbations
        )
        if not (
            np.array_equal(self._cached_input[0], self._serialized_state)
            and np.array_equal(self._cached_input[1], self._independent_input_array)
        ):
            if self.comm.rank == 0:
                print("Preparing checkpoints by calling compute()")
            outputs = self._vector_class("nonlinear", "output", self)
            self.compute(inputs, outputs)
        self._to_numpy_array_time_integration(
            d_outputs, self._serialized_state_perturbations
        )
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
                if self.options["adaptive_time_stepping"]:
                    raise Warning(
                        "Functional Quantities doesn't work currently "
                        "with adaptive schemes."
                    )
                print("Finished computation of functional contribution.")
            self._get_functional_contribution_from_om_output_vec(d_outputs)
            self._add_functional_perturbations_to_state_perturbations(
                self.options["integration_control"].num_steps
            )
            if self.comm.rank == 0:
                print("Finished computation of functional contribution.")
        self._checkpointer.iterate_reverse(self._serialized_state_perturbations)

        self._from_numpy_array_time_integration(
            self._serialized_state_perturbations, d_inputs
        )
        self._from_numpy_array_independent_input(
            self._independent_input_perturbations, d_inputs
        )
        self._configure_write_out()

        # using the checkpoints invalidates the checkpointer in general
        self._cached_input = (np.full(0, np.nan), np.full(0, np.nan))

        if self.comm.rank == 0:
            print(
                "\n\n" + "=" * 25 + " fwd-mode jacvec product ends " + "=" * 25 + "\n"
            )

    def _to_numpy_array_postprocessing(
        self, om_vector: OMVector, np_postproc_array: np.ndarray
    ):
        name_suffix = "_final"
        for quantity in self._time_integration_metadata.postprocessing_quantity_list:
            start = quantity.array_metadata.start_index
            end = quantity.array_metadata.end_index
            np_postproc_array[start:end] = om_vector[
                quantity.name + name_suffix
            ].flatten()

    def _get_functional_contribution_from_om_output_vec(self, d_outputs: OMVector):
        for quantity in chain(
            self._time_integration_metadata.time_integration_quantity_list,
            self._time_integration_metadata.postprocessing_quantity_list,
        ):
            if quantity.array_metadata.functionally_integrated:
                start = quantity.array_metadata.functional_start_index
                end = quantity.array_metadata.functional_end_index
                self._functional_part_perturbations[start:end] = d_outputs[
                    quantity.name + "_functional"
                ]

    def _add_functional_perturbations_to_state_perturbations(
        self, step, postprocessor_linearization_args=None
    ):
        if postprocessor_linearization_args is None:
            postprocessor_linearization_args = {}
        functional_coefficients: FunctionalCoefficients = self.options[
            "functional_coefficients"
        ]
        postprocessing_functional_perturbations = np.zeros(
            self._time_integration_metadata.postprocessing_array_size
        )
        for quantity in chain(
            self._time_integration_metadata.time_integration_quantity_list,
            self._time_integration_metadata.postprocessing_quantity_list,
        ):
            if quantity.array_metadata.functionally_integrated:
                if self.comm.rank == 0:
                    print("Starting computation of functional contribution.")
                start_functional = quantity.array_metadata.functional_start_index
                end_functional = quantity.array_metadata.functional_end_index
                if quantity.type == "time_integration":
                    start = quantity.array_metadata.start_index
                    end = quantity.array_metadata.end_index
                    self._serialized_state_perturbations[start:end] += (
                        functional_coefficients.get_coefficient(step, quantity.name)
                    ) * self._functional_part_perturbations[
                        start_functional:end_functional
                    ]
                elif quantity.type == "postprocessing":
                    start = quantity.array_metadata.start_index
                    end = quantity.array_metadata.end_index
                    postprocessing_functional_perturbations[start:end] += (
                        functional_coefficients.get_coefficient(step, quantity.name)
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

    def _run_step_jacvec_rev(self, serialized_state, serialized_state_perturbations):
        """
        Function to get the sensitvities of the rkschemes at the last step of the
        the current checkpointing iteration.
        """
        step = self.options["integration_control"].step
        if self.comm.rank == 0:
            print(f"\nStarting step <{step}> of rev-mode jacvec product.")
        self._serialized_state = serialized_state
        self._serialized_state_perturbations = serialized_state_perturbations
        new_serialized_state_perturbations = serialized_state_perturbations.copy()
        delta_t = self.options["integration_control"].delta_t
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]

        inputs_cache = {}
        outputs_cache = {}
        # forward iteration
        for stage in range(butcher_tableau.number_of_stages()):
            prob_inputs, prob_outputs, _ = self.options[
                "time_stage_problem"
            ].model.get_nonlinear_vectors()
            if self.comm.rank == 0:
                print(
                    f"Starting stage [{stage + 1}] of the fwd iteration of rev-mode "
                    f"jvp in step <{step}>."
                )
            self._stage_computation(stage, out=False)
            inputs_cache[stage] = prob_inputs.asarray(copy=True)
            outputs_cache[stage] = prob_outputs.asarray(copy=True)
            if self.comm.rank == 0:
                print(
                    f"Finished stage [{stage + 1}] of the fwd iteration of rev-mode "
                    f"jvp in step <{step}>."
                )
        # backward iteration

        for stage in reversed(range(butcher_tableau.number_of_stages())):
            if self.comm.rank == 0:
                print(
                    f"Starting stage [{stage + 1}] of the rev iteration of rev-mode "
                    f"jvp in step <{step}>."
                )
            linearization_args = (
                self.options["integration_control"].step_time
                + butcher_tableau.butcher_time_stages[stage] * delta_t,
                inputs_cache[stage],
                outputs_cache[stage],
            )
            joined_perturbations = self._runge_kutta_scheme.join_perturbations(
                stage,
                self._serialized_state_perturbations,
                self._stage_perturbations_cache,
            )
            (
                wrt_old_state,
                self._stage_perturbations_cache[stage, :],
                parameter_perturbation,
            ) = self._runge_kutta_scheme.compute_stage_transposed_jacvec(
                stage,
                delta_t,
                joined_perturbations,
                linearization_args,
            )
            new_serialized_state_perturbations += (
                self.options["integration_control"].delta_t * wrt_old_state
            )
            self._independent_input_perturbations += (
                delta_t
                # * butcher_tableau.butcher_weight_vector[stage]
                * parameter_perturbation
            )
            if self.comm.rank == 0:
                print(
                    f"Finished stage [{stage + 1}] of the rev iteration of rev-mode "
                    f"jvp in step <{step}>."
                )
        self._serialized_state_perturbations = new_serialized_state_perturbations
        if self.options["postprocessing_problem"] is not None:
            self._postprocessor.postprocess(self._serialized_state)

        self._add_functional_perturbations_to_state_perturbations(
            self.options["integration_control"].step - 1
        )

        if self.comm.rank == 0:
            print(f"\nFinishing step <{step}> " f"of rev-mode jacvec product.\n")
        return self._serialized_state_perturbations
