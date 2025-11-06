# pylint: disable=missing-module-docstring, protected-access

from __future__ import annotations
from copy import deepcopy

import numpy as np
import openmdao.api as om
from openmdao.vectors.vector import Vector as OMVector

from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODEResultState
from rkopenmdao.error_measurer import ErrorMeasurer, SimpleErrorMeasurer
from rkopenmdao.time_discretization.stage_ordered_runge_kutta_discretization import (
    StageOrderedRungeKuttaDiscretization,
    StageOrderedEmbeddedRungeKuttaDiscretization,
)
from rkopenmdao.time_discretization.runge_kutta_discretization_state import (
    RungeKuttaDiscretizationState,
    EmbeddedRungeKuttaDiscretizationState,
)
from rkopenmdao.time_integration_state import TimeIntegrationState

from .butcher_tableau import ButcherTableau
from .discretized_ode.openmdao_ode import OpenMDAOODE
from .integration_control import IntegrationControl
from .error_controller import ErrorController
from .error_controllers import integral

from .checkpoint_interface.checkpoint_interface import CheckpointInterface
from .checkpoint_interface.no_checkpointer import NoCheckpointer

from .file_writer import FileWriterInterface, Hdf5FileWriter


class RungeKuttaIntegrator(om.ExplicitComponent):
    """Outer component for solving time-dependent problems with explicit or diagonally
    implicit Runge-Kutta schemes. One stage of the scheme is modelled by an inner
    OpenMDAO-problem.
    OpenMDAO inputs: - initial values of the quantities for the time integration
    OpenMDAO output: - final values of the quantities for the time integration
    """

    _of_vars: list
    _wrt_vars: list

    _ode: OpenMDAOODE | None

    _runge_kutta_discretization: StageOrderedRungeKuttaDiscretization | None
    _time_integration_state: TimeIntegrationState | None
    _time_integration_state_perturbations: TimeIntegrationState | None

    _cached_input: RungeKuttaDiscretizationState | None

    _error_controller: ErrorController | None
    _error_measurer: ErrorMeasurer | None

    _disable_write_out: bool
    _file_writer: FileWriterInterface | None

    _checkpointer: CheckpointInterface | None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._of_vars = []
        self._wrt_vars = []

        self._runge_kutta_discretization = None
        self._time_integration_state = None
        self._time_integration_state_perturbations = None

        self._cached_input = None

        self._error_controller = None

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
            "error_measurer",
            default=SimpleErrorMeasurer(),
            types=ErrorMeasurer,
            desc="""""",
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
    def check_error_controller(name, value):
        """Checks whether the passed error_controller type for the options is an actual
        a callable and has the right parameters"""
        # pylint: disable=unused-argument
        # OpenMDAO needs that specific interface, even if we don't need it fully.
        for method in value:
            if not callable(method):
                raise TypeError(f"{method} is not a callable.")
            temp = method(p=1)
            if not isinstance(temp, ErrorController):
                raise TypeError(
                    f"{method} does not instantiate an instance of ErrorController"
                )

    def setup(self):
        if self.comm.rank == 0:
            print("\n" + "=" * 33 + " setup starts " + "=" * 33 + "\n")
        self._setup_ode()
        self._setup_wrt_and_of_vars()
        self._add_inputs_and_outputs()
        self._setup_runge_kutta_discretization()
        self._setup_error_controller()
        self._setup_integration_states()
        self._configure_write_out()
        self._setup_checkpointing()
        if self.comm.rank == 0:
            print("\n" + "=" * 34 + " setup ends " + "=" * 34 + "\n")

    def _setup_ode(self):
        self.options["time_stage_problem"].setup()
        self.options["time_stage_problem"].final_setup()
        self._ode = OpenMDAOODE(
            self.options["time_stage_problem"],
            self.options["integration_control"],
            self.options["time_integration_quantities"],
            self.options["time_independent_input_quantities"],
        )

    def _add_inputs_and_outputs(self):
        self._add_time_integration_inputs_and_outputs()
        self._add_time_independent_inputs()

    def _add_time_integration_inputs_and_outputs(self):
        time_stage_problem: om.Problem = self.options["time_stage_problem"]
        for (
            quantity
        ) in self._ode.time_integration_metadata.time_integration_quantity_list:
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
        ) in self._ode.time_integration_metadata.time_independent_input_quantity_list:
            self.add_input(
                quantity.name,
                shape=quantity.array_metadata.shape,
                val=time_stage_problem.get_val(
                    quantity.translation_metadata.time_independent_input_var
                ),
                distributed=quantity.array_metadata.distributed,
            )

    def _setup_wrt_and_of_vars(self):
        self._add_time_integration_derivative_info()
        self._add_time_independent_variable_derivative_info()

    def _add_time_integration_derivative_info(self):
        for (
            quantity
        ) in self._ode.time_integration_metadata.time_integration_quantity_list:
            self._of_vars.append(quantity.translation_metadata.stage_output_var)
            if quantity.translation_metadata.step_input_var is not None:
                self._wrt_vars.append(quantity.translation_metadata.step_input_var)
                self._wrt_vars.append(
                    quantity.translation_metadata.accumulated_stage_var
                )

    def _add_time_independent_variable_derivative_info(self):
        for (
            quantity
        ) in self._ode.time_integration_metadata.time_independent_input_quantity_list:
            self._wrt_vars.append(
                quantity.translation_metadata.time_independent_input_var
            )

    def _setup_runge_kutta_discretization(self):
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        if butcher_tableau.is_embedded:
            self._runge_kutta_discretization = (
                StageOrderedEmbeddedRungeKuttaDiscretization(butcher_tableau)
            )
        else:
            self._runge_kutta_discretization = StageOrderedRungeKuttaDiscretization(
                butcher_tableau,
            )
        if self.comm.rank == 0:
            print(f"\n{self._runge_kutta_discretization.butcher_tableau}\n")

    def _setup_error_controller(self):
        self._error_measurer = self.options["error_measurer"]
        p = self.options["butcher_tableau"].min_p_order()
        self._error_controller = None
        # Sets initial delta_t to be at boundary if given wrongly by mistake
        for controller in self.options["error_controller"]:
            self._error_controller = controller(
                p=p,
                **self.options["error_controller_options"],
                base=self._error_controller,
            )
        if self.comm.rank == 0:
            print(f"\n{self._error_controller}")

    def _setup_integration_states(self):
        self._time_integration_state = TimeIntegrationState(
            self._runge_kutta_discretization.create_empty_discretization_state(
                self._ode
            ),
            np.array([self.options["integration_control"].delta_t]),
            # FIXME: The error controller should report how far back time steps and
            # error estimates need to be provided, currently this is hardcoded to two
            # for both.
            np.zeros(2),
            np.full(2, self._error_controller.tol),
        )
        self._time_integration_state_perturbations = TimeIntegrationState(
            self._runge_kutta_discretization.create_empty_discretization_state(
                self._ode
            ),
            np.zeros(0),
            np.zeros(0),  #  Currently don't need histories in derivative values
            np.zeros(0),
        )

    def _configure_write_out(self):
        self._disable_write_out = self.options["write_out_distance"] == 0
        if not self._disable_write_out:
            self._file_writer = self.options["file_writing_implementation"](
                self.options["write_file"],
                self._ode.time_integration_metadata,
                self.comm,
            )
            self._write_out(
                0,
                self.options["integration_control"].initial_time,
                self._time_integration_state.discretization_state.final_state,
            )

    def _setup_checkpointing(self):
        self._checkpointer = self.options["checkpointing_type"](
            integration_control=self.options["integration_control"],
            run_step_func=self._run_step,
            run_step_jacvec_rev_func=self._run_step_jacvec_rev,
            state=self._time_integration_state,
            state_perturbation=self._time_integration_state_perturbations,
            **self.options["checkpoint_options"],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.comm.rank == 0:
            print("\n" + "=" * 32 + " compute starts " + "=" * 32 + "\n\n")
        self._compute_preparation_phase(inputs, self._time_integration_state)
        self._checkpointer.create_checkpointer()
        self._time_integration_state.set(
            self._checkpointer.iterate_forward(self._time_integration_state)
        )
        self._from_numpy_array_time_integration(
            self._time_integration_state.discretization_state.final_state, outputs
        )
        if self.comm.rank == 0:
            print("\n" + "=" * 33 + " compute ends " + "=" * 33 + "\n")

    def _compute_preparation_phase(
        self, inputs: OMVector, integration_state: TimeIntegrationState
    ):
        self.options["integration_control"].reset()
        self._to_numpy_array_time_integration(
            inputs, integration_state.discretization_state.final_state
        )
        self._to_numpy_array_independent_inputs(
            inputs, integration_state.discretization_state.independent_inputs
        )
        integration_state.discretization_state.final_time[0] = self.options[
            "integration_control"
        ].initial_time
        self._cached_input = deepcopy(integration_state.discretization_state)

        self._reset_error_control(integration_state)
        integration_state.discretization_state.linearization_points.fill(0.0)

    def _to_numpy_array_time_integration(
        self, om_vector: OMVector, np_array: np.ndarray
    ):
        if om_vector._kind == "input":
            name_suffix = "_initial"
        else:
            name_suffix = "_final"
        for (
            quantity
        ) in self._ode.time_integration_metadata.time_integration_quantity_list:
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
            ) in (
                self._ode.time_integration_metadata.time_independent_input_quantity_list
            ):
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                np_array[start:end] = om_vector[quantity.name].flatten()

    def _write_out(
        self,
        step: int,
        time: float,
        serialized_state: np.ndarray,
        error_norm: float = None,
    ):
        if self._file_writer is not None:
            self._file_writer.write_step(
                step,
                time,
                serialized_state,
                error_norm,
            )

    def _from_numpy_array_time_integration(
        self, np_array: np.ndarray, om_vector: OMVector
    ):
        if om_vector._kind == "input":
            name_suffix = "_initial"
        else:
            name_suffix = "_final"
        for (
            quantity
        ) in self._ode.time_integration_metadata.time_integration_quantity_list:
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
            ) in (
                self._ode.time_integration_metadata.time_independent_input_quantity_list
            ):
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                om_vector[quantity.name] = np_array[start:end].reshape(
                    quantity.array_metadata.shape
                )

    def _run_step(
        self, time_integration_state: TimeIntegrationState
    ) -> TimeIntegrationState:
        step = self.options["integration_control"].step
        if self.comm.rank == 0:
            print(
                f"\nStarting step <{step}> of compute "
                f"at {self.options['integration_control'].step_time:.5f}.\n"
            )

        time_integration_state.set(self._iterate_on_step(time_integration_state))
        self.options["integration_control"].step_time = (
            time_integration_state.discretization_state.final_time[0]
        )
        self._run_step_write_out_phase(time_integration_state)
        if self.comm.rank == 0:
            print(f"\nFinishing step <{step}> of compute.\n")
        return time_integration_state

    def _reset_error_control(self, integration_state: TimeIntegrationState):
        if (
            self.options["integration_control"].step
            == self.options["integration_control"].initial_step
        ):
            integration_state.step_size_history.fill(0.0)
            integration_state.error_history.fill(self._error_controller.tol)
            integration_state.step_size_suggestion[0] = self.options[
                "integration_control"
            ].initial_delta_t

    def _iterate_on_step(
        self, time_integration_state: TimeIntegrationState, i: int = 0
    ):
        """Iterates on the a step until a time step that
        complies with the norm's tolerance"""
        if self.options["adaptive_time_stepping"]:
            if self.comm.rank == 0:
                print(
                    f"Start Iteration {i}: Trying with {time_integration_state.step_size_suggestion[0]}"
                )

        temp_discretization_state = deepcopy(
            time_integration_state.discretization_state
        )
        temp_discretization_state = self._runge_kutta_discretization.compute_step(
            self._ode,
            temp_discretization_state,
            time_integration_state.step_size_suggestion[0],
        )
        if self.options["adaptive_time_stepping"]:
            error_measure = self._get_error_measure(temp_discretization_state)
            remaining_time = self.options["integration_control"].remaining_time(
                temp_discretization_state.final_time
            )
            error_controller_status = self._error_controller(
                error_measure,
                time_integration_state.step_size_suggestion[0],
                remaining_time,
                time_integration_state.error_history,
                time_integration_state.step_size_history,
            )
            if self.comm.rank == 0:
                nl = "\n"
                print(
                    f"""End Iteration {i}: {
                    self.options['integration_control'].delta_t} {f'succeeded. {nl}'
                    'Estimation for next step is:' 
                    if error_controller_status.accepted else f'failed. {nl}' 
                    'retrying with:'
                    } {error_controller_status.step_size_suggestion}
                    """
                )

            if error_controller_status.accepted:
                new_step_size_history = np.roll(
                    time_integration_state.step_size_history, 1
                )
                new_step_size_history[0] = time_integration_state.step_size_suggestion[
                    0
                ]
                new_error_history = np.roll(time_integration_state.error_history, 1)
                new_error_history[0] = error_measure
                time_integration_state.discretization_state.set(
                    temp_discretization_state
                )
                time_integration_state.step_size_suggestion[0] = (
                    error_controller_status.step_size_suggestion
                )
                time_integration_state.step_size_history[:] = new_step_size_history
                time_integration_state.error_history[:] = new_error_history
            else:
                time_integration_state.step_size_suggestion[0] = (
                    error_controller_status.step_size_suggestion
                )
                time_integration_state.set(
                    self._iterate_on_step(
                        time_integration_state,
                        i + 1,
                    )
                )
        else:
            time_integration_state.discretization_state.set(temp_discretization_state)
            np.roll(time_integration_state.step_size_history, 1)
            time_integration_state.step_size_history[0] = (
                time_integration_state.step_size_suggestion[0]
            )
        return time_integration_state

    def _get_error_measure(
        self, discretization_state: EmbeddedRungeKuttaDiscretizationState
    ):
        state_error_estimate = DiscretizedODEResultState(
            np.zeros(0), discretization_state.error_estimate, np.zeros(0)
        )
        state = DiscretizedODEResultState(
            np.zeros(0), discretization_state.final_state, np.zeros(0)
        )

        return self._error_measurer.get_measure(state_error_estimate, state, self._ode)

    def _run_step_write_out_phase(self, integration_state: TimeIntegrationState):
        step = self.options["integration_control"].step
        time = self.options["integration_control"].step_time
        if not self._disable_write_out and (
            step % self.options["write_out_distance"] == 0
            or self.options["integration_control"].is_last_time_step()
        ):
            self._write_out(
                step,
                time,
                integration_state.discretization_state.final_state,
                (
                    integration_state.error_history[0]
                    if self.options["butcher_tableau"].is_embedded
                    else None
                ),
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
        self._compute_jacvec_fwd_preparation_phase(
            inputs,
            d_inputs,
            self._time_integration_state,
            self._time_integration_state_perturbations,
        )
        self._time_integration_state_perturbations.set(
            self._compute_jacvec_fwd_run_steps(
                self._time_integration_state, self._time_integration_state_perturbations
            )
        )
        self._from_numpy_array_time_integration(
            self._time_integration_state_perturbations.discretization_state.final_state,
            d_outputs,
        )
        if self.comm.rank == 0:
            print(
                "\n\n" + "=" * 25 + " fwd-mode jacvec product ends " + "=" * 25 + "\n"
            )

    def _compute_jacvec_fwd_preparation_phase(
        self,
        inputs: OMVector,
        d_inputs: OMVector,
        integration_state: TimeIntegrationState,
        integration_state_perturbations: TimeIntegrationState,
    ):
        self.options["integration_control"].reset()
        self._to_numpy_array_time_integration(
            inputs, integration_state.discretization_state.final_state
        )
        self._to_numpy_array_time_integration(
            d_inputs, integration_state_perturbations.discretization_state.final_state
        )
        self._to_numpy_array_independent_inputs(
            inputs, integration_state.discretization_state.independent_inputs
        )
        self._to_numpy_array_independent_inputs(
            d_inputs,
            integration_state_perturbations.discretization_state.independent_inputs,
        )
        integration_state.discretization_state.final_time[0] = self.options[
            "integration_control"
        ].initial_time

        self._reset_error_control(integration_state)

    def _compute_jacvec_fwd_run_steps(
        self,
        time_integration_state: TimeIntegrationState,
        time_integration_state_perturbations: TimeIntegrationState,
    ) -> TimeIntegrationState:
        while self.options["integration_control"].termination_condition_status():
            time_integration_state, time_integration_state_perturbations = (
                self._run_step_jacvec_fwd(
                    time_integration_state, time_integration_state_perturbations
                )
            )
        return time_integration_state_perturbations

    def _run_step_jacvec_fwd(
        self,
        time_integration_state: TimeIntegrationState,
        time_integration_state_perturbation: TimeIntegrationState,
    ) -> TimeIntegrationState:
        if self.comm.rank == 0:
            print(
                f"\nStarting step {self.options['integration_control'].step} "
                f"of fwd-mode jacvec product."
            )
        time_integration_state.set(self._iterate_on_step(time_integration_state))
        self.options["integration_control"].step_time = (
            time_integration_state.discretization_state.final_time[0]
        )
        time_integration_state_perturbation.discretization_state.set(
            self._runge_kutta_discretization.compute_step_derivative(
                self._ode,
                time_integration_state.discretization_state,
                time_integration_state_perturbation.discretization_state,
                time_integration_state.step_size_history[0],
            )
        )
        if self.comm.rank == 0:
            print(
                f"\nFinished step {self.options['integration_control'].step} "
                f"of fwd-mode jacvec product.\n"
            )

        return time_integration_state, time_integration_state_perturbation

    def _compute_jacvec_product_rev(self, inputs, d_inputs, d_outputs):
        if self.comm.rank == 0:
            print(
                "\n" + "=" * 24 + " rev-mode jacvec product starts " + "=" * 24 + "\n\n"
            )
        self._compute_jacvec_rev_preparation_phase(
            inputs,
            d_outputs,
            self._time_integration_state_perturbations,
        )
        self._time_integration_state_perturbations.set(
            self._checkpointer.iterate_reverse(
                self._time_integration_state_perturbations
            )
        )

        self._from_numpy_array_time_integration(
            self._time_integration_state_perturbations.discretization_state.final_state,
            d_inputs,
        )
        self._from_numpy_array_independent_input(
            self._time_integration_state_perturbations.discretization_state.independent_inputs,
            d_inputs,
        )
        self._configure_write_out()

        # using the checkpoints invalidates the checkpointer in general
        self._cached_input = None

        if self.comm.rank == 0:
            print(
                "\n\n" + "=" * 25 + " fwd-mode jacvec product ends " + "=" * 25 + "\n"
            )

    def _compute_jacvec_rev_preparation_phase(
        self,
        inputs: OMVector,
        d_outputs: OMVector,
        integration_state_perturbations: TimeIntegrationState,
    ):
        self._disable_write_out = True
        temporary_integration_state = TimeIntegrationState(
            self._runge_kutta_discretization.create_empty_discretization_state(
                self._ode
            ),
            np.array([self.options["integration_control"].delta_t]),
            np.zeros(0),
            np.zeros(0),
        )
        self._to_numpy_array_time_integration(
            inputs, temporary_integration_state.discretization_state.final_state
        )
        self._to_numpy_array_independent_inputs(
            inputs, temporary_integration_state.discretization_state.independent_inputs
        )
        temporary_integration_state.discretization_state.final_time[0] = self.options[
            "integration_control"
        ].initial_time
        if self._cached_input is None or not (
            np.array_equal(
                self._cached_input.final_state,
                temporary_integration_state.discretization_state.final_state,
            )
            and np.array_equal(
                self._cached_input.independent_inputs,
                temporary_integration_state.discretization_state.independent_inputs,
            )
        ):
            if self.comm.rank == 0:
                print("Preparing checkpoints by calling compute()")
            outputs = self._vector_class("nonlinear", "output", self)
            self.compute(inputs, outputs)

        self._to_numpy_array_time_integration(
            d_outputs, integration_state_perturbations.discretization_state.final_state
        )
        integration_state_perturbations.discretization_state.independent_inputs.fill(
            0.0
        )

    def _run_step_jacvec_rev(
        self,
        time_integation_state: TimeIntegrationState,
        time_integration_state_perturbations: TimeIntegrationState,
    ) -> TimeIntegrationState:
        step = self.options["integration_control"].step
        if self.comm.rank == 0:
            print(f"\nStarting step <{step}> of rev-mode jacvec product.")
        time_integration_state_perturbations.discretization_state.set(
            self._runge_kutta_discretization.compute_step_adjoint_derivative(
                self._ode,
                time_integation_state.discretization_state,
                time_integration_state_perturbations.discretization_state,
                time_integation_state.step_size_history[0],
            )
        )
        if self.comm.rank == 0:
            print(f"\nFinishing step <{step}> of rev-mode jacvec product.")

        return time_integration_state_perturbations
