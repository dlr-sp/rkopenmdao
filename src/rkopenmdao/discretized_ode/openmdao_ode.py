"""ODE implementation for an OpenMDAO Problem."""

# pylint: disable=unused-argument

import functools
import inspect
from typing import Union, Callable, Optional

from mpi4py import MPI
import numpy as np
import openmdao.api as om
from openmdao.vectors.vector import Vector

from rkopenmdao.components import UnsteadyComponentMixin
from rkopenmdao.metadata_extractor import (
    TimeIntegrationMetadata,
    TimeIntegrationQuantity,
    extract_time_integration_metadata,
    add_time_independent_input_metadata,
    add_distributivity_information,
)

from rkopenmdao.om_data_exchange import OMDataExchange
from rkopenmdao.discretized_ode.discretized_ode import DiscretizedODE
from rkopenmdao.states import (
    DiscretizedODEInputState,
    DiscretizedODEResultState,
)


class OpenMDAOODE(DiscretizedODE):
    """
    Wraps an OpenMDAO problem into an instance of discretized ODE, handling the
    transfer of data from/to the inner OpenMDAO problem, as well automatically calling
    all necessary model evaluation methods.

    Parameters
    ----------
    time_stage_problem: om.Problem
        OpenMDAO problem to be wrapped into a discretized ODE. Needs to be in a state
        where its final_setup() method has been called already.
    time_integration_quantities: list
        Quantities to be time integrated that are searched for in the inner problem.
    independent_input_quantities: list
        Quantities that act as time independent inputs that are seached for in the inner
        problem.
    norm_exclusions: Optional[list]
        List of excluded quantities during calculation of norms. By default None.
    norm_order: Union[float, str]
        Order used for the norm. By default 2, resulting in the euclidean norm.


    Attributes
    ----------
    time_integration_metadata : TimeIntegrationMetadata
        Metadata containing information about the shape and location of
        quantities related to time integration.
    """

    time_integration_metadata: TimeIntegrationMetadata

    _time_stage_problem: om.Problem
    _om_data_exchange: OMDataExchange
    _om_run_solve_linear: Callable[[str], None]
    _norm_exclusions: list
    _norm_order: Union[float, str]

    def __init__(
        self,
        time_stage_problem: om.Problem,
        time_integration_quantities: list,
        independent_input_quantities: Optional[list] = None,
        norm_exclusions: Optional[list] = None,
        norm_order: Union[float, str] = 2.0,
    ):
        """
        Initializes the discretized ODE by wrapping the passed OpenMDAO problem,
        extracting the metadata for the time integrated and time independent
        quantities, and configuring the options for norm computation.

        Parameters
        ----------
        time_stage_problem: om.Problem
            OpenMDAO problem to be wrapped into a discretized ODE. Needs to be in a
            state where its final_setup() method has been called already.
        time_integration_quantities: list
            Quantities to be time integrated that are searched for in the inner
            problem.
        independent_input_quantities: list
            Quantities that act as time independent inputs that are searched for in
            the inner problem.
        norm_exclusions: list
            List of quantities excluded during calculation of norms.
        norm_order: Union[float, str]
            Order used for the norm.
        """
        self._time_stage_problem = time_stage_problem
        # Create data exchange object to OpenMDAO and overwrite all instances in
        # unsteady components in the model of time_stage_problem with it.
        self._om_data_exchange = OMDataExchange()
        for subsys in self._time_stage_problem.model.system_iter(
            recurse=True, typ=UnsteadyComponentMixin
        ):
            subsys.om_data_exchange = self._om_data_exchange

        self.time_integration_metadata = extract_time_integration_metadata(
            self._time_stage_problem, time_integration_quantities
        )
        if independent_input_quantities:
            add_time_independent_input_metadata(
                self._time_stage_problem,
                independent_input_quantities,
                self.time_integration_metadata,
            )
        add_distributivity_information(
            self._time_stage_problem, self.time_integration_metadata
        )
        if (
            len(
                inspect.signature(
                    self._time_stage_problem.model.run_solve_linear
                ).parameters
            )
            == 1
        ):
            self._om_run_solve_linear = functools.partial(
                self._time_stage_problem.model.run_solve_linear
            )
        else:
            self._om_run_solve_linear = functools.partial(
                self._time_stage_problem.model.run_solve_linear, vec_name=["linear"]
            )
        if norm_exclusions is None:
            self._norm_exclusions = []
        else:
            self._norm_exclusions = norm_exclusions
        self._norm_order = norm_order

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        """
        Computes the update of one time stage of the wrapped ODE by transferring
        `ode_input` into the OpenMDAO problem, running its nonlinear solve, and
        reading the resulting stage update, stage state, and linearization point
        back out.

        Parameters
        ----------
        ode_input: DiscretizedODEInputState
            Input for the calculation of the time stage.
        step_size: float
            Step size of the current time step
        stage_factor: float
            A stage specific factor on the step size

        Returns
        -------
        ode_result: DiscretizedODEResultState
            Result for the calculation of the time stage.
        """
        _, outputs, _ = self._time_stage_problem.model.get_nonlinear_vectors()
        self._input_state_to_om_vector(
            np.array([ode_input.time]),
            ode_input.step_input,
            ode_input.stage_input,
            ode_input.independent_input,
            outputs,
        )
        self._om_data_exchange.step_size = step_size
        self._om_data_exchange.stage_factor = stage_factor
        self._time_stage_problem.model.run_solve_nonlinear()

        stage_update = np.zeros_like(ode_input.step_input)
        stage_state = np.zeros_like(stage_update)
        independent_output = np.zeros(
            0
        )  # Currently not used, needs update in metadata_extractor

        self._om_vector_to_output_state(
            outputs, stage_update, stage_state, independent_output
        )

        linearization_point = self._get_linearization_point()
        return DiscretizedODEResultState(
            stage_update, stage_state, independent_output, linearization_point
        )

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        """
        Computes the matrix-vector product with the jacobian matrix of the stage
        update, the stage state and independent output wrt. step input, stage input,
        independent inputs and time by linearizing the OpenMDAO problem at the
        given linearization point and running its forward linear solve. Step size
        and stage factor are assumed to be constants, so there are no entries
        wrt. them in the jacobian.

        Parameters
        ----------
        ode_input_perturbation: DiscretizedODEInputState
            Input perturbation for the calculation of the derivative of the time
            stage.
        step_size: float
            Step size of the current time step
        stage_factor: float
            Stage specific factor on the step size

        Returns
        -------
        ode_result_perturbation: DiscretizedODEResultState
            Result perturbation for the calculation of the derivative of the time
            stage.
        """
        self._set_linearization_point(ode_input_perturbation.linearization_point)
        self._om_data_exchange.step_size = step_size
        self._om_data_exchange.stage_factor = stage_factor
        self._time_stage_problem.model.run_linearize()
        _, d_outputs, d_residuals = self._time_stage_problem.model.get_linear_vectors()
        d_residuals.asarray()[:] *= 0.0
        self._input_state_to_om_vector(
            np.array([ode_input_perturbation.time]),
            ode_input_perturbation.step_input,
            ode_input_perturbation.stage_input,
            ode_input_perturbation.independent_input,
            d_residuals,
            -1.0,
        )

        self._om_run_solve_linear("fwd")

        stage_update_pert = np.zeros_like(ode_input_perturbation.step_input)
        stage_state_pert = np.zeros_like(stage_update_pert)
        independent_output_pert = np.zeros(
            0
        )  # Currently not used, needs update in metadata_extractor
        self._om_vector_to_output_state(
            d_outputs, stage_update_pert, stage_state_pert, independent_output_pert
        )

        return DiscretizedODEResultState(
            stage_update_pert, stage_state_pert, independent_output_pert
        )

    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        """
        Computes the matrix-vector product with the adjoint jacobian matrix of the
        stage update, the stage state and independent output wrt. step input, stage
        input, independent inputs and time by linearizing the OpenMDAO problem at
        the given linearization point and running its reverse linear solve. Step
        size and stage factor are assumed to be constants, so there are no entries
        wrt. them in the jacobian.

        Parameters
        ----------
        ode_result_perturbation: DiscretizedODEResultState
            Result perturbation for the calculation of the adjoint derivative of the
            time stage.
        step_size: float
            Step size of the current time step
        stage_factor: float
            Stage specific factor on the step size

        Returns
        -------
        ode_input_perturbation: DiscretizedODEInputState
            Input perturbation for the calculation of the adjoint derivative of the
            time stage.
        """
        self._set_linearization_point(ode_result_perturbation.linearization_point)
        self._om_data_exchange.step_size = step_size
        self._om_data_exchange.stage_factor = stage_factor
        self._time_stage_problem.model.run_linearize()
        _, d_outputs, d_residuals = self._time_stage_problem.model.get_linear_vectors()
        d_outputs.asarray()[:] *= 0.0
        self._output_state_to_om_vector(
            ode_result_perturbation.stage_update,
            ode_result_perturbation.stage_state,
            ode_result_perturbation.independent_output,
            d_outputs,
            -1.0,
        )
        self._om_run_solve_linear("rev")
        step_input_pert = np.zeros(
            self.time_integration_metadata.time_integration_array_size
        )
        stage_input_pert = np.zeros(
            self.time_integration_metadata.time_integration_array_size
        )
        independent_input_pert = np.zeros(
            self.time_integration_metadata.time_independent_input_size
        )
        time_pert_arr = np.zeros(1)
        self._om_vector_to_input_state(
            d_residuals,
            time_pert_arr,
            step_input_pert,
            stage_input_pert,
            independent_input_pert,
        )
        return DiscretizedODEInputState(
            step_input_pert, stage_input_pert, independent_input_pert, time_pert_arr[0]
        )

    def get_state_size(self) -> int:
        """Returns the size of the state vector to be time integrated."""
        return self.time_integration_metadata.time_integration_array_size

    def get_independent_input_size(self) -> int:
        """Returns the size of the time independent input vector."""
        return self.time_integration_metadata.time_independent_input_size

    def get_independent_output_size(self) -> int:
        """Returns the size of the independent output vector, currently not
        implemented and thus zero."""
        return 0  # Not implemented yet

    def get_linearization_point_size(self) -> int:
        """Returns the combined size of the input and output vectors of the wrapped
        OpenMDAO problem."""
        inputs, outputs, _ = self._time_stage_problem.model.get_nonlinear_vectors()
        return inputs.asarray().size + outputs.asarray().size

    def _get_linearization_point(self) -> np.ndarray:
        """
        Serializes the current input and output vectors of the wrapped OpenMDAO
        problem into a single array representing the linearization point.

        Returns
        -------
        linearization_point: np.ndarray
            The input and output data of the OpenMDAO problem as one array.
        """
        inputs, outputs, _ = self._time_stage_problem.model.get_nonlinear_vectors()
        serialized_array = np.zeros(inputs.asarray().size + outputs.asarray().size)
        serialized_array[: inputs.asarray().size] = inputs.asarray(copy=True)
        serialized_array[inputs.asarray().size :] = outputs.asarray(copy=True)
        return serialized_array

    def _set_linearization_point(
        self,
        linearization_state: np.ndarray,
    ) -> None:
        """
        Restores the input and output vectors of the wrapped OpenMDAO problem
        from a serialized linearization point created by
        `_get_linearization_point`.

        Parameters
        ----------
        linearization_state: np.ndarray
            Serialized input and output data of the OpenMDAO problem.
        """
        inputs, outputs, _ = self._time_stage_problem.model.get_nonlinear_vectors()
        inputs.asarray()[:] = linearization_state[: inputs.asarray().size]
        outputs.asarray()[:] = linearization_state[inputs.asarray().size :]

    def _input_state_to_om_vector(
        self,
        time: np.ndarray,
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
        om_vector: Vector,
        factor: float = 1.0,
    ) -> None:
        """
        Transfers the time and the input quantities of a discretized ODE input state
        into the source locations of the given OpenMDAO vector.

        Parameters
        ----------
        time: np.ndarray
            Time at which the ODE is evaluated.
        step_input: np.ndarray
            Input data coming from the start of a time step.
        stage_input: np.ndarray
            Input data coming from the start of a time stage.
        independent_input: np.ndarray
            Time independent input data.
        om_vector: Vector
            OpenMDAO vector into which the data is written.
        factor: float
            Factor multiplying all transferred data.
        """
        if self.time_integration_metadata.time_variable:
            om_vector[
                self._time_stage_problem.model.get_source(
                    self.time_integration_metadata.time_variable
                )
            ] = (
                time[0] * factor
            )  # pylint: disable=superfluous-parens
            # black introduces them, and black > pylint
        for quantity in self.time_integration_metadata.time_integration_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                if quantity.translation_metadata.step_input_var is not None:
                    om_vector[
                        self._time_stage_problem.model.get_source(
                            quantity.translation_metadata.step_input_var
                        )
                    ] = factor * step_input[start:end].reshape(
                        quantity.array_metadata.shape
                    )
                    om_vector[
                        self._time_stage_problem.model.get_source(
                            quantity.translation_metadata.accumulated_stage_var
                        )
                    ] = factor * stage_input[start:end].reshape(
                        quantity.array_metadata.shape
                    )
        for (
            quantity
        ) in self.time_integration_metadata.time_independent_input_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                om_vector[
                    self._time_stage_problem.model.get_source(
                        quantity.translation_metadata.time_independent_input_var
                    )
                ] = factor * independent_input[start:end].reshape(
                    quantity.array_metadata.shape
                )

    def _output_state_to_om_vector(
        self,
        stage_update: np.ndarray,
        stage_state: np.ndarray,  # currently not used
        independent_output: np.ndarray,  # currently not used
        om_vector: Vector,
        factor=1.0,
    ) -> None:
        """
        Transfers the stage update of a discretized ODE result state into the stage
        output variables of the given OpenMDAO vector.

        Parameters
        ----------
        stage_update: np.ndarray
            Output data coming from the update of a time stage.
        stage_state: np.ndarray
            State data of a time stage (currently not used).
        independent_output: np.ndarray
            Time independent output data (currently not used).
        om_vector: Vector
            OpenMDAO vector into which the data is written.
        factor: float
            Factor multiplying all transferred data.
        """
        for quantity in self.time_integration_metadata.time_integration_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                om_vector[quantity.translation_metadata.stage_output_var] = factor * (
                    stage_update[start:end].reshape(quantity.array_metadata.shape)
                )

    def _om_vector_to_input_state(
        self,
        om_vector: Vector,
        time: np.ndarray,
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
    ) -> None:
        """
        Reads the time and the input quantities from the given OpenMDAO vector back
        into the passed arrays of a discretized ODE input state.

        Parameters
        ----------
        om_vector: Vector
            OpenMDAO vector from which the data is read.
        time: np.ndarray
            Array receiving the time at which the ODE is evaluated.
        step_input: np.ndarray
            Array receiving the input data coming from the start of a time step.
        stage_input: np.ndarray
            Array receiving the input data coming from the start of a time stage.
        independent_input: np.ndarray
            Array receiving the time independent input data.
        """
        if self.time_integration_metadata.time_variable:
            time[0] = om_vector[
                self._time_stage_problem.model.get_source(
                    self.time_integration_metadata.time_variable
                )
            ][0]
        for quantity in self.time_integration_metadata.time_integration_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                if quantity.translation_metadata.step_input_var is not None:
                    step_input[start:end] = om_vector[
                        self._time_stage_problem.model.get_source(
                            quantity.translation_metadata.step_input_var
                        )
                    ].flatten()
                    stage_input[start:end] = om_vector[
                        self._time_stage_problem.model.get_source(
                            quantity.translation_metadata.accumulated_stage_var
                        )
                    ].flatten()
        for (
            quantity
        ) in self.time_integration_metadata.time_independent_input_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                independent_input[start:end] = om_vector[
                    self._time_stage_problem.model.get_source(
                        quantity.translation_metadata.time_independent_input_var
                    )
                ].flatten()

    def _om_vector_to_output_state(
        self,
        om_vector: Vector,
        stage_update: np.ndarray,
        stage_state: np.ndarray,  # currently not used
        independent_output: np.ndarray,  # currently not used
    ) -> None:
        """
        Reads the stage update from the stage output variables of the given OpenMDAO
        vector back into the passed arrays of a discretized ODE result state.

        Parameters
        ----------
        om_vector: Vector
            OpenMDAO vector from which the data is read.
        stage_update: np.ndarray
            Array receiving the output data coming from the update of a time stage.
        stage_state: np.ndarray
            Array receiving the state data of a time stage (currently not used).
        independent_output: np.ndarray
            Array receiving the time independent output data (currently not used).
        """
        for quantity in self.time_integration_metadata.time_integration_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                stage_update[start:end] = om_vector[
                    quantity.translation_metadata.stage_output_var
                ].flatten()

    def compute_state_norm(self, state: DiscretizedODEResultState) -> float:
        """
        Computes the norm of the stage state of the provided state by accumulating
        the partial norms of the time integrated quantities, where the contribution
        of distributed quantities is combined across MPI processes. Excluded
        quantities are skipped and the norm is taken in the configured order.

        Parameters
        ----------
        state: DiscretizedODEResultState
            State of which the norm is to be calculated

        Returns
        -------
        norm: float
            Norm of provided state
        """
        stage_state = state.stage_state
        non_distributed_intermediate = 0.0
        distributed_intermediate = 0.0
        for quantity in self.time_integration_metadata.time_integration_quantity_list:
            if quantity.name not in self._norm_exclusions:
                norm_intermediate = self._partial_norm_intermediate(
                    stage_state, quantity, self._norm_order
                )
                if quantity.array_metadata.distributed:
                    distributed_intermediate = self._add_to_intermediate(
                        distributed_intermediate, norm_intermediate, self._norm_order
                    )
                else:
                    non_distributed_intermediate = self._add_to_intermediate(
                        non_distributed_intermediate,
                        norm_intermediate,
                        self._norm_order,
                    )
        global_intermediate = self._allreduce_own_intermediate(
            distributed_intermediate, self._norm_order
        )
        global_intermediate = self._add_to_intermediate(
            global_intermediate, non_distributed_intermediate, self._norm_order
        )
        return self._normalize(global_intermediate, self._norm_order)

    def _partial_norm_intermediate(
        self, state: np.ndarray, quantity: TimeIntegrationQuantity, order: float
    ) -> float:
        """
        Computes the contribution of one time integrated quantity to the norm as its
        partial norm raised to the power of the norm order.

        Parameters
        ----------
        state: np.ndarray
            State of which the partial norm is to be calculated.
        quantity: TimeIntegrationQuantity
            The time integrated quantity the partial norm belongs to.
        order: float
            Order of the norm.

        Returns
        -------
        intermediate: float
            Partial norm contribution of the quantity.
        """
        start = quantity.array_metadata.start_index
        end = quantity.array_metadata.end_index
        exponent = 1.0 if order in [np.inf, -np.inf, 0.0] else order
        return np.linalg.norm(state[start:end], order) ** exponent

    def _add_to_intermediate(
        self, intermediate_value: float, added_value: float, order: float
    ) -> float:
        """
        Combines two intermediate norm values according to the norm order, using the
        maximum for the inf order, the minimum for the -inf order, and addition
        otherwise.

        Parameters
        ----------
        intermediate_value: float
            Accumulated intermediate norm value so far.
        added_value: float
            Intermediate norm value to combine.
        order: float
            Order of the norm.

        Returns
        -------
        intermediate: float
            Combined intermediate norm value.
        """
        if order == np.inf:
            return max(intermediate_value, added_value)
        elif order == -np.inf:
            return min(intermediate_value, added_value)
        else:
            return intermediate_value + added_value

    def _allreduce_own_intermediate(
        self, intermediate_value: float, order: float
    ) -> float:
        """
        Combines the intermediate norm value of this MPI process with the ones of all
        other processes via an allreduce, using an operation matching the norm
        order.

        Parameters
        ----------
        intermediate_value: float
            Intermediate norm value of this process.
        order: float
            Order of the norm.

        Returns
        -------
        intermediate: float
            Intermediate norm value combined across all processes.
        """
        mpi_op = (
            MPI.MAX if order == np.inf else MPI.MIN if order == -np.inf else MPI.SUM
        )
        return self._time_stage_problem.comm.allreduce(intermediate_value, mpi_op)

    def _normalize(self, norm_intermediate: float, order: float) -> float:
        """
        Inverts the accumulation of the norm by raising the intermediate norm value
        to the power of the reciprocal norm order.
        Parameters
        ----------
        norm_intermediate: float
            Accumulated intermediate norm value.
        order: float
            Order of the norm.

        Returns
        -------
        norm: float
            Completed norm.
        """
        exponent = 1.0 if order in [np.inf, -np.inf, 0.0] else 1.0 / order
        return norm_intermediate**exponent
