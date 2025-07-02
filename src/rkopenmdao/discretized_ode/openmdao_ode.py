# pylint: disable=missing-module-docstring, unused-argument

from dataclasses import dataclass
import inspect
from typing import Union, Callable

import numpy as np
import openmdao.api as om
from openmdao.vectors.vector import Vector


from rkopenmdao.metadata_extractor import (
    TimeIntegrationMetadata,
    extract_time_integration_metadata,
    add_time_independent_input_metadata,
    add_distributivity_information,
)
from rkopenmdao.integration_control import IntegrationControl
from .discretized_ode import (
    DiscretizedODE,
    DiscretizedODELinearizationPoint,
    DiscretizedODEInputState,
    DiscretizedODEResultState,
)


@dataclass
class OpenMDAOODELinearizationPoint(DiscretizedODELinearizationPoint):
    """Linearization point class for OpenMDAOODE."""

    stage_time: float
    inputs_copy: np.ndarray
    outputs_copy: np.ndarray

    def to_numpy_array(self) -> np.ndarray:
        serialized_array = np.zeros(1 + self.inputs_copy.size + self.outputs_copy.size)
        serialized_array[0] = self.stage_time
        serialized_array[1 : 1 + self.inputs_copy.size] = self.inputs_copy
        serialized_array[1 + self.outputs_copy.size :] = self.outputs_copy
        return serialized_array

    def from_numpy_array(self, array: np.ndarray) -> None:
        self.stage_time = array[0]
        self.inputs_copy = array[1 : 1 + self.inputs_copy.size]
        self.outputs_copy = array[1 + self.outputs_copy.size :]


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
    integration_control: IntegrationControl
        Object for passing time related information between this class and the inner
        OpenMDAO problem.
    time_integration_quantities: list
        Quantities to be time integrated that are searched for in the inner problem.
    independent_input_quantities: list
        Quantities that act as time independent inputs that are seached for in the inner
        problem.

    Attributes
    ----------
    time_integration_metadata : TimeIntegrationMetadata
        Metadata containing information about the shape and location of
        quantities related to time integration.
    """

    time_integration_metadata: TimeIntegrationMetadata

    _time_stage_problem: om.Problem
    _integration_control: IntegrationControl
    _om_solve_linear_runner: Callable[[str], None]

    def __init__(
        self,
        time_stage_problem: om.Problem,
        integration_control: IntegrationControl,
        time_integration_quantities: list,
        independent_input_quantities: Union[list, None] = None,
    ):
        self._time_stage_problem = time_stage_problem
        self._integration_control = integration_control
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
            self._om_solve_linear_runner = (
                lambda mode: self._time_stage_problem.model.run_solve_linear(mode=mode)
            )
        else:
            self._om_solve_linear_runner = (
                lambda mode: self._time_stage_problem.model.run_solve_linear(
                    vec_names=["linear"], mode=mode
                )
            )

    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:

        _, outputs, _ = self._time_stage_problem.model.get_nonlinear_vectors()
        self._input_state_to_om_vector(
            ode_input.step_input,
            ode_input.stage_input,
            ode_input.independent_input,
            outputs,
        )
        self._integration_control.stage_time = ode_input.time
        self._integration_control.delta_t = step_size
        self._integration_control.butcher_diagonal_element = stage_factor
        self._time_stage_problem.model.run_solve_nonlinear()

        stage_update = np.zeros_like(ode_input.step_input)
        stage_state = np.zeros_like(
            0
        )  # Currently not used, needs update in metadata_extractor
        independent_output = np.zeros(
            0
        )  # Currently not used, needs update in metadata_extractor

        self._om_vector_to_output_state(
            outputs, stage_update, stage_state, independent_output
        )

        return DiscretizedODEResultState(stage_update, stage_state, independent_output)

    def get_linearization_point(self) -> OpenMDAOODELinearizationPoint:
        inputs, outputs, _ = self._time_stage_problem.model.get_nonlinear_vectors()
        return OpenMDAOODELinearizationPoint(
            self._integration_control.stage_time,
            inputs.asarray(copy=True),
            outputs.asarray(copy=True),
        )

    def set_linearization_point(self, cache: OpenMDAOODELinearizationPoint) -> None:
        self._integration_control.stage_time = cache.stage_time
        inputs, outputs, _ = self._time_stage_problem.model.get_nonlinear_vectors()
        inputs.asarray()[:] = cache.inputs_copy
        outputs.asarray()[:] = cache.outputs_copy

    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        self._integration_control.delta_t = step_size
        self._integration_control.butcher_diagonal_element = stage_factor
        self._time_stage_problem.model.run_linearize()
        (_, d_outputs, d_residuals) = (
            self._time_stage_problem.model.get_linear_vectors()
        )
        d_residuals.asarray()[:] *= 0.0
        self._input_state_to_om_vector(
            ode_input_perturbation.step_input,
            ode_input_perturbation.stage_input,
            ode_input_perturbation.independent_input,
            d_residuals,
            -1.0,
        )

        self._om_solve_linear_runner("fwd")

        stage_update_pert = np.zeros_like(ode_input_perturbation.step_input)
        stage_state_pert = np.zeros(
            0
        )  # Currently not used, needs update in metadata_extractor
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
        self._integration_control.delta_t = step_size
        self._integration_control.butcher_diagonal_element = stage_factor
        self._time_stage_problem.model.run_linearize()
        (_, d_outputs, d_residuals) = (
            self._time_stage_problem.model.get_linear_vectors()
        )
        d_outputs.asarray()[:] *= 0.0
        self._output_state_to_om_vector(
            ode_result_perturbation.stage_update,
            ode_result_perturbation.stage_state,
            ode_result_perturbation.independent_output,
            d_outputs,
            -1.0,
        )
        self._om_solve_linear_runner("rev")
        step_input_pert = np.zeros(
            self.time_integration_metadata.time_integration_array_size
        )
        stage_input_pert = np.zeros(
            self.time_integration_metadata.time_integration_array_size
        )
        independent_input_pert = np.zeros(
            self.time_integration_metadata.time_independent_input_size
        )
        self._om_vector_to_input_state(
            d_residuals, step_input_pert, stage_input_pert, independent_input_pert
        )
        return DiscretizedODEInputState(
            step_input_pert, stage_input_pert, independent_input_pert, 0.0
        )

    def _input_state_to_om_vector(
        self,
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
        om_vector: Vector,
        factor: float = 1.0,
    ) -> None:
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
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
    ) -> None:
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
        for quantity in self.time_integration_metadata.time_integration_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                stage_update[start:end] = om_vector[
                    quantity.translation_metadata.stage_output_var
                ].flatten()
