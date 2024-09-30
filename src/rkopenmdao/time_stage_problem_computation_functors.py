"""Wrapper classes to make a openMDAO problem modelling a time stage for the
RungeKuttaIntegrator usable with the RungeKuttaScheme class."""

from __future__ import annotations

import numpy as np
import openmdao.api as om

from .integration_control import IntegrationControl
from .errors import TimeStageError
from .metadata_extractor import TimeIntegrationMetadata


class TimeStageProblemComputeFunctor:
    """Wraps an openMDAO problem (specifically its models run_solve_nonlinear method) to
    a functor usable in the RungeKuttaScheme class."""

    def __init__(
        self,
        time_stage_problem: om.Problem,
        integration_control: IntegrationControl,
        time_integration_metadata: TimeIntegrationMetadata,
    ):
        self.time_stage_problem: om.Problem = time_stage_problem
        self.integration_control: IntegrationControl = integration_control
        self.time_integration_metadata: TimeIntegrationMetadata = (
            time_integration_metadata
        )

    def __call__(
        self,
        old_state: np.ndarray,
        accumulated_stages: np.ndarray,
        parameters: np.ndarray,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
    ) -> np.ndarray:
        self.fill_problem_data(old_state, accumulated_stages, parameters)

        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element

        self.time_stage_problem.model.run_solve_nonlinear()

        stage_state = np.zeros_like(old_state)
        self.get_problem_data(stage_state)
        return stage_state

    def fill_problem_data(
        self,
        old_state: np.ndarray,
        accumulated_stage: np.ndarray,
        parameters: np.ndarray,
    ):
        """Fills internal OpenMDAO vectors."""
        _, outputs, _ = self.time_stage_problem.model.get_nonlinear_vectors()
        for quantity in self.time_integration_metadata.quantity_list:
            if quantity.type == "time_integration" and quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                if quantity.translation_metadata.step_input_var is not None:
                    outputs[
                        self.time_stage_problem.model.get_source(
                            quantity.translation_metadata.step_input_var
                        )
                    ] = old_state[start:end].reshape(quantity.array_metadata.shape)
                    outputs[
                        self.time_stage_problem.model.get_source(
                            quantity.translation_metadata.accumulated_stage_var
                        )
                    ] = accumulated_stage[start:end].reshape(
                        quantity.array_metadata.shape
                    )
            elif quantity.type == "independent_input" and quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                outputs[
                    self.time_stage_problem.model.get_source(
                        quantity.translation_metadata.time_independent_input_var
                    )
                ] = parameters[start:end]

    def get_problem_data(self, stage_state: np.ndarray):
        """Extract data from the output vectors of the owned problem."""
        _, outputs, _ = self.time_stage_problem.model.get_nonlinear_vectors()
        for quantity in self.time_integration_metadata.quantity_list:
            if quantity.type == "time_integration" and quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                stage_state[start:end] = outputs[
                    quantity.translation_metadata.stage_output_var
                ].flatten()


class TimeStageProblemComputeJacvecFunctor:
    """Wraps an openMDAO problem (specifically its compute_jacvec_problem function) to a
    functor usable in the RungeKuttaScheme class. Uses the 'fwd'-mode of said
    function."""

    def __init__(
        self,
        time_stage_problem: om.Problem,
        integration_control: IntegrationControl,
        time_integration_metadata: TimeIntegrationMetadata,
        of_vars: list,
        wrt_vars: list,
    ):
        self.time_stage_problem: om.Problem = time_stage_problem
        self.integration_control: IntegrationControl = integration_control
        self.time_integration_metadata: TimeIntegrationMetadata = (
            time_integration_metadata
        )
        self.of_vars = of_vars
        self.wrt_vars = wrt_vars

    def __call__(
        self,
        old_state_perturbation: np.ndarray,
        accumulated_stage_perturbation: np.ndarray,
        parameter_perturbations: np.ndarray,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
    ) -> np.ndarray:
        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element
        self.time_stage_problem.model.run_linearize()
        self.fill_problem_data(
            old_state_perturbation,
            accumulated_stage_perturbation,
            parameter_perturbations,
        )

        try:
            self.time_stage_problem.model.run_solve_linear("fwd")
        except TypeError:  # old openMDAO version had different interface
            self.time_stage_problem.model.run_solve_linear(
                vec_names=["linear"], mode="fwd"
            )

        stage_perturbation = np.zeros_like(old_state_perturbation)
        self.get_problem_data(stage_perturbation)

        return stage_perturbation

    def fill_problem_data(
        self,
        old_state_perturbation: np.ndarray,
        accumulated_stage_perturbation: np.ndarray,
        parameter_perturbations: np.ndarray,
    ):
        """Fills d_residuals of the time_stage_problem to prepare for jacvec product."""
        (_, _, d_residuals) = self.time_stage_problem.model.get_linear_vectors()
        d_residuals.asarray()[:] *= 0.0
        for quantity in self.time_integration_metadata.quantity_list:
            if quantity.type == "time_integration" and quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                if quantity.translation_metadata.step_input_var is not None:
                    d_residuals[
                        self.time_stage_problem.model.get_source(
                            quantity.translation_metadata.step_input_var
                        )
                    ] = -old_state_perturbation[start:end].reshape(
                        quantity.array_metadata.shape
                    )
                    d_residuals[
                        self.time_stage_problem.model.get_source(
                            quantity.translation_metadata.accumulated_stage_var
                        )
                    ] = -accumulated_stage_perturbation[start:end].reshape(
                        quantity.array_metadata.shape
                    )
            elif quantity.type == "independent_input" and quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                d_residuals[
                    self.time_stage_problem.model.get_source(
                        quantity.translation_metadata.time_independent_input_var
                    )
                ] = -parameter_perturbations[start:end].reshape(
                    quantity.array_metadata.shape
                )

    def get_problem_data(self, stage_perturbation: np.ndarray):
        """Extracts the result of the jacvec product from d_outputs of the
        time_stage_problem."""
        _, d_outputs, _ = self.time_stage_problem.model.get_linear_vectors()
        for quantity in self.time_integration_metadata.quantity_list:
            if quantity.type == "time_integration" and quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                stage_perturbation[start:end] = d_outputs[
                    quantity.translation_metadata.stage_output_var
                ].flatten()

    def linearize(self, inputs=None, outputs=None):
        """Sets inputs and outputs of the owned problem to prepare for a different
        linearization point."""
        if inputs is not None and outputs is not None:
            (
                prob_inputs,
                prob_outputs,
                _,
            ) = self.time_stage_problem.model.get_nonlinear_vectors()
            prob_inputs.set_val(inputs)
            prob_outputs.set_val(outputs)
        elif inputs is not None or outputs is not None:
            raise TimeStageError(
                "Either both or none of inputs and outputs must be given."
            )


class TimeStageProblemComputeTransposeJacvecFunctor:
    """Wraps an openMDAO problem (specifically its compute_jacvec_problem function) to a
    functor usable in the RungeKuttaScheme class. Uses the 'rev'-mode of said
    function."""

    def __init__(
        self,
        time_stage_problem: om.Problem,
        integration_control: IntegrationControl,
        time_integration_metadata: TimeIntegrationMetadata,
        of_vars: list,
        wrt_vars: list,
    ):
        self.time_stage_problem: om.Problem = time_stage_problem
        self.integration_control: IntegrationControl = integration_control
        self.time_integration_metadata: TimeIntegrationMetadata = (
            time_integration_metadata
        )

        self.of_vars = of_vars
        self.wrt_vars = wrt_vars

    def __call__(
        self,
        stage_perturbation: np.ndarray,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element
        self.time_stage_problem.model.run_linearize()
        self.fill_problem_data(stage_perturbation)

        try:
            self.time_stage_problem.model.run_solve_linear("rev")
        except TypeError:  # old openMDAO version had different interface
            self.time_stage_problem.model.run_solve_linear(
                vec_names=["linear"], mode="rev"
            )

        old_state_perturbation = np.zeros_like(stage_perturbation)
        accumulated_stage_perturbation = np.zeros_like(stage_perturbation)
        parameter_perturbations = np.zeros(
            self.time_integration_metadata.time_independent_input_size
        )
        self.get_problem_data(
            old_state_perturbation,
            accumulated_stage_perturbation,
            parameter_perturbations,
        )
        return (
            old_state_perturbation,
            accumulated_stage_perturbation,
            parameter_perturbations,
        )

    def fill_problem_data(self, stage_perturbation: np.ndarray):
        """Fills d_outputs of the time_stage_problem to prepare for jacvec product."""
        _, d_outputs, _ = self.time_stage_problem.model.get_linear_vectors()
        d_outputs.asarray()[:] *= 0
        for quantity in self.time_integration_metadata.quantity_list:
            if quantity.type == "time_integration" and quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                d_outputs[quantity.translation_metadata.stage_output_var] = (
                    -stage_perturbation[start:end].reshape(
                        quantity.array_metadata.shape
                    )
                )

    def get_problem_data(
        self,
        old_state_perturbation: np.ndarray,
        accumulated_stage_perturbation: np.ndarray,
        parameter_perturbations: np.ndarray,
    ):
        """Extracts the result of the jacvec product from d_residuals of the
        time_stage_problem."""
        _, _, d_residuals = self.time_stage_problem.model.get_linear_vectors()
        for quantity in self.time_integration_metadata.quantity_list:
            if quantity.type == "time_integration" and quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                if quantity.translation_metadata.step_input_var is not None:
                    old_state_perturbation[start:end] = d_residuals[
                        self.time_stage_problem.model.get_source(
                            quantity.translation_metadata.step_input_var
                        )
                    ].flatten()
                    accumulated_stage_perturbation[start:end] = d_residuals[
                        self.time_stage_problem.model.get_source(
                            quantity.translation_metadata.accumulated_stage_var
                        )
                    ].flatten()
            if quantity.type == "independent_input" and quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                parameter_perturbations[start:end] = d_residuals[
                    self.time_stage_problem.model.get_source(
                        quantity.translation_metadata.time_independent_input_var
                    )
                ].flatten()

    def linearize(self, inputs=None, outputs=None):
        """Sets inputs and outputs of the owned problem to prepare for a different
        linearization point."""
        if inputs is not None and outputs is not None:
            (
                prob_inputs,
                prob_outputs,
                _,
            ) = self.time_stage_problem.model.get_nonlinear_vectors()
            prob_inputs.set_val(inputs)
            prob_outputs.set_val(outputs)
        elif inputs is not None or outputs is not None:
            raise TimeStageError(
                "Either both or none of inputs and outputs must be given."
            )
