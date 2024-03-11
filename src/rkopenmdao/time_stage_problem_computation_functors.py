"""
Wrapper classes to make a openMDAO problem modelling a time stage for the Runge-Kutta integrator usable with the
RungeKuttaScheme class.
"""

from typing import Tuple

import numpy as np
import openmdao.api as om

from .integration_control import IntegrationControl
from .errors import TimeStageError


class TimeStageProblemComputeFunctor:
    """
    Wraps an openMDAO problem (specifically its models run_solve_nonlinear method) to a functor usable in the
    RungeKuttaScheme class.
    """

    def __init__(
        self,
        time_stage_problem: om.Problem,
        integration_control: IntegrationControl,
        quantity_metadata: dict,
    ):
        self.time_stage_problem: om.Problem = time_stage_problem
        self.integration_control: IntegrationControl = integration_control
        self.quantity_metadata: dict = quantity_metadata

    def __call__(
        self,
        old_state: np.ndarray,
        accumulated_stages: np.ndarray,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
    ) -> np.ndarray:
        self.fill_inputs(old_state, accumulated_stages)

        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element

        self.time_stage_problem.model.run_solve_nonlinear()

        stage_state = np.zeros_like(old_state)
        self.get_outputs(stage_state)
        return stage_state

    def fill_inputs(self, old_state: np.ndarray, accumulated_stage: np.ndarray):
        """Write data into the input vectors of the owned problem."""
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = start + np.prod(metadata["shape"])
                if metadata["step_input_var"] is not None:
                    self.time_stage_problem.set_val(
                        metadata["step_input_var"],
                        old_state[start:end].reshape(metadata["shape"]),
                    )

                    self.time_stage_problem.set_val(
                        metadata["accumulated_stage_var"],
                        accumulated_stage[start:end].reshape(metadata["shape"]),
                    )

    def get_outputs(self, stage_state: np.ndarray):
        """Extract data from the output vectors of the owned problem."""
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = start + np.prod(metadata["shape"])
                stage_state[start:end] = self.time_stage_problem.get_val(
                    metadata["stage_output_var"], get_remote=False
                ).flatten()


class TimeStageProblemComputeJacvecFunctor:
    """
    Wraps an openMDAO problem (specifically its compute_jacvec_problem function) to a functor usable in the
    RungeKuttaScheme class. Uses the 'fwd'-mode of said function.
    """

    def __init__(
        self,
        time_stage_problem: om.Problem,
        integration_control: IntegrationControl,
        quantity_metadata: dict,
        of_vars: list,
        wrt_vars: list,
    ):
        self.time_stage_problem: om.Problem = time_stage_problem
        self.integration_control: IntegrationControl = integration_control
        self.quantity_metadata: dict = quantity_metadata
        self.of_vars = of_vars
        self.wrt_vars = wrt_vars

    def __call__(
        self,
        old_state_perturbation: np.ndarray,
        accumulated_stage_perturbation: np.ndarray,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
    ) -> np.ndarray:
        self.time_stage_problem.model.run_linearize()
        self.fill_vector(old_state_perturbation, accumulated_stage_perturbation)

        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element
        self.time_stage_problem.model.run_solve_linear("fwd")

        stage_perturbation = np.zeros_like(old_state_perturbation)
        self.extract_vector(stage_perturbation)

        return stage_perturbation

    def fill_vector(
        self,
        old_state_perturbation: np.ndarray,
        accumulated_stage_perturbation: np.ndarray,
    ):
        """Fills d_residuals of the time_stage_problem to prepare for jacvec product."""
        (_, _, d_residuals) = self.time_stage_problem.model.get_linear_vectors()
        d_residuals.asarray()[:] *= 0.0
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = start + np.prod(metadata["shape"])
                if metadata["step_input_var"] is not None:
                    d_residuals[
                        self.time_stage_problem.model.get_source(
                            metadata["step_input_var"]
                        )
                    ] = -old_state_perturbation[start:end].reshape(metadata["shape"])
                    d_residuals[
                        self.time_stage_problem.model.get_source(
                            metadata["accumulated_stage_var"]
                        )
                    ] = -accumulated_stage_perturbation[start:end].reshape(
                        metadata["shape"]
                    )

    def extract_vector(self, stage_perturbation: np.ndarray):
        """Extracts the result of the jacvec product from d_outputs of the time_stage_problem."""
        for metadata in self.quantity_metadata.values():
            _, d_outputs, _ = self.time_stage_problem.model.get_linear_vectors()
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = start + np.prod(metadata["shape"])
                stage_perturbation[start:end] = d_outputs[
                    metadata["stage_output_var"]
                ].flatten()

    def linearize(self, inputs=None, outputs=None):
        """Sets inputs and outputs of the owned problem to prepare for a different linearization point."""
        if inputs is not None and outputs is not None:
            (
                prob_inputs,
                prob_outputs,
                _,
            ) = self.time_stage_problem.model.get_nonlinear_vectors()
            prob_inputs.set_vec(inputs)
            prob_outputs.set_vec(outputs)
        elif inputs is not None or outputs is not None:
            raise TimeStageError(
                "Either both or none of inputs and outputs must be given."
            )


class TimeStageProblemComputeTransposeJacvecFunctor:
    """
    Wraps an openMDAO problem (specifically its compute_jacvec_problem function) to a functor usable in the
    RungeKuttaScheme class. Uses the 'rev'-mode of said function.
    """

    def __init__(
        self,
        time_stage_problem: om.Problem,
        integration_control: IntegrationControl,
        quantity_metadata: dict,
        of_vars: list,
        wrt_vars: list,
    ):
        self.time_stage_problem: om.Problem = time_stage_problem
        self.integration_control: IntegrationControl = integration_control
        self.quantity_metadata: dict = quantity_metadata
        self.of_vars = of_vars
        self.wrt_vars = wrt_vars

    def __call__(
        self,
        stage_perturbation: np.ndarray,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.time_stage_problem.model.run_linearize()
        self.fill_vector(stage_perturbation)
        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element

        self.time_stage_problem.model.run_solve_linear("rev")

        old_state_perturbation = np.zeros_like(stage_perturbation)
        accumulated_stage_perturbation = np.zeros_like(stage_perturbation)
        self.extract_vector(old_state_perturbation, accumulated_stage_perturbation)
        return old_state_perturbation, accumulated_stage_perturbation

    def fill_vector(self, stage_perturbation: np.ndarray):
        """Fills d_outputs of the time_stage_problem to prepare for jacvec product."""
        (_, d_outputs, _) = self.time_stage_problem.model.get_linear_vectors()
        d_outputs.asarray()[:] *= 0
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = start + np.prod(metadata["shape"])
                d_outputs[metadata["stage_output_var"]] = -stage_perturbation[
                    start:end
                ].reshape(metadata["shape"])

    def extract_vector(
        self,
        old_state_perturbation: np.ndarray,
        accumulated_stage_perturbation: np.ndarray,
    ):
        """Extracts the result of the jacvec product from d_residuals of the time_stage_problem."""
        _, _, d_residuals = self.time_stage_problem.model.get_linear_vectors()
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = start + np.prod(metadata["shape"])
                if metadata["step_input_var"] is not None:
                    old_state_perturbation[start:end] = d_residuals[
                        self.time_stage_problem.model.get_source(
                            metadata["step_input_var"]
                        )
                    ].flatten()
                    accumulated_stage_perturbation[start:end] = d_residuals[
                        self.time_stage_problem.model.get_source(
                            metadata["accumulated_stage_var"]
                        )
                    ].flatten()

    def linearize(self, inputs=None, outputs=None):
        """Sets inputs and outputs of the owned problem to prepare for a different linearization point."""
        if inputs is not None and outputs is not None:
            (
                prob_inputs,
                prob_outputs,
                _,
            ) = self.time_stage_problem.model.get_nonlinear_vectors()
            prob_inputs.set_vec(inputs)
            prob_outputs.set_vec(outputs)
        elif inputs is not None or outputs is not None:
            raise TimeStageError(
                "Either both or none of inputs and outputs must be given."
            )
