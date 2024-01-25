"""
Wrapper classes to make a openMDAO problem modelling a time stage for the Runge-Kutta integrator usable with the
RungeKuttaScheme class.
"""

from typing import Tuple

import numpy as np
import openmdao.api as om
from openmdao.utils.general_utils import ContainsAll

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
        seed = {}
        self.fill_seed(old_state_perturbation, accumulated_stage_perturbation, seed)
        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element

        jvp = self.time_stage_problem.compute_jacvec_product(
            of=self.of_vars, wrt=self.wrt_vars, mode="fwd", seed=seed
        )
        # for key, vector in jvp.items():
        #     print(key, vector)
        stage_perturbation = np.zeros_like(old_state_perturbation)
        self.extract_jvp(jvp, stage_perturbation)

        return stage_perturbation

    def fill_seed(
        self,
        old_state_perturbation: np.ndarray,
        accumulated_stage_perturbation: np.ndarray,
        seed,
    ):
        """Write data into a seed to be used by the compute_jacvec_product function by the owned problem."""
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = start + np.prod(metadata["shape"])
                if metadata["step_input_var"] is not None:
                    seed[metadata["step_input_var"]] = old_state_perturbation[
                        start:end
                    ].reshape(metadata["shape"])
                    seed[
                        metadata["accumulated_stage_var"]
                    ] = accumulated_stage_perturbation[start:end].reshape(
                        metadata["shape"]
                    )

    def extract_jvp(self, jvp: dict, stage_perturbation: np.ndarray):
        """Extracts data from the result of the owned problems compute_jacvec_product functions."""
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = start + np.prod(metadata["shape"])
                stage_perturbation[start:end] = jvp[
                    metadata["stage_output_var"]
                ].flatten()

    def linearize(self, inputs=None, outputs=None):
        """Sets inputs and outputs of the owned problem to prepare for a different linearization point."""
        if inputs is not None and outputs is not None:
            self.time_stage_problem.model._inputs.set_vec(inputs)
            self.time_stage_problem.model._outputs.set_vec(outputs)
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
        seed = {}
        self.fill_seed(stage_perturbation, seed)
        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element

        jvp = self.time_stage_problem.compute_jacvec_product(
            of=self.of_vars, wrt=self.wrt_vars, mode="rev", seed=seed
        )

        old_state_perturbation = np.zeros_like(stage_perturbation)
        accumulated_stage_perturbation = np.zeros_like(stage_perturbation)
        self.extract_jvp(jvp, old_state_perturbation, accumulated_stage_perturbation)
        return old_state_perturbation, accumulated_stage_perturbation

    def fill_seed(self, stage_perturbation: np.ndarray, seed: dict):
        """Write data into a seed to be used by the compute_jacvec_product function by the owned problem."""
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = start + np.prod(metadata["shape"])
                seed[metadata["stage_output_var"]] = stage_perturbation[
                    start:end
                ].reshape(metadata["shape"])

    def extract_jvp(
        self,
        jvp: dict,
        old_state_perturbation: np.ndarray,
        accumulated_stage_perturbation: np.ndarray,
    ):
        """Extracts data from the result of the owned problems compute_jacvec_product functions."""
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "time_integration":
                start = metadata["numpy_start_index"]
                end = start + np.prod(metadata["shape"])
                if metadata["step_input_var"] is not None:
                    old_state_perturbation[start:end] = jvp[
                        metadata["step_input_var"]
                    ].flatten()
                    accumulated_stage_perturbation[start:end] = jvp[
                        metadata["accumulated_stage_var"]
                    ].flatten()

    def linearize(self, inputs=None, outputs=None):
        """Sets inputs and outputs of the owned problem to prepare for a different linearization point."""
        if inputs is not None and outputs is not None:
            self.time_stage_problem.model._inputs.set_vec(inputs)
            self.time_stage_problem.model._outputs.set_vec(outputs)
        elif inputs is not None or outputs is not None:
            raise TimeStageError(
                "Either both or none of inputs and outputs must be given."
            )
