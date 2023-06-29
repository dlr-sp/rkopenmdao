from typing import Tuple, Union

import numpy as np
import openmdao.api as om

from .integration_control import IntegrationControl


class InnerProblemComputeFunctor:
    def __init__(
        self,
        inner_problem: om.Problem,
        integration_control: IntegrationControl,
        quantity_metadata: dict,
        resets: Union[bool, list],
    ):
        self.inner_problem: om.Problem = inner_problem
        self.integration_control: IntegrationControl = integration_control
        self.quantity_metadata: dict = quantity_metadata
        self.resets = resets

    def __call__(
        self,
        old_state: np.ndarray,
        accumulated_stages: np.ndarray,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
    ) -> np.ndarray:
        if self.resets:
            self.reset_values()
        self.fill_inputs(old_state, accumulated_stages)

        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element

        self.inner_problem.model.run_solve_nonlinear()

        stage_state = np.zeros_like(old_state)
        self.get_outputs(stage_state)
        return stage_state

    def reset_values(self):
        if isinstance(self.resets, bool):
            self.inner_problem.model._outputs.imul(0.0)
            self.inner_problem.model._inputs.imul(0.0)
        else:
            for var in self.resets:
                self.inner_problem[var] *= 0.0

    def fill_inputs(self, old_state: np.ndarray, accumulated_stage: np.ndarray):
        for quantity, metadata in self.quantity_metadata.items():
            start = metadata["numpy_start_index"]
            end = start + np.prod(metadata["shape"])
            self.inner_problem.set_val(
                metadata["step_input_var"], old_state[start:end].reshape(metadata["shape"])
            )

            self.inner_problem.model.set_val(
                metadata["accumulated_stage_var"],
                accumulated_stage[start:end].reshape(metadata["shape"]),
            )

    def get_outputs(self, stage_state: np.ndarray):
        for quantity, metadata in self.quantity_metadata.items():
            start = metadata["numpy_start_index"]
            end = start + np.prod(metadata["shape"])
            stage_state[start:end] = self.inner_problem.get_val(
                metadata["stage_output_var"], get_remote=False
            ).flatten()


class InnerProblemComputeJacvecFunctor:
    def __init__(
        self,
        inner_problem: om.Problem,
        integration_control: IntegrationControl,
        quantity_metadata: dict,
        of_vars: list,
        wrt_vars: list,
    ):
        self.inner_problem: om.Problem = inner_problem
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
        self.inner_problem.model.run_linearize()
        seed = {}
        self.fill_seed(old_state_perturbation, accumulated_stage_perturbation, seed)
        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element
        jvp = self.inner_problem.compute_jacvec_product(
            of=self.of_vars, wrt=self.wrt_vars, mode="fwd", seed=seed
        )

        # for key, vector in jvp.items():
        #     print(key, vector)
        stage_perturbation = np.zeros_like(old_state_perturbation)
        self.extract_jvp(jvp, stage_perturbation)

        return stage_perturbation

    def fill_seed(
        self, old_state_perturbation: np.ndarray, accumulated_stage_perturbation: np.ndarray, seed
    ):
        for quantity, metadata in self.quantity_metadata.items():
            start = metadata["numpy_start_index"]
            end = start + np.prod(metadata["shape"])
            seed[metadata["step_input_var"]] = old_state_perturbation[start:end].reshape(
                metadata["shape"]
            )
            seed[metadata["accumulated_stage_var"]] = accumulated_stage_perturbation[
                start:end
            ].reshape(metadata["shape"])

    def extract_jvp(self, jvp: dict, stage_perturbation: np.ndarray):
        for quantity, metadata in self.quantity_metadata.items():
            start = metadata["numpy_start_index"]
            end = start + np.prod(metadata["shape"])
            stage_perturbation[start:end] = jvp[metadata["stage_output_var"]].flatten()

    def linearize(self, inputs=None, outputs=None):
        # linearize always needs to be called, this checks whether we need to solve the nonlinear
        # system beforehand (e.g. due to manually changed inputs/outputs)
        if inputs is not None and outputs is not None:
            self.inner_problem.model._inputs = inputs
            self.inner_problem.model._outputs = outputs
        elif inputs is not None or outputs is not None:
            # TODO: raise actual error
            print("Error")


class InnerProblemComputeTransposeJacvecFunctor:
    def __init__(
        self,
        inner_problem: om.Problem,
        integration_control: IntegrationControl,
        quantity_metadata: dict,
        of_vars: list,
        wrt_vars: list,
    ):
        self.inner_problem: om.Problem = inner_problem
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
        self.inner_problem.model.run_linearize()
        seed = {}
        self.fill_seed(stage_perturbation, seed)
        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element

        jvp = self.inner_problem.compute_jacvec_product(
            of=self.of_vars, wrt=self.wrt_vars, mode="rev", seed=seed
        )

        old_state_perturbation = np.zeros_like(stage_perturbation)
        accumulated_stage_perturbation = np.zeros_like(stage_perturbation)
        self.extract_jvp(jvp, old_state_perturbation, accumulated_stage_perturbation)
        return old_state_perturbation, accumulated_stage_perturbation

    def fill_seed(self, stage_perturbation: np.ndarray, seed: dict):
        for quantity, metadata in self.quantity_metadata.items():
            start = metadata["numpy_start_index"]
            end = start + np.prod(metadata["shape"])
            seed[metadata["stage_output_var"]] = stage_perturbation[start:end].reshape(
                metadata["shape"]
            )

    def extract_jvp(
        self,
        jvp: dict,
        old_state_perturbation: np.ndarray,
        accumulated_stage_perturbation: np.ndarray,
    ):
        for quantity, metadata in self.quantity_metadata.items():
            start = metadata["numpy_start_index"]
            end = start + np.prod(metadata["shape"])
            old_state_perturbation[start:end] = jvp[metadata["step_input_var"]].flatten()
            accumulated_stage_perturbation[start:end] = jvp[
                metadata["accumulated_stage_var"]
            ].flatten()

    def linearize(self, inputs=None, outputs=None):
        # linearize always needs to be called, this checks whether we need to solve the nonlinear
        # system beforehand (e.g. due to manually changed inputs/outputs)
        if inputs is not None and outputs is not None:
            self.inner_problem.model._inputs = inputs
            self.inner_problem.model._outputs = outputs
        elif inputs is not None or outputs is not None:
            # TODO: raise actual error
            print("Error")
