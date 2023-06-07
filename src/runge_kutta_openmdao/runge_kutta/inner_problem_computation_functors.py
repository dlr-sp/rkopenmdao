from typing import Tuple

import numpy as np
import openmdao.api as om

from .integration_control import IntegrationControl


class InnerProblemComputeFunctor:
    def __init__(
        self,
        inner_problem: om.Problem,
        integration_control: IntegrationControl,
        quantity_metadata: dict,
    ):
        self.inner_problem: om.Problem = inner_problem
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

        self.inner_problem.model.run_solve_nonlinear()

        stage_state = np.zeros_like(old_state)
        self.get_outputs(stage_state)
        return stage_state

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
        self.need_call_solve_nonlinear = False

    def __call__(
        self,
        old_state_perturbation: np.ndarray,
        accumulated_stage_perturbation: np.ndarray,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
    ) -> np.ndarray:
        if self.need_call_solve_nonlinear:
            self.inner_problem.model.run_solve_nonlinear()
            self.need_call_solve_nonlinear = False

        self.inner_problem.model.run_linearize()
        seed = {}
        self.fill_seed(old_state_perturbation, accumulated_stage_perturbation, seed)
        self.integration_control.stage_time = stage_time
        self.integration_control.butcher_diagonal_element = butcher_diagonal_element

        jvp = self.inner_problem.compute_jacvec_product(
            of=self.of_vars, wrt=self.wrt_vars, mode="fwd", seed=seed
        )

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

    def linearize(
        self,
        inner_input_vec: om.DefaultVector = None,
        inner_output_vec: om.DefaultVector = None,
        numpy_old_state_vec: np.ndarray = None,
        numpy_acc_stage_vec: np.ndarray = None,
    ):
        # linearize always needs to be called, this checks whether we need to solve the nonlinear
        # system beforehand (e.g. due to manually changed inputs/outputs)
        if inner_input_vec is not None:
            self.inner_problem.model._inputs.set_vec(inner_input_vec)
            if inner_output_vec is not None:
                self.inner_problem.model._outputs.set_vec(inner_output_vec)
            else:
                self.need_call_solve_nonlinear = True
        elif numpy_old_state_vec is not None or numpy_acc_stage_vec is not None:
            self.need_call_solve_nonlinear = True
            if numpy_old_state_vec is not None:
                for quantity, metadata in self.quantity_metadata.items():
                    start = metadata["numpy_start_index"]
                    end = start + np.prod(metadata["shape"])
                    self.inner_problem.model._inputs[
                        metadata["step_input_var"]
                    ] = numpy_acc_stage_vec[start:end].reshape(metadata["shape"])
            if numpy_acc_stage_vec is not None:
                for quantity, metadata in self.quantity_metadata.items():
                    start = metadata["numpy_start_index"]
                    end = start + np.prod(metadata["shape"])
                    self.inner_problem.model._inputs[
                        metadata["accumulated_stage_var"]
                    ] = numpy_acc_stage_vec[start:end].reshape(metadata["shape"])


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
        self.need_call_solve_nonlinear = False

    def __call__(
        self,
        stage_perturbation: np.ndarray,
        stage_time: float,
        delta_t: float,
        butcher_diagonal_element: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.need_call_solve_nonlinear:
            self.inner_problem.model.run_solve_nonlinear()
            self.need_call_solve_nonlinear = False

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

    def linearize(
        self,
        inner_input_vec: om.DefaultVector = None,
        inner_output_vec: om.DefaultVector = None,
        numpy_old_state_vec: np.ndarray = None,
        numpy_acc_stage_vec: np.ndarray = None,
    ):
        # linearize always needs to be called, this checks whether we need to solve the nonlinear
        # system beforehand (e.g. due to manually changed inputs/outputs)
        if inner_input_vec is not None:
            self.inner_problem.model._inputs.set_vec(inner_input_vec)
            if inner_output_vec is not None:
                self.inner_problem.model._outputs.set_vec(inner_output_vec)
            else:
                self.need_call_solve_nonlinear = True
        elif numpy_old_state_vec is not None or numpy_acc_stage_vec is not None:
            self.need_call_solve_nonlinear = True
            if numpy_old_state_vec is not None:
                for quantity, metadata in self.quantity_metadata.items():
                    start = metadata["numpy_start_index"]
                    end = start + np.prod(metadata["shape"])
                    self.inner_problem.model._inputs[
                        metadata["step_input_var"]
                    ] = numpy_acc_stage_vec[start:end].reshape(metadata["shape"])
            if numpy_acc_stage_vec is not None:
                for quantity, metadata in self.quantity_metadata.items():
                    start = metadata["numpy_start_index"]
                    end = start + np.prod(metadata["shape"])
                    self.inner_problem.model._inputs[
                        metadata["accumulated_stage_var"]
                    ] = numpy_acc_stage_vec[start:end].reshape(metadata["shape"])
