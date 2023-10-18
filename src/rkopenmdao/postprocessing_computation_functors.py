"""
Wrapper classes to make a openMDAO problem used for postprocessing in the Runge-Kutta integrator usable with the
Postprocessor class.
"""

import numpy as np
import openmdao.api as om

from .errors import PostprocessingError


class PostprocessingProblemComputeFunctor:
    """
    Wraps an openMDAO problem (specifically its models run_solve_nonlinear method) to a functor usable in the
    Postprocessor class.
    """

    def __init__(
        self,
        postprocessing_problem: om.Problem,
        quantity_metadata: dict,
        state_size: int,
        postproc_size: int,
    ):
        self.postprocessing_problem: om.Problem = postprocessing_problem
        self.quantity_metadata: dict = quantity_metadata
        self.state_size = state_size
        self.postproc_size = postproc_size

    def __call__(self, input_vector: np.ndarray) -> np.ndarray:
        self.fill_inputs(input_vector)
        self.postprocessing_problem.model.run_solve_nonlinear()
        postproc_state = np.zeros(self.postproc_size)
        self.get_outputs(postproc_state)
        return postproc_state

    def fill_inputs(self, input_vector):
        """Write data into the input vectors of the owned problem."""
        for metadata in self.quantity_metadata.values():
            if (
                metadata["type"] == "time_integration"
                and "postproc_input_var" in metadata
            ):
                start = metadata["numpy_start_index"]
                end = metadata["numpy_end_index"]
                self.postprocessing_problem.set_val(
                    metadata["postproc_input_var"],
                    input_vector[start:end].reshape(metadata["shape"]),
                )

    def get_outputs(self, postproc_state: np.ndarray):
        """Extract data from the output vectors of the owned problem."""
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "postprocessing":
                start = metadata["numpy_postproc_start_index"]
                end = metadata["numpy_postproc_end_index"]
                postproc_state[start:end] = self.postprocessing_problem.get_val(
                    metadata["postproc_output_var"], get_remote=False
                ).flatten()


class PostprocessingProblemComputeJacvecFunctor:
    """
    Wraps an openMDAO problem (specifically its compute_jacvec_problem function) to a functor usable in the
    Postprocessor class. Uses the 'fwd'-mode of said function.
    """

    def __init__(
        self,
        postprocessing_problem: om.Problem,
        quantity_metadata: dict,
        state_size: int,
        postproc_size: int,
        of_vars: list,
        wrt_vars: list,
    ):
        self.postprocessing_problem: om.Problem = postprocessing_problem
        self.quantity_metadata: dict = quantity_metadata
        self.state_size = state_size
        self.postproc_size = postproc_size
        self.of_vars = of_vars
        self.wrt_vars = wrt_vars

    def __call__(self, input_perturbation: np.ndarray) -> np.ndarray:
        self.postprocessing_problem.model.run_linearize()
        seed = {}
        self.fill_seed(input_perturbation, seed)
        jvp = self.postprocessing_problem.compute_jacvec_product(
            of=self.of_vars, wrt=self.wrt_vars, mode="fwd", seed=seed
        )
        postproc_perturbations = np.zeros(self.postproc_size)
        self.extract_jvp(jvp, postproc_perturbations)
        return postproc_perturbations

    def fill_seed(self, input_vector, seed):
        """Write data into a seed to be used by the compute_jacvec_product function by the owned problem."""
        for metadata in self.quantity_metadata.values():
            if (
                metadata["type"] == "time_integration"
                and "postproc_input_var" in metadata
            ):
                start = metadata["numpy_start_index"]
                end = metadata["numpy_end_index"]
                seed[metadata["postproc_input_var"]] = input_vector[start:end].reshape(
                    metadata["shape"]
                )

    def extract_jvp(self, jvp: dict, postproc_perturbations: np.ndarray):
        """Extracts data from the result of the owned problems compute_jacvec_product functions."""
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "postprocessing":
                start = metadata["numpy_postproc_start_index"]
                end = metadata["numpy_postproc_end_index"]
                postproc_perturbations[start:end] = jvp[
                    metadata["postproc_output_var"]
                ].flatten()

    def linearize(self, inputs=None, outputs=None):
        """Sets inputs and outputs of the owned problem to prepare for a different linearization point."""
        if inputs is not None and outputs is not None:
            self.postprocessing_problem.model._inputs = inputs
            self.postprocessing_problem.model._outputs = outputs
        elif inputs is not None or outputs is not None:
            raise PostprocessingError(
                "Either both or none of inputs and outputs must be given."
            )


class PostprocessingProblemComputeTransposeJacvecFunctor:
    """
    Wraps an openMDAO problem (specifically its compute_jacvec_problem function) to a functor usable in the
    Postprocessor class. Uses the 'rev'-mode of said function.
    """

    def __init__(
        self,
        postprocessing_problem: om.Problem,
        quantity_metadata: dict,
        state_size: int,
        postproc_size: int,
        of_vars: list,
        wrt_vars: list,
    ):
        self.postprocessing_problem: om.Problem = postprocessing_problem
        self.quantity_metadata: dict = quantity_metadata
        self.state_size = state_size
        self.postproc_size = postproc_size
        self.of_vars = of_vars
        self.wrt_vars = wrt_vars

    def __call__(self, postproc_perturbations: np.ndarray) -> np.ndarray:
        self.postprocessing_problem.model.run_linearize()
        seed = {}
        self.fill_seed(postproc_perturbations, seed)
        jvp = self.postprocessing_problem.compute_jacvec_product(
            of=self.of_vars, wrt=self.wrt_vars, mode="rev", seed=seed
        )
        input_perturbations = np.zeros(self.state_size)
        self.extract_jvp(jvp, input_perturbations)
        return input_perturbations

    def fill_seed(self, postproc_perturbations, seed):
        """Write data into a seed to be used by the compute_jacvec_product function by the owned problem."""
        for metadata in self.quantity_metadata.values():
            if metadata["type"] == "postprocessing":
                start = metadata["numpy_postproc_start_index"]
                end = metadata["numpy_postproc_end_index"]
                seed[metadata["postproc_output_var"]] = postproc_perturbations[
                    start:end
                ].reshape(metadata["shape"])

    def extract_jvp(self, jvp: dict, input_perturbations: np.ndarray):
        """Extracts data from the result of the owned problems compute_jacvec_product functions."""
        for metadata in self.quantity_metadata.values():
            if (
                metadata["type"] == "time_integration"
                and "postproc_input_var" in metadata
            ):
                start = metadata["numpy_start_index"]
                end = metadata["numpy_end_index"]
                input_perturbations[start:end] = jvp[
                    metadata["postproc_input_var"]
                ].flatten()

    def linearize(self, inputs=None, outputs=None):
        """Sets inputs and outputs of the owned problem to prepare for a different linearization point."""
        if inputs is not None and outputs is not None:
            self.postprocessing_problem.model._inputs = inputs
            self.postprocessing_problem.model._outputs = outputs
        elif inputs is not None or outputs is not None:
            raise PostprocessingError(
                "Either both or none of inputs and outputs must be given."
            )
