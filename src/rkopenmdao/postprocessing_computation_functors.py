"""Wrapper classes to make a OpenMDAO problem used for postprocessing in the
RungeKuttaIntegrator usable with the Postprocessor class."""

import numpy as np
import openmdao.api as om

from .errors import PostprocessingError


class PostprocessingProblemComputeFunctor:
    """Wraps an OpenMDAO problem (specifically its models run_solve_nonlinear method)
    to a functor usable in the Postprocessor class."""

    def __init__(
        self,
        postprocessing_problem: om.Problem,
        quantity_metadata: dict,
        translation_metadata: dict,
        state_size: int,
        postproc_size: int,
    ):
        self.postprocessing_problem: om.Problem = postprocessing_problem
        self.quantity_metadata: dict = quantity_metadata
        self.translation_metadata: dict = translation_metadata
        self.state_size = state_size
        self.postproc_size = postproc_size

    def __call__(self, input_vector: np.ndarray) -> np.ndarray:
        self.fill_problem_data(input_vector)
        self.postprocessing_problem.model.run_solve_nonlinear()
        postproc_state = np.zeros(self.postproc_size)
        self.get_problem_data(postproc_state)
        return postproc_state

    def fill_problem_data(self, input_vector):
        """Write data into the internal nonlinear vectors of the owned problem."""
        _, outputs, _ = self.postprocessing_problem.model.get_nonlinear_vectors()
        for quantity, metadata in self.quantity_metadata.items():
            if (
                metadata["type"] == "time_integration"
                and self.translation_metadata[quantity]["postproc_input_var"]
                is not None
            ):
                start = metadata["start_index"]
                end = metadata["end_index"]
                outputs[
                    self.postprocessing_problem.model.get_source(
                        self.translation_metadata[quantity]["postproc_input_var"]
                    )
                ] = (input_vector[start:end].reshape(metadata["shape"]),)

    def get_problem_data(self, postproc_state: np.ndarray):
        """Extract data from the internal nonlinear vectors of the owned problem."""
        _, outputs, _ = self.postprocessing_problem.model.get_nonlinear_vectors()
        for quantity, metadata in self.quantity_metadata.items():
            if metadata["type"] == "postprocessing":
                start = metadata["start_index"]
                end = metadata["end_index"]
                postproc_state[start:end] = outputs[
                    self.translation_metadata[quantity]["postproc_output_var"]
                ].flatten()


class PostprocessingProblemComputeJacvecFunctor:
    """Wraps an openMDAO problem (specifically its compute_jacvec_problem function) to
    a functor usable in the Postprocessor class. Uses the 'fwd'-mode of said
    function."""

    def __init__(
        self,
        postprocessing_problem: om.Problem,
        quantity_metadata: dict,
        translation_metadata: dict,
        state_size: int,
        postproc_size: int,
        of_vars: list,
        wrt_vars: list,
    ):
        self.postprocessing_problem: om.Problem = postprocessing_problem
        self.quantity_metadata: dict = quantity_metadata
        self.translation_metadata: dict = translation_metadata
        self.state_size = state_size
        self.postproc_size = postproc_size
        self.of_vars = of_vars
        self.wrt_vars = wrt_vars

    def __call__(self, input_perturbation: np.ndarray) -> np.ndarray:
        self.postprocessing_problem.model.run_linearize()
        self.fill_problem_data(input_perturbation)

        try:
            self.postprocessing_problem.model.run_solve_linear("fwd")
        except TypeError:  # old openMDAO version had different interface
            self.postprocessing_problem.model.run_solve_linear(
                vec_names=["linear"], mode="fwd"
            )
        postproc_perturbations = np.zeros(self.postproc_size)
        self.get_problem_data(postproc_perturbations)
        return postproc_perturbations

    def fill_problem_data(self, input_vector):
        """Write data into the internal linear vectors of the owned problem."""
        (_, _, d_residuals) = self.postprocessing_problem.model.get_linear_vectors()
        d_residuals.asarray()[:] *= 0.0
        for quantity, metadata in self.quantity_metadata.items():
            if (
                metadata["type"] == "time_integration"
                and self.translation_metadata[quantity]["postproc_input_var"]
                is not None
            ):
                start = metadata["start_index"]
                end = metadata["end_index"]
                d_residuals[
                    self.postprocessing_problem.model.get_source(
                        self.translation_metadata[quantity]["postproc_input_var"]
                    )
                ] = -input_vector[start:end].reshape(metadata["shape"])

    def get_problem_data(self, postproc_perturbations: np.ndarray):
        """Extract data from the internal linear vectors of the owned problem."""
        _, d_outputs, _ = self.postprocessing_problem.model.get_linear_vectors()
        for quantity, metadata in self.quantity_metadata.items():
            if metadata["type"] == "postprocessing":
                start = metadata["start_index"]
                end = metadata["end_index"]
                postproc_perturbations[start:end] = d_outputs[
                    self.translation_metadata[quantity]["postproc_output_var"]
                ].flatten()

    def linearize(self, inputs=None, outputs=None):
        """Sets inputs and outputs of the owned problem to prepare for a different
        linearization point."""
        if inputs is not None and outputs is not None:
            (
                prob_inputs,
                prob_outputs,
                _,
            ) = self.postprocessing_problem.model.get_nonlinear_vectors()
            prob_inputs.set_vec(inputs)
            prob_outputs.set_vec(outputs)
        elif inputs is not None or outputs is not None:
            raise PostprocessingError(
                "Either both or none of inputs and outputs must be given."
            )


class PostprocessingProblemComputeTransposeJacvecFunctor:
    """Wraps an OpenMDAO problem (specifically its compute_jacvec_problem function) to
    a functor usable in the Postprocessor class. Uses the 'rev'-mode of said
    function."""

    def __init__(
        self,
        postprocessing_problem: om.Problem,
        quantity_metadata: dict,
        translation_metadata: dict,
        state_size: int,
        postproc_size: int,
        of_vars: list,
        wrt_vars: list,
    ):
        self.postprocessing_problem: om.Problem = postprocessing_problem
        self.quantity_metadata: dict = quantity_metadata
        self.translation_metadata: dict = translation_metadata
        self.state_size = state_size
        self.postproc_size = postproc_size
        self.of_vars = of_vars
        self.wrt_vars = wrt_vars

    def __call__(self, postproc_perturbations: np.ndarray) -> np.ndarray:
        self.postprocessing_problem.model.run_linearize()

        self.fill_problem_data(postproc_perturbations)
        try:
            self.postprocessing_problem.model.run_solve_linear("rev")
        except TypeError:  # old openMDAO version had different interface
            self.postprocessing_problem.model.run_solve_linear(
                vec_names=["linear"], mode="rev"
            )
        input_perturbations = np.zeros(self.state_size)
        self.get_problem_data(input_perturbations)
        return input_perturbations

    def fill_problem_data(self, postproc_perturbations):
        """Write data into the internal linear vectors of the owned problem."""
        _, d_outputs, _ = self.postprocessing_problem.model.get_linear_vectors()
        d_outputs.asarray()[:] *= 0
        for quantity, metadata in self.quantity_metadata.items():
            if metadata["type"] == "postprocessing":
                start = metadata["start_index"]
                end = metadata["end_index"]
                d_outputs[
                    self.translation_metadata[quantity]["postproc_output_var"]
                ] = -postproc_perturbations[start:end].reshape(metadata["shape"])

    def get_problem_data(self, input_perturbations: np.ndarray):
        """Extract data from the internal linear vectors of the owned problem."""
        _, _, d_residuals = self.postprocessing_problem.model.get_linear_vectors()
        for quantity, metadata in self.quantity_metadata.items():
            if (
                metadata["type"] == "time_integration"
                and self.translation_metadata[quantity]["postproc_input_var"]
                is not None
            ):
                start = metadata["start_index"]
                end = metadata["end_index"]
                input_perturbations[start:end] = d_residuals[
                    self.postprocessing_problem.model.get_source(
                        self.translation_metadata[quantity]["postproc_input_var"]
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
            ) = self.postprocessing_problem.model.get_nonlinear_vectors()
            prob_inputs.set_vec(inputs)
            prob_outputs.set_vec(outputs)
        elif inputs is not None or outputs is not None:
            raise PostprocessingError(
                "Either both or none of inputs and outputs must be given."
            )
