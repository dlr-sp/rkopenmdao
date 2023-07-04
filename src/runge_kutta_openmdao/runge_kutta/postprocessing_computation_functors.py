import numpy as np
import openmdao.api as om


class PostprocessingProblemComputeFunctor:
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
        for quantity, metadata in self.quantity_metadata.items():
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
        for quantity, metadata in self.quantity_metadata.items():
            if metadata["type"] == "postprocessing":
                start = metadata["numpy_postproc_start_index"]
                end = metadata["numpy_postproc_end_index"]
                postproc_state[start:end] = self.postprocessing_problem.get_val(
                    metadata["postproc_output_var"], get_remote=False
                ).flatten()


class PostprocessingProblemComputeJacvecFunctor:
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
        for quantity, metadata in self.quantity_metadata.items():
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
        for quantity, metadata in self.quantity_metadata.items():
            if metadata["type"] == "postprocessing":
                start = metadata["numpy_postproc_start_index"]
                end = metadata["numpy_postproc_end_index"]
                postproc_perturbations[start:end] = jvp[
                    metadata["postproc_output_var"]
                ].flatten()

    def linearize(self, inputs=None, outputs=None):
        if inputs is not None and outputs is not None:
            self.postprocessing_problem.model._inputs = inputs
            self.postprocessing_problem.model._outputs = outputs
        elif inputs is not None or outputs is not None:
            # TODO: raise actual error
            print("Error")


class PostprocessingProblemComputeTransposeJacvecFunctor:
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
        for quantity, metadata in self.quantity_metadata.items():
            if metadata["type"] == "postprocessing":
                start = metadata["numpy_postproc_start_index"]
                end = metadata["numpy_postproc_end_index"]
                seed[metadata["postproc_output_var"]] = postproc_perturbations[
                    start:end
                ].reshape(metadata["shape"])

    def extract_jvp(self, jvp: dict, input_perturbations: np.ndarray):
        for quantity, metadata in self.quantity_metadata.items():
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
        if inputs is not None and outputs is not None:
            self.postprocessing_problem.model._inputs = inputs
            self.postprocessing_problem.model._outputs = outputs
        elif inputs is not None or outputs is not None:
            # TODO: raise actual error
            print("Error")
