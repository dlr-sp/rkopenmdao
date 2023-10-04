from typing import Callable
import numpy as np


class Postprocessor:
    """
    Wrapper to make the postprocessing in the Runge-Kutta-integrator more abstract.
    """

    def __init__(
        self,
        postprocessing_computation_functor: Callable[[np.ndarray], np.ndarray],
        postprocessing_computation_functor_jacvec: Callable[[np.ndarray], np.ndarray],
        postprocessing_computation_functor_jacvec_transposed: Callable[
            [np.ndarray], np.ndarray
        ],
    ):
        self.postprocessing_computation_functor = postprocessing_computation_functor
        self.postprocessing_computation_functor_jacvec = (
            postprocessing_computation_functor_jacvec
        )
        self.postprocessing_computation_functor_jacvec_transposed = (
            postprocessing_computation_functor_jacvec_transposed
        )

    def postprocess(self, input_data: np.ndarray) -> np.ndarray:
        """Applies the postprocessing_computation_functor on the input_data."""
        return self.postprocessing_computation_functor(input_data)

    def postprocess_jacvec(
        self, input_perturbation, **linearization_args
    ) -> np.ndarray:
        """
        Applies the matrix-vector-product with the jacobian of the postprocessing, given by the
        postprocessing_computation_functor_jacvec, to the input_perturbation. Also does some linearization beforehand,
        if applicable.
        """
        if hasattr(self.postprocessing_computation_functor_jacvec, "linearize"):
            self.postprocessing_computation_functor_jacvec.linearize(
                **linearization_args
            )
        return self.postprocessing_computation_functor_jacvec(input_perturbation)

    def postprocess_jacvec_transposed(
        self, output_perturbation, **linearization_args
    ) -> np.ndarray:
        """
        Applies the matrix-vector-product with the transposed jacobian of the postprocessing, given by the
        postprocessing_computation_functor_jacvec, to the input_perturbation. Also does some linearization beforehand,
        if applicable.
        """
        if hasattr(self.postprocessing_computation_functor_jacvec, "linearize"):
            self.postprocessing_computation_functor_jacvec.linearize(
                **linearization_args
            )
        return self.postprocessing_computation_functor_jacvec_transposed(
            output_perturbation
        )
