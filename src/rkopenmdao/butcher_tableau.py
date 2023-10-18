"""Contains classes for the representation of Butcher tableaux."""
import numpy as np


class ButcherTableau:
    """
    Representation of a Butcher tableau for Runge-Kutta methods.
    """

    def __init__(
        self,
        butcher_matrix: np.ndarray,
        butcher_weight_vector: np.ndarray,
        butcher_time_stages: np.ndarray,
    ):
        self.butcher_matrix = butcher_matrix
        self.butcher_weight_vector = butcher_weight_vector
        self.butcher_time_stages = butcher_time_stages
        self.stages = butcher_weight_vector.size
        if (
            self.butcher_matrix.shape != (self.stages, self.stages)
            or self.butcher_weight_vector.shape != self.butcher_time_stages.shape
        ):
            raise AssertionError("Sizes of matrix and vectors doesn't match")

    def number_of_stages(self):
        """Returns the number of stages of the tableau"""
        return self.stages

    def is_explicit(self) -> bool:
        """Checks whether this tableau is explicit or not."""
        for i in range(self.stages):
            for j in range(i, self.stages):
                if self.butcher_matrix[i, j] != 0.0:
                    return False
        return True

    def is_diagonally_implicit(self) -> bool:
        """
        Checks whether this tableau is diagonally implicit or not. This requires at least one non-zero diagonal element
        in the Butcher matrix, i.e. we don't count explicit schemes as diagonally implicit.
        """
        if self.is_explicit():
            return False
        for i in range(self.stages):
            for j in range(i + 1, self.stages):
                if self.butcher_matrix[i, j] != 0.0:
                    return False
        return True

    def is_implicit(self) -> bool:
        """
        Checks whether this tableau is fully implicit or not.
        """
        if self.is_explicit() or self.is_diagonally_implicit():
            return False
        return True


# TODO: is currently not used, need to re-add adaptive step size control at some point
# (but this currently clashes with checkpointing)
class AdaptiveButcherTableau(ButcherTableau):
    """
    Representation of a Butcher tableau with internal error control (i.e. has a second weight vector for representing a
    second scheme with lower order).
    """

    def __init__(
        self,
        butcher_matrix: np.ndarray,
        butcher_weight_vector: np.ndarray,
        butcher_adaptive_weights: np.ndarray,
        butcher_time_stages: np.ndarray,
    ):
        super().__init__(butcher_matrix, butcher_weight_vector, butcher_time_stages)
        self.butcher_adaptive_weights = butcher_adaptive_weights
        if self.butcher_adaptive_weights.shape != self.butcher_weight_vector.shape:
            raise AssertionError("Size of adaptive stage doesn't match.")
