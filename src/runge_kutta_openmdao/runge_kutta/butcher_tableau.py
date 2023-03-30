import numpy as np


class ButcherTableau:
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
        return self.stages

    def is_explicit(self) -> bool:
        for i in range(self.stages):
            for j in range(i, self.stages):
                if self.butcher_matrix[i, j] != 0.0:
                    return False
        return True

    def is_diagonally_implicit(self) -> bool:
        if self.is_explicit():
            return False
        for i in range(self.stages):
            for j in range(i + 1, self.stages):
                if self.butcher_matrix[i, j] != 0.0:
                    return False
        return True

    def is_implicit(self) -> bool:
        if self.is_explicit() or self.is_diagonally_implicit():
            return False
        return True


class AdaptiveButcherTableau(ButcherTableau):
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
