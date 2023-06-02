"""
Multiplies the fdm-matrix for the Heat equation with a vector.
The matrix isn't assembled for that, it happens matrix free.
"""
import numpy as np

from .domain import Domain


class FdmMatrix:
    def __init__(self, domain: Domain, diffusivity: float) -> None:
        self.domain = domain

        self.diffusivity = diffusivity
        self.vector_size = self.domain.n_x * self.domain.n_y

    def mat_vec_prod(self, old_vector: np.ndarray) -> np.ndarray:
        if old_vector.size != self.vector_size:
            raise AssertionError("Vector has the wrong dimension")

        new_vector = np.zeros((self.domain.n_y, self.domain.n_x))
        old_vector_reshape = old_vector.reshape((self.domain.n_y, self.domain.n_x))

        # diagonal entries
        new_vector -= 2 * old_vector_reshape / self.domain.delta_x**2
        new_vector -= 2 * old_vector_reshape / self.domain.delta_y**2

        # direct sup/superdiagonal coming from the x-direction
        new_vector[:, 0] += 2 * old_vector_reshape[:, 1] / self.domain.delta_x**2
        new_vector[:, self.domain.n_x - 1] += (
            2 * old_vector_reshape[:, self.domain.n_x - 2] / self.domain.delta_x**2
        )
        new_vector[:, 1 : self.domain.n_x - 1] += (
            old_vector_reshape[:, : self.domain.n_x - 2] + old_vector_reshape[:, 2:]
        ) / self.domain.delta_x**2

        # sup/superdiagonal coming from the y-direction
        new_vector[0, :] += 2 * old_vector_reshape[1, :] / self.domain.delta_y**2
        new_vector[self.domain.n_y - 1, :] += (
            2 * old_vector_reshape[self.domain.n_y - 2, :] / self.domain.delta_y**2
        )
        new_vector[1 : self.domain.n_y - 1, :] += (
            old_vector_reshape[: self.domain.n_y - 2, :] + old_vector_reshape[2:, :]
        ) / self.domain.delta_y**2

        new_vector *= self.diffusivity
        return new_vector.reshape(self.vector_size)

    def mat_vec_prod_transpose(self, old_vector: np.ndarray) -> np.ndarray:
        if old_vector.size != self.vector_size:
            print(
                "old_vec_size: ",
                old_vector.size,
                "expected_vec_size: ",
                self.vector_size,
            )
            raise AssertionError("Vector has the wrong dimension")

        new_vector = np.zeros((self.domain.n_y, self.domain.n_x))
        old_vector_reshape = old_vector.reshape((self.domain.n_y, self.domain.n_x))

        # diagonal entries
        new_vector -= 2 * old_vector_reshape / self.domain.delta_x**2
        new_vector -= 2 * old_vector_reshape / self.domain.delta_y**2

        # direct sup/superdiagonal coming from the x-direction
        new_vector[:, 0] += old_vector_reshape[:, 1] / self.domain.delta_x**2
        new_vector[:, 1] += (old_vector_reshape[:, 0]) / self.domain.delta_x**2
        new_vector[:, self.domain.n_x - 2] += (
            old_vector_reshape[:, self.domain.n_x - 1]
        ) / self.domain.delta_x**2

        new_vector[:, self.domain.n_x - 1] += (
            old_vector_reshape[:, self.domain.n_x - 2] / self.domain.delta_x**2
        )
        new_vector[:, 1 : self.domain.n_x - 1] += (
            old_vector_reshape[:, : self.domain.n_x - 2]
            + old_vector_reshape[:, 2 : self.domain.n_x]
        ) / self.domain.delta_x**2

        # sup/superdiagonal coming from the y-direction
        new_vector[0, :] += old_vector_reshape[1, :] / self.domain.delta_y**2
        new_vector[1, :] += (old_vector_reshape[0, :]) / self.domain.delta_y**2
        new_vector[self.domain.n_y - 2, :] += (
            old_vector_reshape[self.domain.n_y - 1, :]
        ) / self.domain.delta_y**2
        new_vector[self.domain.n_y - 1, :] += (
            old_vector_reshape[self.domain.n_y - 2, :] / self.domain.delta_y**2
        )

        new_vector[1 : self.domain.n_y - 1, :] += (
            old_vector_reshape[: self.domain.n_y - 2, :]
            + old_vector_reshape[2 : self.domain.n_y, :]
        ) / self.domain.delta_y**2

        new_vector *= self.diffusivity
        return new_vector.reshape(self.vector_size)

    def mat_mat_prod(self, old_matrix: np.ndarray) -> np.ndarray:
        if old_matrix.shape[0] != self.vector_size:
            raise AssertionError("Matrix has the wrong dimension")

        new_matrix = np.zeros_like(old_matrix)
        for i in range(new_matrix.shape[1]):
            new_matrix[:, i] = self.mat_vec_prod(old_matrix[:, i])
        return new_matrix

    def mat_mat_prod_transpose(self, old_matrix: np.ndarray) -> np.ndarray:
        new_matrix = np.zeros_like(old_matrix)
        for i in range(new_matrix.shape[1]):
            new_matrix[:, i] = self.mat_vec_prod_transpose(old_matrix[:, i])
        return new_matrix
