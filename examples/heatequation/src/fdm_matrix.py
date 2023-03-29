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

        new_vector = np.zeros_like(old_vector)
        for j in range(self.domain.n_y):
            for i in range(self.domain.n_x):
                index = self.domain.ij_to_index(i, j)
                new_vector[index] -= 2 * old_vector[index] / self.domain.delta_x**2
                if i == 0:
                    new_vector[index] += (
                        2 * old_vector[index + 1] / self.domain.delta_x**2
                    )
                elif i == self.domain.n_x - 1:
                    new_vector[index] += (
                        2 * old_vector[index - 1] / self.domain.delta_x**2
                    )
                else:
                    new_vector[index] += (
                        old_vector[index + 1] + old_vector[index - 1]
                    ) / self.domain.delta_x**2

                new_vector[index] -= 2 * old_vector[index] / self.domain.delta_y**2
                if j == 0:
                    new_vector[index] += (
                        2
                        * old_vector[index + self.domain.n_x]
                        / self.domain.delta_y**2
                    )
                elif j == self.domain.n_y - 1:
                    new_vector[index] += (
                        2
                        * old_vector[index - self.domain.n_x]
                        / self.domain.delta_y**2
                    )
                else:
                    new_vector[index] += (
                        old_vector[index + self.domain.n_x]
                        + old_vector[index - self.domain.n_x]
                    ) / self.domain.delta_y**2

        new_vector *= self.diffusivity
        return new_vector

    def mat_vec_prod_transpose(self, old_vector: np.ndarray) -> np.ndarray:
        new_vector = np.zeros_like(old_vector)

        for j in range(self.domain.n_y):
            for i in range(self.domain.n_x):
                index = self.domain.ij_to_index(i, j)
                new_vector[index] -= 2 * old_vector[index] / self.domain.delta_x**2
                if i == 0:
                    new_vector[index] += (
                        old_vector[index + 1] / self.domain.delta_x**2
                    )
                elif i == 1:
                    new_vector[index] += (
                        old_vector[index + 1] + 2 * old_vector[index - 1]
                    ) / (self.domain.delta_x**2)
                elif i == self.domain.n_x - 2:
                    new_vector[index] += (
                        2 * old_vector[index + 1] + old_vector[index - 1]
                    ) / (self.domain.delta_x**2)
                elif i == self.domain.n_x - 1:
                    new_vector[index] += (
                        old_vector[index - 1] / self.domain.delta_x**2
                    )
                else:
                    new_vector[index] += (
                        old_vector[index + 1] + old_vector[index - 1]
                    ) / self.domain.delta_x**2

                new_vector[index] -= 2 * old_vector[index] / self.domain.delta_y**2
                if j == 0:
                    new_vector[index] += (
                        old_vector[index + self.domain.n_x] / self.domain.delta_y**2
                    )
                elif j == 1:
                    new_vector[index] += (
                        old_vector[index + self.domain.n_x]
                        + 2 * old_vector[index - self.domain.n_x]
                    ) / self.domain.delta_y**2
                elif j == self.domain.n_y - 2:
                    new_vector[index] += (
                        2 * old_vector[index + self.domain.n_x]
                        + old_vector[index - self.domain.n_x]
                    ) / self.domain.delta_y**2
                elif j == self.domain.n_y - 1:
                    new_vector[index] += (
                        old_vector[index - self.domain.n_x] / self.domain.delta_y**2
                    )
                else:
                    new_vector[index] += (
                        old_vector[index + self.domain.n_x]
                        + old_vector[index - self.domain.n_x]
                    ) / self.domain.delta_y**2

        new_vector *= self.diffusivity
        return new_vector

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
