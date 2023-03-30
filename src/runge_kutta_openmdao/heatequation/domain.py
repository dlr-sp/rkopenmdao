"""
Describes the domain of the heat equation, and its discretization
"""

import numpy as np


class Domain:
    """
    Class for the domain of the heat equation
    """

    def __init__(
        self, x_range: np.ndarray, y_range: np.ndarray, n_x: int, n_y: int
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.n_x = n_x
        self.n_y = n_y
        self.delta_x = (x_range[1] - x_range[0]) / (n_x - 1.0)
        self.delta_y = (y_range[1] - y_range[0]) / (n_y - 1.0)

    def boundary_coordinates(self, segment: str) -> np.ndarray:
        """
        Returns the discretized boundary coordinates of the chosen segment,
        where segment can either be "lower", "upper", "left", "right"
        """
        coordinates = None
        if segment in ("left", "right"):
            coordinates = np.zeros((self.n_y, 2))
            for i in range(self.n_y):
                coordinates[i, :] = self.ij_coordinates(
                    0 if segment == "left" else self.n_x - 1, i
                )
        if segment in ("lower", "upper"):
            coordinates = np.zeros((self.n_x, 2))
            for i in range(self.n_x):
                coordinates[i, :] = self.ij_coordinates(
                    i, 0 if segment == "lower" else self.n_y - 1
                )
        return coordinates

    def boundary_indices(self, segment: str) -> np.ndarray:
        """
        Calculates the indices of the chosen boundary segment, starting from left/lower.
        """
        indices = None
        if segment in ("left", "right"):
            indices = np.zeros(self.n_y, dtype=int)
            for i in range(self.n_y):
                indices[i] = self.ij_to_index(
                    0 if segment == "left" else self.n_x - 1, i
                )
        if segment in ("lower", "upper"):
            indices = np.zeros(self.n_x, dtype=int)
            for i in range(self.n_x):
                indices[i] = self.ij_to_index(
                    i, 0 if segment == "lower" else self.n_y - 1
                )
        return indices

    def ij_coordinates(self, i: int, j: int) -> np.ndarray:
        """
        Return the coordinates of the i-th point in x-, and and j-th point in y-direction
        """
        coordinate = np.zeros(2)
        coordinate[0] = self.x_range[0] + i * self.delta_x
        coordinate[1] = self.y_range[0] + j * self.delta_y
        return coordinate

    def ij_to_index(self, i: int, j: int) -> int:
        """
        Computes the index in the global numbering scheme from the i-th index in x-,
        and j-th index in y-direction
        """
        return i + self.n_x * j

    def index_to_ij(self, index: int) -> np.ndarray:
        """
        Computes from an index in the global numbering scheme the indices in x- and y-direction.
        """
        ij_arr = np.zeros(2, dtype=int)
        ij_arr[1] = index // self.n_x
        ij_arr[0] = index - ij_arr[1] * self.n_x
        return ij_arr

