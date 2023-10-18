from typing import Callable
import numpy as np

from .domain import Domain
from .boundary import BoundaryCondition


class InhomogeneityVector:
    """Class to present the discretized inhomogeneity of a heat equation (including inhomogeneity from boundary)."""

    def __init__(
        self,
        domain: Domain,
        diffusivity: float,
        inhomogeneity_func: Callable[[float, float, float], float],
        boundary_condition: BoundaryCondition,
    ) -> None:
        self.domain = domain
        self.diffusivity = diffusivity
        self.pde_inhomogeneity = np.zeros(self.domain.n_x * self.domain.n_y)
        self.self_updating_boundary_inhomogeneity = np.zeros(
            self.domain.n_x * self.domain.n_y
        )
        self.user_updating_boundary_inhomogeneity = np.zeros(
            (self.domain.n_x * self.domain.n_y, 4)
        )
        self.inhomogeneity_func = inhomogeneity_func
        self.boundary_condition = boundary_condition

    def add_vector(self, vector: np.ndarray) -> np.ndarray:
        """Adds all the inhomogeneity to a vector."""
        if vector.size != self.pde_inhomogeneity.size:
            raise AssertionError("Vector sizes don't")
        return (
            vector
            + self.pde_inhomogeneity
            + self.self_updating_boundary_inhomogeneity
            + np.sum(self.user_updating_boundary_inhomogeneity, axis=1)
        )

    def return_vector(self) -> np.ndarray:
        """Returns the combined inhomogeneity."""
        return (
            self.pde_inhomogeneity
            + self.self_updating_boundary_inhomogeneity
            + np.sum(self.user_updating_boundary_inhomogeneity, axis=1)
        )

    def update_pde_inhomogeneity(self, time: float) -> None:
        """Updates non-boundary inhomogeneity to the current time"""
        for j in range(self.domain.n_y):
            for i in range(self.domain.n_x):
                index = self.domain.ij_to_index(i, j)
                coordinates = self.domain.ij_coordinates(i, j)
                self.pde_inhomogeneity[index] = self.inhomogeneity_func(
                    time, coordinates[0], coordinates[1]
                )

    def update_boundary_inhomogeneity(
        self, time=None, upper=None, lower=None, left=None, right=None
    ) -> None:
        "Updates boundary inhomogeneity to the current time (or by user)"
        update_kinds = self.boundary_condition.boundary_update_kind()
        if time is not None:
            self.self_updating_boundary_inhomogeneity = np.zeros_like(
                self.self_updating_boundary_inhomogeneity
            )
            for segment, kind in update_kinds.items():
                if kind == "self_updating":
                    boundary_coordinates = self.domain.boundary_coordinates(segment)
                    segment_boundary_inhomogeneity = (
                        self.boundary_condition.self_update(
                            segment, time, boundary_coordinates
                        )
                    )
                    boundary_indices = self.domain.boundary_indices(segment)
                    self.self_updating_boundary_inhomogeneity[boundary_indices] += (
                        2
                        * self.diffusivity
                        * segment_boundary_inhomogeneity
                        / (
                            self.domain.delta_x
                            if segment in ("right", "left")
                            else self.domain.delta_y
                        )
                    )

        for segment, kind in update_kinds.items():
            boundary_slice_index = -1
            boundary_indices = self.domain.boundary_indices(segment)
            segment_boundary_inhomogeneity = np.zeros(0)
            update = False
            if segment == "upper" and upper is not None:
                segment_boundary_inhomogeneity = upper
                update = True
                boundary_slice_index = 0
            elif segment == "lower" and lower is not None:
                segment_boundary_inhomogeneity = lower
                update = True
                boundary_slice_index = 1
            elif segment == "left" and left is not None:
                segment_boundary_inhomogeneity = left
                update = True
                boundary_slice_index = 2
            elif segment == "right" and right is not None:
                segment_boundary_inhomogeneity = right
                update = True
                boundary_slice_index = 3
            if update:
                self.user_updating_boundary_inhomogeneity[
                    boundary_indices, boundary_slice_index
                ] = (
                    2
                    * self.diffusivity
                    * segment_boundary_inhomogeneity
                    / (
                        self.domain.delta_x
                        if segment in ("right", "left")
                        else self.domain.delta_y
                    )
                )
