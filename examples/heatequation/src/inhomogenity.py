from typing import Callable
import numpy as np

from .domain import Domain
from .boundary import BoundaryCondition


class InhomogenityVector:
    def __init__(
        self,
        domain: Domain,
        diffusivity: float,
        inhomogenity_func: Callable[[float, float, float], float],
        boundary_condition: BoundaryCondition,
    ) -> None:
        self.domain = domain
        self.diffusivity = diffusivity
        self.pde_inhomogenity = np.zeros(self.domain.n_x * self.domain.n_y)
        self.self_updating_boundary_inhomogenity = np.zeros(
            self.domain.n_x * self.domain.n_y
        )
        self.user_updating_boundary_inhomogenity = np.zeros(
            (self.domain.n_x * self.domain.n_y, 4)
        )
        self.inhomogenity_func = inhomogenity_func
        self.boundary_condition = boundary_condition

    def add_vector(self, vector: np.ndarray) -> np.ndarray:
        if vector.size != self.pde_inhomogenity.size:
            raise AssertionError("Vector sizes don't")
        return (
            vector
            + self.pde_inhomogenity
            + self.self_updating_boundary_inhomogenity
            + np.sum(self.user_updating_boundary_inhomogenity, axis=1)
        )

    def return_vector(self) -> np.ndarray:
        return (
            self.pde_inhomogenity
            + self.self_updating_boundary_inhomogenity
            + np.sum(self.user_updating_boundary_inhomogenity, axis=1)
        )

    def update_pde_inhomogenity(self, time: float) -> None:
        for j in range(self.domain.n_y):
            for i in range(self.domain.n_x):
                index = self.domain.ij_to_index(i, j)
                coordinates = self.domain.ij_coordinates(i, j)
                self.pde_inhomogenity[index] = self.inhomogenity_func(
                    time, coordinates[0], coordinates[1]
                )

    def update_boundary_inhomogenity(
        self, time=None, upper=None, lower=None, left=None, right=None
    ) -> None:
        update_kinds = self.boundary_condition.boundary_update_kind()
        if time is not None:
            self.self_updating_boundary_inhomogenity = np.zeros_like(
                self.self_updating_boundary_inhomogenity
            )
            for segment, kind in update_kinds.items():
                if kind == "self_updating":
                    boundary_coordinates = self.domain.boundary_coordinates(segment)
                    segment_boundary_inhomogenity = self.boundary_condition.self_update(
                        segment, time, boundary_coordinates
                    )
                    boundary_indices = self.domain.boundary_indices(segment)
                    for i in range(segment_boundary_inhomogenity.size):
                        self.self_updating_boundary_inhomogenity[
                            boundary_indices[i]
                        ] += (
                            2
                            * self.diffusivity
                            * segment_boundary_inhomogenity[i]
                            / (
                                self.domain.delta_x
                                if segment in ("right", "left")
                                else self.domain.delta_y
                            )
                        )

        for segment, kind in update_kinds.items():
            boundary_slice_index = -1
            boundary_indices = self.domain.boundary_indices(segment)
            segment_boundary_inhomogenity = np.zeros(0)
            update = False
            if segment == "upper" and upper is not None:
                segment_boundary_inhomogenity = upper
                update = True
                boundary_slice_index = 0
            elif segment == "lower" and lower is not None:
                segment_boundary_inhomogenity = lower
                update = True
                boundary_slice_index = 1
            elif segment == "left" and left is not None:
                segment_boundary_inhomogenity = left
                update = True
                boundary_slice_index = 2
            elif segment == "right" and right is not None:
                segment_boundary_inhomogenity = right
                update = True
                boundary_slice_index = 3
            if update:
                for i in range(segment_boundary_inhomogenity.size):
                    self.user_updating_boundary_inhomogenity[
                        boundary_indices[i], boundary_slice_index
                    ] = (
                        2
                        * self.diffusivity
                        * segment_boundary_inhomogenity[i]
                        / (
                            self.domain.delta_x
                            if segment in ("right", "left")
                            else self.domain.delta_y
                        )
                    )
                # print(
                #     self.user_updating_boundary_inhomogenity[
                #         boundary_indices, boundary_slice_index
                #     ]
                # )
                # print(
                #     2
                #     * self.diffusivity
                #     * segment_boundary_inhomogenity
                #     / (
                #         self.domain.delta_x
                #         if segment in ("right", "left")
                #         else self.domain.delta_y
                #     )
                # )
