# pylint: disable=missing-module-docstring
from typing import Callable
import numpy as np


# TODO: either block non-None nonCallables, or create a constant lambda out of it


class BoundaryCondition:
    """A class for the boundary management of the heatequation class"""

    def __init__(
        self,
        upper=None,
        lower=None,
        left=None,
        right=None,
    ):
        self.upper = upper
        self.lower = lower
        self.left = left
        self.right = right

    def boundary_update_kind(self) -> dict:
        """Returns a dict containing the update kinds of the different boundary
        segments."""
        boundary_types_dict = {}
        if self.upper is None:
            boundary_types_dict["upper"] = "user_updating"
        else:
            boundary_types_dict["upper"] = "self_updating"

        if self.lower is None:
            boundary_types_dict["lower"] = "user_updating"
        else:
            boundary_types_dict["lower"] = "self_updating"

        if self.left is None:
            boundary_types_dict["left"] = "user_updating"
        else:
            boundary_types_dict["left"] = "self_updating"

        if self.right is None:
            boundary_types_dict["right"] = "user_updating"
        else:
            boundary_types_dict["right"] = "self_updating"

        return boundary_types_dict

    def get_function(self, segment: str) -> Callable[[float, float, float], float]:
        """Returns the updater of the requested boundary segment."""
        if segment == "upper":
            return self.upper
        elif segment == "lower":
            return self.lower
        elif segment == "left":
            return self.left
        elif segment == "right":
            return self.right
        else:
            raise ValueError(
                "Segment is not one of 'upper', 'lower', 'left' or 'right'"
            )

    def self_update(
        self, segment: str, time: float, coordinates: np.ndarray
    ) -> np.ndarray:
        """Returns updated values at the boundary for the requested segment, time and
        coordinates."""
        result = np.zeros(coordinates.shape[0])
        for i in range(result.size):
            result[i] = self.get_function(segment)(
                time, coordinates[i, 0], coordinates[i, 1]
            )
        return result
