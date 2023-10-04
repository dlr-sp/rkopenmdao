"""Base class and default for the object to describe linear combinations to the Runge-Kutta-integrator."""

from typing import List, Union

import numpy as np


class FunctionalCoefficients:
    """Base class for the linear combination coefficients object."""

    def list_quantities(self) -> List[str]:
        """Returns the list of quantities this objects operates on."""
        raise NotImplementedError(
            "Method 'list_quantities' must be implemented in child class by user."
        )

    def get_coefficient(
        self, time_step: int, quantity: str
    ) -> Union[float, np.ndarray]:
        """Given a quantity and a time step, returns the fitting coefficient for the linear combination."""
        raise NotImplementedError(
            "Method 'get_coefficient' must be implemented in child class by user."
        )


class EmptyFunctionalCoefficients(FunctionalCoefficients):
    """
    Linear combination coefficients objects that has an empty quantity list and returns 0 for all times and quantities
    """

    def list_quantities(self) -> List[str]:
        return []

    def get_coefficient(
        self, time_step: int, quantity: str
    ) -> Union[float, np.ndarray]:
        return 0.0
