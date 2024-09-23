"""FunctionalCoefficients that are useful for tests, but not for real usage."""

from __future__ import annotations

import numpy as np

from rkopenmdao.functional_coefficients import (
    FunctionalCoefficients,
)


class FifthStepOfQuantity(FunctionalCoefficients):
    """A functional that only takes the state at time step five of the given
    quantity."""

    def __init__(self, quantity: str):
        self._quantity_list = [quantity]

    def list_quantities(self) -> list[str]:
        return self._quantity_list

    def get_coefficient(self, time_step: int, quantity: str) -> float | np.ndarray:
        if quantity == self._quantity_list[0] and time_step == 5:
            return 1.0
        else:
            return 0.0
