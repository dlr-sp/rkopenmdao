from typing import List, Union

import numpy as np

from runge_kutta_openmdao.runge_kutta.functional_coefficients import (
    FunctionalCoefficients,
)
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl


class AverageCoefficients(FunctionalCoefficients):
    def __init__(
        self, integration_control: IntegrationControl, quantity_list: List[str]
    ):
        self._quantity_list: List[str] = quantity_list
        self._integration_control: IntegrationControl = integration_control

    def list_quantities(self) -> List[str]:
        return self._quantity_list

    def get_coefficient(
        self, time_step: int, quantity: str
    ) -> Union[float, np.ndarray]:
        return (self._integration_control.num_steps + 1) ** -1


class CompositeTrapezoidalCoefficients(FunctionalCoefficients):
    def __init__(
        self, integration_control: IntegrationControl, quantity_list: List[str]
    ):
        self._quantity_list: List[str] = quantity_list
        self._integration_control: IntegrationControl = integration_control

    def list_quantities(self) -> List[str]:
        return self._quantity_list

    def get_coefficient(
        self, time_step: int, quantity: str
    ) -> Union[float, np.ndarray]:
        if time_step == 0 or time_step == self._integration_control.num_steps:
            return 0.5 * self._integration_control.delta_t
        else:
            return self._integration_control.delta_t


class FifthStepOfQuantity(FunctionalCoefficients):
    def __init__(self, quantity: str):
        self._quantity_list = [quantity]

    def list_quantities(self) -> List[str]:
        return self._quantity_list

    def get_coefficient(
        self, time_step: int, quantity: str
    ) -> Union[float, np.ndarray]:
        if quantity == self._quantity_list[0] and time_step == 5:
            return 1.0
        else:
            return 0.0
