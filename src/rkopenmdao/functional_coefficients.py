"""Base class and default for the object to describe linear combinations to the
RungeKuttaIntegrator."""

from __future__ import annotations
import numpy as np

from .integration_control import IntegrationControl


class FunctionalCoefficients:
    """Base class for the functional coefficients object."""

    def list_quantities(self) -> list[str]:
        """Returns the list of quantities this objects operates on."""
        raise NotImplementedError(
            "Method 'list_quantities' must be implemented in child class by user."
        )

    def get_coefficient(self, time_step: int, quantity: str) -> float | np.ndarray:
        """Given a quantity and a time step, returns the fitting coefficient for the
        linear combination."""
        raise NotImplementedError(
            "Method 'get_coefficient' must be implemented in child class by user."
        )


class EmptyFunctionalCoefficients(FunctionalCoefficients):
    """
    Functional coefficients objects that has an empty quantity list and returns 0 for
    all times and quantities.
    """

    def list_quantities(self) -> list[str]:
        return []

    def get_coefficient(self, time_step: int, quantity: str) -> float | np.ndarray:
        return 0.0


class AverageCoefficients(FunctionalCoefficients):
    """
    Functional coefficients object that computes the averages of its given quantities.
    """

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
    """
    Functional coefficients object that computes integrals over the given quantities
    via the composite trapezoidal rule.
    """

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
        if time_step in (0, self._integration_control.num_steps):
            return 0.5 * self._integration_control.delta_t
        else:
            return self._integration_control.delta_t
