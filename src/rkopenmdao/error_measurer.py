"""Contains the classes for the error Estimator of the error controller."""

# pylint: disable = c-extension-no-member

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rkopenmdao.discretized_ode.discretized_ode import (
    DiscretizedODE,
    DiscretizedODEResultState,
)


class ErrorMeasurer(ABC):
    """
    Interface for classes that compute a measure of the error for the use in an error
    controller for adaptive time stepping.
    """

    @abstractmethod
    def get_measure(
        self,
        state_error_estimate: DiscretizedODEResultState,
        state: DiscretizedODEResultState,
        ode: DiscretizedODE,
    ) -> float:
        """
        Parameters
        ----------
        state_error_estimate : DiscretizedODEResultState
            Error estimate of one step of a time integration
        state : DiscretizedODEResultState
            Solution of one step of a time integration
        ode: DiscretizedODE:
            Equation that the error estimate and solution belong to.

        Returns
        -------
        float
            Error measure for use in adaptive time stepping.
        """


class SimpleErrorMeasurer(ErrorMeasurer):
    """
    A simple error measure that only uses the norm of the error estimate.
    """

    def get_measure(
        self,
        state_error_estimate: DiscretizedODEResultState,
        state: DiscretizedODEResultState,
        ode: DiscretizedODE,
    ) -> float:
        return ode.compute_state_norm(state_error_estimate)


@dataclass
class ImprovedErrorMeasurer(ErrorMeasurer):
    """
    A error measure that calculates a weighted average of the absolute and relative
    norms.

    Parameters
    ----------
    eta : float, optional
        A small positive absolute tolerance which added to avoid division by zero.
    eps: float, optional
        relative error tolerance
    """

    eta: float = 1e-6
    eps: float = 1e-6

    def get_measure(
        self,
        state_error_estimate: DiscretizedODEResultState,
        state: DiscretizedODEResultState,
        ode: DiscretizedODE,
    ) -> float:

        error_estimate_norm = ode.compute_state_norm(state_error_estimate)
        state_norm = ode.compute_state_norm(state)

        return error_estimate_norm / (state_norm + self.eta / self.eps)
