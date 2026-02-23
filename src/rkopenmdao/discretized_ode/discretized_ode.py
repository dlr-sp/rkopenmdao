"""Interface for representing ODEs in RKOpenMDAO."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import numpy as np


@dataclass
class DiscretizedODEInputState:
    """
    Dataclass containing all the information about the input state of a discretized ODE.
    An instance of this class is used as an input for a non-differentiated run and for a
    differentiated run in forward mode. A run for reverse-mode derivatives has an
    instance of this class as result.

    Parameters
    ----------
    step_input: np.ndarray
        A vector corresponding to the input or a perturbation of the input data coming
        from the start of a time step.
    stage_input: np.ndarray
        A vector corresponding to the input or a perturbation of the input data coming
        from the start of a time stage.
    independent_input: np.ndarray
        A vector corresponding to the input or a perturbation of the input data that is
        independent of the time.
    time: float
        The time the ODE is evaluated at.
    """

    step_input: np.ndarray
    stage_input: np.ndarray
    independent_input: np.ndarray
    time: float
    linearization_point: np.ndarray | None = None


@dataclass
class DiscretizedODEResultState:
    """
    Dataclass containing all the information about the resulting state of a discretized
    ODE. An instance of this class is the result of both a non-differentiated run and
    for a differentiated run in forward mode. A run for reverse-mode derivatives needs
    an instance of this class as input.

    Parameters
    ----------
    stage_update: np.ndarray
        A vector corresponding to the output or a perturbation of the output data coming
        from the update of a time stage.
    stage_state: np.ndarray
        A vector corresponding to the output or a perturbation of the output data coming
        from the state of a time stage.
    independent_output: np.ndarray
        A vector corresponding to the output or a perturbation of the output data that
        is not directly dependent of the time integration (i.e. there is no time
        derivative for the contained data in the ODE system).
    """

    stage_update: np.ndarray
    stage_state: np.ndarray
    independent_output: np.ndarray


class DiscretizedODE(ABC):
    """
    Base class for the representation of ordinary differential equations (ODEs) in
    RKOpenMDAO.
    """

    CacheType = TypeVar("CacheType")

    @abstractmethod
    def compute_update(
        self,
        ode_input: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        """
        Given the information from the start of the time step and stage, as well as time
        independent information, time step size and a stage specific step size factor,
        computes the resulting stage update, stage state, and independent output
        information. After a call, the class instance is in a state where the functions
        compute_update_derivative and compute_update_adjoint_derivative can be called.
        Must to implemented in child class.

        Parameters
        ----------
        ode_input: DiscretizedODEInputState
            Input for the calculation of the time stage.
        step_size: float
            Step size of the current time step
        stage_factor: float
            A stage specific factor on the step size

        Returns
        -------
        ode_result: DiscretizedODEResultState
            Result for the calculation of the time stage.
        """

    @abstractmethod
    def get_linearization_point(self) -> DiscretizedODELinearizationPoint:
        """
        Exports the data of the ODE necessary for linearization. Must to implemented in
        child class.

        Returns
        -------
        linearization_state: DiscretizedODELinearizationPoint
            An object containing all the information necessary to linearize the class
            instance.
        """

    @abstractmethod
    def set_linearization_point(
        self, linearization_state: DiscretizedODELinearizationPoint
    ) -> None:
        """
        Imports the data of the ODE necessary for linearization. After a call, the class
        instance is in a state where the functions compute_update_derivative and
        compute_update_adjoint_derivative can be called. Must to implemented in
        child class.

        Parameters
        ----------
        linearization_state: DiscretizedODELinearizationPoint
            An object containing all the information necessary to linearize the class
            instance.
        """

    @abstractmethod
    def compute_update_derivative(
        self,
        ode_input_perturbation: DiscretizedODEInputState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEResultState:
        """
        Computes the matrix-vector product with the jacobian matrix of the stage update,
        the stage state and independent output wrt. step input, stage input independent
        inputs and time. Step size and stage factor are assumed to be constants, so
        there is no entries wrt. them in the jacobian. Must to implemented in
        child class.

        Parameters
        ----------
        ode_input_perturbation: DiscretizedODEInputState
            Input perturbation for the calculation of the derivative of time stage.
        step_size: float
            Step size of the current time step
        stage_factor: float
            Stage specific factor on the step size

        Returns
        -------
        ode_result_perturbation: DiscretizedODEResultState
            Result perturbation for the calculation of the derivative of the time stage.
        """

    @abstractmethod
    def compute_update_adjoint_derivative(
        self,
        ode_result_perturbation: DiscretizedODEResultState,
        step_size: float,
        stage_factor: float,
    ) -> DiscretizedODEInputState:
        """
        Computes the matrix-vector product with the adjoint jacobian matrix of the stage
        update, the stage state and independent output wrt. step input, stage input
        independent inputs and time. Step size and stage factor are assumed to be
        constants, so there is no entries wrt. them in the jacobian. Must to implemented
        in child class.

        Parameters
        ----------
        ode_result_perturbation: DiscretizedODEResultState
            Result perturbation for the calculation of the adjoint derivative of the
            time stage.
        step_size: float
            Step size of the current time step
        stage_factor: float
            Stage specific factor on the step size

        Returns
        -------
        ode_input_perturbation: DiscretizedODEInputState
            Input perturbation for the calculation of the adjoint derivative of time
            stage.
        """

    @abstractmethod
    def compute_state_norm(self, state: DiscretizedODEResultState) -> float:
        """
        Computes a norm of the provided state for the given ODE.

        Parameters
        ----------
        state: DiscretizedODEResultState
            State of which the norm is to be calculated

        Returns
        -------
        norm: float
            Norm of provided state
        """
