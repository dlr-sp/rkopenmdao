# pylint: disable=missing-module-docstring

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np


class DiscretizedODE(ABC):
    """
    Base class for the representation of ordinary differential equations (ODEs) in
    RKOpenMDAO.
    """

    CacheType = TypeVar("CacheType")

    @abstractmethod
    def compute_update(
        self,
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
        time: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given the information from the start of the time step and stage, as well as time
        independent information, time step size and a stage specific step size factor,
        computes the resulting stage update, stage state, and independent output
        information. After a call, the class instance is in a state where the functions
        compute_update_derivative and compute_update_adjoint_derivative can be called.
        Must to implemented in child class.

        Parameters
        ----------
        step_input: np.ndarray
            Input from start of time step
        stage_input: np.ndarray
            Input from start of time stage
        independent_input: np.ndarray
            Time independent input information
        time: float
            Time of the current time stage
        step_size: float
            Step size of the current time step
        stage_factor: float
            A stage specific factor on the step size

        Returns
        -------
        stage_update: np.ndarray
            Update for the current stage of the time integration scheme
        stage_state: np.ndarray
            State for the current stage of the time integration scheme
        independent_output: np.ndarray
            Output for the quantities not directly related to the time integration of
            the current stage of the scheme
        """

    @abstractmethod
    def export_linearization(self) -> CacheType:
        """
        Exports the data of the ODE necessary for linearization. Must to implemented in
        child class.

        Returns
        -------
        cache: CacheType
            A vector containing the linearization info the class instance.
        """

    @abstractmethod
    def import_linearization(self, cache: CacheType) -> None:
        """
        Imports the data of the ODE necessary for linearization. After a call, the class
        instance is in a state where the functions compute_update_derivative and
        compute_update_adjoint_derivative can be called. Must to implemented in
        child class.

        Parameters
        ----------
        cache: CacheType
            A vector containing the linearization info the class instance.
        """

    @abstractmethod
    def compute_update_derivative(
        self,
        step_input_pert: np.ndarray,
        stage_input_pert: np.ndarray,
        independent_input_pert: np.ndarray,
        time_pert: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the matrix-vector product with the jacobian matrix of the stage update,
        the stage state and independent output wrt. step input, stage input independent
        inputs and time. Step size and stage factor are assumed to be constants, so
        there is no entries wrt. them in the jacobian. Must to implemented in
        child class.

        Parameters
        ----------
        step_input_pert: np.ndarray
            Input perturbation from start of time step
        stage_input_pert: np.ndarray
            Input perturbation from start of time stage
        independent_input_pert: np.ndarray
            Time independent input perturbation
        time_pert: float
            Time perturbation of the current time step
        step_size: float
            Step size of the current time step
        stage_factor: float
            Stage specific factor on the step size

        Returns
        -------
        stage_update_pert: np.ndarray
            Perturbation for the update of the current stage of the time integration
            scheme
        stage_stage_pert: np.ndarray
            Perturbation for the state of the current stage of the time integration
            scheme
        independent_output_pert: np.ndarray
            Perturbation of the output for the quantities not directly related to the
            time integration of the current stage of the scheme
        """

    @abstractmethod
    def compute_update_adjoint_derivative(
        self,
        stage_update_pert: np.ndarray,
        stage_output_pert: np.ndarray,
        independent_output_pert: np.ndarray,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Computes the matrix-vector product with the adjoint jacobian matrix of the stage
        update, the stage state and independent output wrt. step input, stage input
        independent inputs and time. Step size and stage factor are assumed to be
        constants, so there is no entries wrt. them in the jacobian. Must to implemented
        in child class.

        Parameters
        ----------
        stage_update_pert: np.ndarray
            Stage output perturbation from end of time stage
        stage_output_pert: np.ndarray
            Stage output perturbation from end of time stage
        independent_output_pert: np.ndarray
            Perturbation of the output for the quantities not directly related to the
            time integration
        step_size: float
            Step size of the current time step
        stage_factor: float
            Stage specific factor on the step size

        Returns
        -------
        step_input_pert: np.ndarray
            Input perturbation from start of time step
        stage_input_pert: np.ndarray
            Input perturbation from start of time stage
        independent_input_pert: np.ndarray
            Time independent input perturbation
        time_pert: float
            Time perturbation of the current time step
        """
