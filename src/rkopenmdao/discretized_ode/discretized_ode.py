"""Interface for representing ODEs in RKOpenMDAO."""

from abc import ABC, abstractmethod

from rkopenmdao.states import (
    DiscretizedODEInputState,
    DiscretizedODEResultState,
)


class DiscretizedODE(ABC):
    """
    Base class for the representation of ordinary differential equations (ODEs) in
    RKOpenMDAO.
    """

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

    @abstractmethod
    def get_state_size(self) -> int:
        """
        Returns the size of the time integration state of the ODE.

        Returns
        -------
        Time integration state size.
        """

    @abstractmethod
    def get_independent_input_size(self) -> int:
        """
        Returns the size of the inputs independent of time of the ODE.

        Returns
        -------
        Independent input size.
        """

    @abstractmethod
    def get_independent_output_size(self) -> int:
        """
        Returns the size of the outputs independent of time of the ODE.

        Returns
        -------
        Independent output size.
        """

    @abstractmethod
    def get_linearization_point_size(self) -> int:
        """
        Returns the size of a linearization point of the ODE.

        Returns
        -------
        Linearization point size.
        """
