"""Base implementation of a time integration as an OpenMDAO component."""

from abc import ABC, abstractmethod
from copy import deepcopy

import openmdao.api as om
from openmdao.vectors.vector import Vector

from rkopenmdao.states import FinalizationValues, StartingValues
from rkopenmdao.time_integration_interface import TimeIntegrationInterface


class OpenMDAOTimeIntegrationWrapper(om.ExplicitComponent, ABC):
    """
    Base class for OpenMDAO components that run a time integration on their inputs by
    delegating to a `TimeIntegrationInterface`, handling the transfer of data between
    the OpenMDAO inputs/outputs and the states of the time integration for the primal
    and the differentiated computations.
    """

    _time_integrator: TimeIntegrationInterface | None
    _cached_final_state: TimeIntegrationInterface | None

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Runs the time integration on the OpenMDAO inputs and writes the resulting
        final values to the OpenMDAO outputs, caching the final discretization state
        for use as linearization point in the reverse mode.
        """
        starting_values = self._get_starting_values_from_inputs(inputs)
        state = self._time_integrator.starting_scheme(starting_values)
        state = self._time_integrator.integrate(state)
        self._cached_final_state = deepcopy(state[-1])
        finalization_values = self._time_integrator.finalization_scheme(state[-1])
        self._get_outputs_from_finalization_values(finalization_values, outputs)

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):
        """
        Computes the jacobian-vector product of the time integration for the given
        mode, propagating the input perturbations through the starting scheme, the
        time integration, and the finalization scheme to the output perturbations.
        """
        starting_values = self._get_starting_values_from_inputs(inputs)
        state = self._time_integrator.starting_scheme(starting_values)
        if mode == "fwd":
            starting_value_perturbations = self._get_starting_values_from_inputs(
                d_inputs
            )
            state_perturbations = self._time_integrator.starting_scheme_derivative(
                starting_values, starting_value_perturbations
            )
            state_perturbations = self._time_integrator.integrate_derivative(
                state, state_perturbations
            )
            finalization_value_perturbations = (
                self._time_integrator.finalization_scheme_derivative(
                    state, state_perturbations[1][-1]
                )
            )
            self._add_finalization_values_to_outputs(
                finalization_value_perturbations, d_outputs
            )
        if mode == "rev":
            if self._cached_final_state is None:
                self.compute(inputs, self._outputs)
            finalization_value_perturbations = (
                self._get_finalization_values_from_outputs(d_outputs)
            )
            state_perturbations = (
                self._time_integrator.finalization_scheme_adjoint_derivative(
                    self._cached_final_state, finalization_value_perturbations
                )
            )
            initial_state_perturbations = (
                self._time_integrator.integrate_adjoint_derivative(
                    state, [state_perturbations]
                )
            )
            starting_value_perturbations = (
                self._time_integrator.starting_scheme_adjoint_derivative(
                    starting_values, initial_state_perturbations
                )
            )
            self._add_starting_values_to_inputs(starting_value_perturbations, d_inputs)

    @abstractmethod
    def _get_starting_values_from_inputs(self, inputs: Vector) -> StartingValues:
        """
        Extracts the starting values for the time integration from the given
        OpenMDAO inputs.

        Parameters
        ----------
        inputs: Vector
            OpenMDAO input vector of the component.

        Returns
        -------
        starting_values: StartingValues
            Starting values for the time integration.
        """

    @abstractmethod
    def _get_inputs_from_starting_values(
        self, starting_values: StartingValues, inputs: Vector
    ):
        """
        Transfers the given starting values to the OpenMDAO inputs.

        Parameters
        ----------
        starting_values: StartingValues
            Starting values to be written to the inputs.
        inputs: Vector
            OpenMDAO input vector of the component.
        """

    @abstractmethod
    def _add_starting_values_to_inputs(
        self, starting_values: StartingValues, inputs: Vector
    ):
        """
        Adds the given starting values to the OpenMDAO inputs, as needed when
        applying perturbations.

        Parameters
        ----------
        starting_values: StartingValues
            Starting values to be added to the inputs.
        inputs: Vector
            OpenMDAO input vector of the component.
        """

    @abstractmethod
    def _get_finalization_values_from_outputs(
        self, outputs: Vector
    ) -> FinalizationValues:
        """
        Extracts the finalization values of the time integration from the given
        OpenMDAO outputs.

        Parameters
        ----------
        outputs: Vector
            OpenMDAO output vector of the component.

        Returns
        -------
        finalization_values: FinalizationValues
            Finalization values of the time integration.
        """

    @abstractmethod
    def _get_outputs_from_finalization_values(
        self, finalization_values: FinalizationValues, outputs: Vector
    ):
        """
        Transfers the given finalization values to the OpenMDAO outputs.

        Parameters
        ----------
        finalization_values: FinalizationValues
            Finalization values to be written to the outputs.
        outputs: Vector
            OpenMDAO output vector of the component.
        """

    @abstractmethod
    def _add_finalization_values_to_outputs(
        self, finalization_values: FinalizationValues, outputs: Vector
    ):
        """
        Adds the given finalization values to the OpenMDAO outputs, as needed when
        applying perturbations.

        Parameters
        ----------
        finalization_values: FinalizationValues
            Finalization values to be added to the outputs.
        outputs: Vector
            OpenMDAO output vector of the component.
        """
