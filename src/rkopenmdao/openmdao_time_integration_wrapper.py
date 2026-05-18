from abc import ABC, abstractmethod

import openmdao.api as om
from openmdao.vectors.vector import Vector

from rkopenmdao.states import FinalizationValues, StartingValues
from rkopenmdao.time_integration_interface import TimeIntegrationInterface


class OpenMDAOTimeIntegrationWrapper(om.ExplicitComponent, ABC):
    _time_integrator: TimeIntegrationInterface | None

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        starting_values = self._get_starting_values_from_inputs(inputs)
        state = self._time_integrator.starting_scheme(starting_values)
        state = self._time_integrator.integrate(state)
        finalization_values = self._time_integrator.finalization_scheme(state)
        self._get_outputs_from_finalization_values(finalization_values, outputs)

    def compute_jacvec_product(
        self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None
    ):
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
                    state, state_perturbations
                )
            )
            self._add_finalization_values_to_outputs(
                finalization_value_perturbations, d_outputs
            )
        if mode == "rev":
            finalization_value_perturbations = (
                self._get_finalization_values_from_outputs(d_outputs)
            )
            state_perturbations = (
                self._time_integrator.finalization_scheme_adjoint_derivative(
                    state, finalization_value_perturbations
                )
            )
            state_perturbations = self._time_integrator.integrate_adjoint_derivative(
                state, state_perturbations
            )
            starting_value_perturbations = (
                self._time_integrator.starting_scheme_adjoint_derivative(
                    starting_values, state_perturbations
                )
            )
            self._add_starting_values_to_inputs(starting_value_perturbations, d_inputs)

    @abstractmethod
    def _get_starting_values_from_inputs(self, inputs: Vector) -> StartingValues:
        """ """

    @abstractmethod
    def _get_inputs_from_starting_values(
        self, starting_values: StartingValues, inputs: Vector
    ):
        """ """

    @abstractmethod
    def _add_starting_values_to_inputs(
        self, starting_values: StartingValues, inputs: Vector
    ):
        """"""

    @abstractmethod
    def _get_finalization_values_from_outputs(
        self, outputs: Vector
    ) -> FinalizationValues:
        """ """

    @abstractmethod
    def _get_outputs_from_finalization_values(
        self, finalization_values: FinalizationValues, outputs: Vector
    ):
        """ """

    @abstractmethod
    def _add_finalization_values_to_outputs(
        self, finalization_values: FinalizationValues, outputs: Vector
    ):
        """"""
