"""Runge-Kutta integration implementation for OpenMDAO."""

# pylint: disable=protected-access

from __future__ import annotations

import numpy as np
from openmdao.vectors.vector import Vector
from rkopenmdao.metadata_extractor import TimeIntegrationMetadata
from rkopenmdao.openmdao_time_integration_wrapper import OpenMDAOTimeIntegrationWrapper

from rkopenmdao.states import FinalizationValues, StartingValues
from rkopenmdao.time_integration_interface import TimeIntegrationInterface

from rkopenmdao.discretized_ode.openmdao_ode import OpenMDAOODE


class OpenMDAOTimeStepping(OpenMDAOTimeIntegrationWrapper):
    """Outer component for solving time-dependent problems with explicit or diagonally
    implicit Runge-Kutta schemes. One stage of the scheme is modelled by an inner
    OpenMDAO-problem.
    OpenMDAO inputs: - initial values of the quantities for the time integration
                     - ((optional) parameters independent from the time integration
    OpenMDAO output: - final values of the quantities for the time integration
    """

    def initialize(self):
        """Declares the `time_integrator` option, validated by `has_om_ode`."""
        self.options.declare(
            "time_integrator",
            types=TimeIntegrationInterface,
            check_valid=self.has_om_ode,
        )

    @staticmethod
    def has_om_ode(name: str, value: TimeIntegrationInterface):
        """
        Validates a candidate value for the `time_integrator` option by requiring it
        to have an attribute `ode` of type `OpenMDAOODE`, raising a ValueError
        otherwise.

        Parameters
        ----------
        name: str
            Name of the option that is checked.
        value: TimeIntegrationInterface
            Candidate value for the option that is checked.
        """
        if hasattr(value, "ode"):
            if isinstance(value.ode, OpenMDAOODE):
                return
        raise ValueError(
            f"Option {name} needs an attribute ode which must be of type OpenMDAOODE."
        )

    def setup(self):
        """Stores the time integrator from the options and adds the inputs and
        outputs of the component."""
        self._time_integrator = self.options["time_integrator"]
        self._cached_final_state = None
        self._add_inputs_and_outputs()

    def _add_inputs_and_outputs(self):
        """Adds all inputs and outputs of the component, namely the time
        integrated quantities and the time independent inputs."""
        self._add_time_integration_inputs_and_outputs()
        self._add_time_independent_inputs()

    def _add_time_integration_inputs_and_outputs(self):
        """
        Adds the `time_initial` and `time_final` variables and, for each time
        integrated quantity, an input named after the quantity with the suffix
        `_initial` and the corresponding output with the suffix `_final`.
        """
        self.add_input("time_initial", shape=1, val=0.0)
        self.add_output("time_final", shape=1)
        time_integration_metadata = self._time_integrator.ode.time_integration_metadata
        for quantity in time_integration_metadata.time_integration_quantity_list:
            if quantity.array_metadata.local:
                self.add_input(
                    quantity.name + "_initial",
                    shape=quantity.array_metadata.shape,
                    val=(
                        # Arrays of size 0 tend to have the wrong shape when aquired
                        # from OpenMDAO, making a manual reshape necessary in that
                        # case.
                        self._time_integrator.ode._time_stage_problem.get_val(
                            quantity.translation_metadata.step_input_var,
                        ).reshape(quantity.array_metadata.shape)
                        if quantity.translation_metadata.step_input_var is not None
                        else np.zeros(quantity.array_metadata.shape)
                    ),
                    distributed=quantity.array_metadata.distributed,
                )
            else:
                self.add_input(
                    quantity.name + "_initial",
                    shape=quantity.array_metadata.shape,
                    distributed=quantity.array_metadata.distributed,
                )
            self.add_output(
                quantity.name + "_final",
                copy_shape=quantity.name + "_initial",
                distributed=quantity.array_metadata.distributed,
            )

    def _add_time_independent_inputs(self):
        """
        Adds one input for each time independent input quantity, initialized with
        the current value of the quantity in the inner OpenMDAO problem.
        """
        time_integration_metadata = self._time_integrator.ode.time_integration_metadata
        for quantity in time_integration_metadata.time_independent_input_quantity_list:
            self.add_input(
                quantity.name,
                shape=quantity.array_metadata.shape,
                val=self._time_integrator.ode._time_stage_problem.get_val(
                    quantity.translation_metadata.time_independent_input_var
                ),
                distributed=quantity.array_metadata.distributed,
            )

    def _get_starting_values_from_inputs(self, inputs):
        """
        Collects the starting values for the time integration from the OpenMDAO
        inputs of the component.

        Parameters
        ----------
        inputs: Vector
            OpenMDAO input vector of the component.

        Returns
        -------
        starting_values: StartingValues
            Starting values for the time integration.
        """
        time_integration_metadata: TimeIntegrationMetadata = (
            self._time_integrator.ode.time_integration_metadata
        )
        starting_values = StartingValues(
            initial_time=inputs["time_initial"][0],
            initial_values=np.zeros(
                time_integration_metadata.time_integration_array_size
            ),
            independent_inputs=np.zeros(
                time_integration_metadata.time_independent_input_size
            ),
        )
        for quantity in time_integration_metadata.time_integration_quantity_list:
            starting_values.initial_values[
                quantity.array_metadata.start_index : quantity.array_metadata.end_index
            ] = inputs[quantity.name + "_initial"].flatten()
        for quantity in time_integration_metadata.time_independent_input_quantity_list:
            starting_values.independent_inputs[
                quantity.array_metadata.start_index : quantity.array_metadata.end_index
            ] = inputs[quantity.name].flatten()

        return starting_values

    def _get_inputs_from_starting_values(self, starting_values, inputs: Vector):
        """
        Transfers the given starting values to the OpenMDAO inputs of the component.

        Parameters
        ----------
        starting_values: StartingValues
            Starting values to be written to the inputs.
        inputs: Vector
            OpenMDAO input vector of the component.
        """
        time_integration_metadata: TimeIntegrationMetadata = (
            self._time_integrator._ode.time_integration_metadata
        )
        inputs["time_initial"][0] = starting_values.initial_time
        for quantity in time_integration_metadata.time_integration_quantity_list:
            inputs[quantity.name + "_initial"] = starting_values.initial_values[
                quantity.array_metadata.start_index : quantity.array_metadata.end_index
            ].reshape(quantity.array_metadata.shape)
        for quantity in time_integration_metadata.time_independent_input_quantity_list:
            inputs[quantity.name] = starting_values.independent_inputs[
                quantity.array_metadata.start_index : quantity.array_metadata.end_index
            ].reshape(quantity.array_metadata.shape)

    def _add_starting_values_to_inputs(self, starting_values, inputs):
        """
        Adds the given starting values to the OpenMDAO inputs of the component, as
        needed when applying perturbations.

        Parameters
        ----------
        starting_values: StartingValues
            Starting values to be added to the inputs.
        inputs: Vector
            OpenMDAO input vector of the component.
        """
        time_integration_metadata: TimeIntegrationMetadata = (
            self._time_integrator.ode.time_integration_metadata
        )
        inputs["time_initial"] += starting_values.initial_time
        for quantity in time_integration_metadata.time_integration_quantity_list:
            inputs[quantity.name + "_initial"] += starting_values.initial_values[
                quantity.array_metadata.start_index : quantity.array_metadata.end_index
            ].reshape(quantity.array_metadata.shape)
        for quantity in time_integration_metadata.time_independent_input_quantity_list:
            inputs[quantity.name] += starting_values.independent_inputs[
                quantity.array_metadata.start_index : quantity.array_metadata.end_index
            ].reshape(quantity.array_metadata.shape)

    def _get_finalization_values_from_outputs(self, outputs):
        """
        Collects the finalization values of the time integration from the OpenMDAO
        outputs of the component.

        Parameters
        ----------
        outputs: Vector
            OpenMDAO output vector of the component.

        Returns
        -------
        finalization_values: FinalizationValues
            Finalization values of the time integration.
        """
        time_integration_metadata: TimeIntegrationMetadata = (
            self._time_integrator.ode.time_integration_metadata
        )
        finalization_values = FinalizationValues(
            final_time=outputs["time_final"][0],
            final_values=np.zeros(
                time_integration_metadata.time_integration_array_size
            ),
            final_independent_outputs=np.zeros(0),  # Not Implemented,
        )
        for quantity in time_integration_metadata.time_integration_quantity_list:
            finalization_values.final_values[
                quantity.array_metadata.start_index : quantity.array_metadata.end_index
            ] = outputs[quantity.name + "_final"].flatten()

        return finalization_values

    def _get_outputs_from_finalization_values(self, finalization_values, outputs):
        """
        Transfers the given finalization values to the OpenMDAO outputs of the
        component.

        Parameters
        ----------
        finalization_values: FinalizationValues
            Finalization values to be written to the outputs.
        outputs: Vector
            OpenMDAO output vector of the component.
        """
        time_integration_metadata: TimeIntegrationMetadata = (
            self._time_integrator.ode.time_integration_metadata
        )
        outputs["time_final"][0] = finalization_values.final_time
        for quantity in time_integration_metadata.time_integration_quantity_list:
            outputs[quantity.name + "_final"] = finalization_values.final_values[
                quantity.array_metadata.start_index : quantity.array_metadata.end_index
            ].reshape(quantity.array_metadata.shape)

    def _add_finalization_values_to_outputs(self, finalization_values, outputs):
        """
        Adds the given finalization values to the OpenMDAO outputs of the component,
        as needed when applying perturbations.

        Parameters
        ----------
        finalization_values: FinalizationValues
            Finalization values to be added to the outputs.
        outputs: Vector
            OpenMDAO output vector of the component.
        """
        time_integration_metadata: TimeIntegrationMetadata = (
            self._time_integrator.ode.time_integration_metadata
        )
        outputs["time_final"][0] += finalization_values.final_time
        for quantity in time_integration_metadata.time_integration_quantity_list:
            outputs[quantity.name + "_final"] += finalization_values.final_values[
                quantity.array_metadata.start_index : quantity.array_metadata.end_index
            ].reshape(quantity.array_metadata.shape)
