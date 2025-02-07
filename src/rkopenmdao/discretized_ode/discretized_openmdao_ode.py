import openmdao.api as om
from openmdao.vectors.vector import Vector as OMVector

from .discretized_ode_interface import (
    DiscretizedODEInterface,
    InputInformation,
    OutputAndResidualInformation,
)


from ..integration_control import IntegrationControl
from ..metadata_extractor import (
    extract_time_integration_metadata,
    add_time_independent_input_metadata,
    TimeIntegrationMetadata,
)


class DiscretizedOpenMDAOODE(DiscretizedODEInterface):
    openmdao_problem: om.Problem
    integration_control: IntegrationControl
    time_integration_metadata: TimeIntegrationMetadata

    def __init__(
        self,
        openmdao_problem: om.Problem,
        integration_control: IntegrationControl,
        time_integration_quantities: list[str],
        independent_input_quantities: list[str] = None,
        independent_output_quantities: list[str] = None,
    ):

        self.openmdao_problem = openmdao_problem
        self.integration_control = integration_control
        self.time_integration_metadata = extract_time_integration_metadata(
            self.openmdao_problem, time_integration_quantities
        )
        if independent_input_quantities is not None:
            add_time_independent_input_metadata(
                self.openmdao_problem,
                independent_input_quantities,
                self.time_integration_metadata,
            )
        if independent_output_quantities is not None:
            # TODO: add path for independent_output_quantities
            pass

    def evalute_instationary_residual(
        self,
        input_information: InputInformation,
        output_information: OutputAndResidualInformation,
    ) -> OutputAndResidualInformation:
        self._insert_input_data(input_information, "nonlinear")
        self._insert_output_data(output_information, "nonlinear")
        self.openmdao_problem.model.run_apply_nonlinear()
        return self._extract_residual_data("nonlinear")

    def solve_instationary_residual(
        self,
        input_information: InputInformation,
    ) -> OutputAndResidualInformation:
        self._insert_input_data(input_information, "nonlinear")
        self.openmdao_problem.model.run_solve_nonlinear()
        return self._extract_output_data("nonlinear")

    def assemble_instationary_jacobian(
        self,
        input_information: InputInformation,
        output_information: OutputAndResidualInformation,
    ) -> None:
        self._insert_input_data(input_information, "nonlinear")
        self._insert_output_data(output_information, "nonlinear")
        self.openmdao_problem.model.run_linearize()

    def evaluate_instationary_jacobian(
        self,
        input_information: InputInformation,
        output_information: OutputAndResidualInformation,
        input_perturbation: InputInformation,
        output_perturbation: OutputAndResidualInformation,
    ) -> OutputAndResidualInformation:
        self._insert_input_data(input_information, "nonlinear")
        self._insert_output_data(output_information, "nonlinear")
        self._insert_input_data(input_perturbation, "linear")
        self._insert_output_data(output_perturbation, "linear")
        try:
            self.openmdao_problem.model.run_apply_linear("fwd")
        except TypeError:  # old openMDAO version had different interface
            self.openmdao_problem.model.run_apply_linear(
                vec_names=["linear"], mode="fwd"
            )
        return self._extract_residual_data("linear")

    def evaluate_instationary_transposed_jacobian(
        self,
        input_information: InputInformation,
        output_information: OutputAndResidualInformation,
        residual_perturbation: OutputAndResidualInformation,
    ) -> tuple[InputInformation, OutputAndResidualInformation]:
        self._insert_input_data(input_information, "nonlinear")
        self._insert_output_data(output_information, "nonlinear")
        self._insert_residual_data(residual_perturbation, "linear")
        try:
            self.openmdao_problem.model.run_apply_linear(mode="rev")
        except TypeError:  # old openMDAO version had different interface
            self.openmdao_problem.model.run_apply_linear(
                vec_names=["linear"], mode="rev"
            )
        return (self._extract_input_data("linear"), self._extract_output_data("linear"))

    def solve_instationary_jacobian(
        self, residual_perturbation: OutputAndResidualInformation
    ) -> OutputAndResidualInformation:
        self._insert_residual_data(residual_perturbation, "linear")
        try:
            self.openmdao_problem.model.run_solve_linear(mode="fwd")
        except TypeError:  # old openMDAO version had different interface
            self.openmdao_problem.model.run_solve_linear(
                vec_names=["linear"], mode="fwd"
            )
        return self._extract_output_data("linear")

    def solve_instationary_transposed_jacobian(
        self, output_perturbation: OutputAndResidualInformation
    ) -> OutputAndResidualInformation:
        self._insert_residual_data(output_perturbation, "linear")
        try:
            self.openmdao_problem.model.run_solve_linear(mode="fwd")
        except TypeError:  # old openMDAO version had different interface
            self.openmdao_problem.model.run_solve_linear(
                vec_names=["linear"], mode="fwd"
            )
        return self._extract_residual_data("linear")

    def _insert_input_data(self, input_information: InputInformation, vector_type: str):
        fill_vector = self._get_vector(vector_type=vector_type, variable_type="input")
        self._insert_time_integration_inputs(fill_vector, input_information)
        self._insert_time_independent_inputs(fill_vector, input_information)

    def _extract_input_data(self, vector_type: str) -> InputInformation:
        input_information = InputInformation(
            self.time_integration_metadata.time_integration_array_size,
            self.time_integration_metadata.time_independent_input_size,
        )
        extract_vector = self._get_vector(
            vector_type=vector_type, variable_type="input"
        )
        self._extract_time_integration_inputs(extract_vector, input_information)
        self._extract_time_independent_inputs(extract_vector, input_information)

        return input_information

    def _insert_output_data(
        self, output_information: OutputAndResidualInformation, vector_type: str
    ):
        fill_vector = self._get_vector(vector_type=vector_type, variable_type="output")
        self._insert_time_integration_outputs_or_residuals(
            fill_vector, output_information
        )
        # TODO: once time independent outputs are ready, insert the following
        # self._insert_time_independent_outputs_or_residuals(
        # fill_vector, output_information
        # )

    def _extract_output_data(self, vector_type: str) -> OutputAndResidualInformation:
        output_information = OutputAndResidualInformation(
            self.time_integration_metadata.time_integration_array_size
        )
        extract_vector = self._get_vector(
            vector_type=vector_type, variable_type="output"
        )
        self._extract_time_integration_outputs_or_residuals(
            extract_vector, output_information
        )
        # TODO: do the same for time independent outputs once ready
        return output_information

    def _insert_residual_data(
        self, residual_information: OutputAndResidualInformation, vector_type: str
    ):
        fill_vector = self._get_vector(
            vector_type=vector_type, variable_type="residual"
        )
        self._insert_time_integration_outputs_or_residuals(
            fill_vector, residual_information
        )
        # TODO: once time independent outputs are ready, insert the following
        # self._insert_time_independent_outputs_or_residuals(
        # fill_vector, residual_information
        # )

    def _extract_residual_data(self, vector_type: str) -> OutputAndResidualInformation:
        residual_information = OutputAndResidualInformation(
            self.time_integration_metadata.time_integration_array_size
        )
        extract_vector = self._get_vector(
            vector_type=vector_type, variable_type="residual"
        )
        self._extract_time_integration_outputs_or_residuals(
            extract_vector, residual_information
        )
        # TODO: do the same for time independent outputs once ready
        return residual_information

    def _get_vector(self, vector_type: str, variable_type: str) -> OMVector:
        if variable_type in ["input", "output"]:
            index = 1
        elif variable_type == "residual":
            index = 2
        else:
            raise Exception(f"unknown variable type: {variable_type}")
        if vector_type == "nonlinear":
            return self.openmdao_problem.model.get_nonlinear_vectors()[index]
        elif vector_type == "linear":
            return self.openmdao_problem.model.get_linear_vectors()[index]
        else:
            raise Exception(f"unknown vector type: {vector_type}")

    def _insert_time_integration_inputs(
        self, fill_vector: OMVector, input_information: InputInformation
    ):
        for quantity in self.time_integration_metadata.time_integration_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                if quantity.translation_metadata.step_input_var is not None:
                    fill_vector[
                        self.openmdao_problem.model.get_source(
                            quantity.translation_metadata.step_input_var
                        )
                    ] = input_information.old_step_vector[start:end].reshape(
                        quantity.array_metadata.shape
                    )
                    fill_vector[
                        self.openmdao_problem.model.get_source(
                            quantity.translation_metadata.accumulated_stage_var
                        )
                    ] = input_information.old_stage_vector[start:end].reshape(
                        quantity.array_metadata.shape
                    )

    def _extract_time_integration_inputs(
        self, extract_vector: OMVector, input_information: InputInformation
    ):
        for quantity in self.time_integration_metadata.time_integration_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                if quantity.translation_metadata.step_input_var is not None:
                    input_information.old_step_vector[start:end] = extract_vector[
                        self.openmdao_problem.model.get_source(
                            quantity.translation_metadata.step_input_var
                        )
                    ].flatten()
                    input_information.old_stage_vector[start:end] = extract_vector[
                        self.openmdao_problem.model.get_source(
                            quantity.translation_metadata.accumulated_stage_var
                        )
                    ].flatten()

    def _insert_time_integration_outputs_or_residuals(
        self,
        fill_vector: OMVector,
        output_or_residual_information: OutputAndResidualInformation,
    ):
        for quantity in self.time_integration_metadata.time_integration_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                fill_vector[quantity.translation_metadata.stage_output_var] = (
                    output_or_residual_information[start:end].reshape(
                        quantity.array_metadata.shape
                    )
                )
                # TODO: do the equivalent for the output state once its part of the
                #  metadata

    def _extract_time_integration_outputs_or_residuals(
        self,
        extract_vector: OMVector,
        output_or_residual_information: OutputAndResidualInformation,
    ):
        for quantity in self.time_integration_metadata.time_integration_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                output_or_residual_information.new_stage_vector[start:end] = (
                    extract_vector[
                        quantity.translation_metadata.stage_output_var
                    ].flatten()
                )

                # TODO: do the equivalent for the output state once its part of the
                #  metadata

    def _insert_time_independent_inputs(
        self, fill_vector: OMVector, input_information: InputInformation
    ):
        for (
            quantity
        ) in self.time_integration_metadata.time_independent_input_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                fill_vector[
                    self.openmdao_problem.model.get_source(
                        quantity.translation_metadata.time_independent_input_var
                    )
                ] = input_information.independent_input_vector[start:end].reshape(
                    quantity.array_metadata.shape
                )

    def _extract_time_independent_inputs(
        self, extract_vector: OMVector, input_information: InputInformation
    ):
        for (
            quantity
        ) in self.time_integration_metadata.time_independent_input_quantity_list:
            if quantity.array_metadata.local:
                start = quantity.array_metadata.start_index
                end = quantity.array_metadata.end_index
                input_information.independent_input_vector[start:end] = extract_vector[
                    self.openmdao_problem.model.get_source(
                        quantity.translation_metadata.time_independent_input_var
                    )
                ].flatten()
