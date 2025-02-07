from abc import ABC, abstractmethod

import numpy as np


class InputInformation:
    def __init__(self, state_size: int, independent_input_size: int = 0):
        self.time = 0
        self.delta_t = 0
        self.butcher_diagonal_element = 0
        self.old_step_vector: np.ndarray = np.zeros(state_size)
        self.old_stage_vector: np.ndarray = np.zeros(state_size)
        self.independent_input_vector: np.ndarray = np.zeros(independent_input_size)


class OutputAndResidualInformation:
    def __init__(self, state_size: int, independent_output_size: int = 0):
        self.new_state_vector: np.ndarray = np.zeros(state_size)
        self.new_stage_vector: np.ndarray = np.zeros(state_size)
        self.independent_output_vector: np.ndarray = np.zeros(independent_output_size)


class DiscretizedODEInterface(ABC):

    @abstractmethod
    def evalute_instationary_residual(
        self,
        input_information: InputInformation,
        output_information: OutputAndResidualInformation,
    ) -> OutputAndResidualInformation:
        """"""

    @abstractmethod
    def solve_instationary_residual(
        self,
        input_information: InputInformation,
    ) -> OutputAndResidualInformation:
        """"""

    @abstractmethod
    def assemble_instationary_jacobian(
        self,
        input_information: InputInformation,
        output_information: OutputAndResidualInformation,
    ) -> None:
        """"""

    @abstractmethod
    def evaluate_instationary_jacobian(
        self,
        input_information: InputInformation,
        output_information: OutputAndResidualInformation,
        input_perturbation: InputInformation,
        output_perturbation: OutputAndResidualInformation,
    ) -> OutputAndResidualInformation:
        """"""

    @abstractmethod
    def evaluate_instationary_transposed_jacobian(
        self,
        input_information: InputInformation,
        output_information: OutputAndResidualInformation,
        residual_perturbation: OutputAndResidualInformation,
    ) -> tuple[InputInformation, OutputAndResidualInformation]:
        """"""

    @abstractmethod
    def solve_instationary_jacobian(
        self, residual_perturbation: OutputAndResidualInformation
    ) -> OutputAndResidualInformation:
        """"""

    @abstractmethod
    def solve_instationary_transposed_jacobian(
        self, output_perturbation: OutputAndResidualInformation
    ) -> OutputAndResidualInformation:
        """"""
