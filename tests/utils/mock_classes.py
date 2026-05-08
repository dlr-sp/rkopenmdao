from copy import deepcopy
import numpy as np

from rkopenmdao.discretized_ode.discretized_ode import (
    DiscretizedODE,
    DiscretizedODEInputState,
    DiscretizedODEResultState,
)
from rkopenmdao.states import TimeDiscretizationStateInterface
from rkopenmdao.time_discretization.time_discretization_scheme_interface import (
    TimeDiscretizationSchemeInterface,
    StartingValues,
    FinalizationValues,
)


class MockODE(DiscretizedODE):
    def compute_update(self, ode_input, step_size, stage_factor):
        return DiscretizedODEResultState(
            np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        )

    def compute_update_derivative(
        self, ode_input_perturbation, step_size, stage_factor
    ):
        return DiscretizedODEResultState(np.zeros(0), np.zeros(0), np.zeros(0))

    def compute_update_adjoint_derivative(
        self, ode_result_perturbation, step_size, stage_factor
    ):
        return DiscretizedODEInputState(np.zeros(0), np.zeros(0), np.zeros(0), 0.0)

    def compute_state_norm(self, state):
        return 0.0

    def get_state_size(self):
        return 0

    def get_independent_input_size(self):
        return 0

    def get_independent_output_size(self):
        return 0

    def get_linearization_point_size(self):
        return 0


class MockDiscretizationState(TimeDiscretizationStateInterface):

    def set(self, other):
        pass

    def to_dict(self):
        return {}

    @classmethod
    def from_dict(cls, state_dict):
        return MockDiscretizationState()


class MockDiscretization(TimeDiscretizationSchemeInterface):

    def create_empty_discretization_state(self, ode):
        return MockDiscretizationState()

    def compute_step(self, ode, time_discretization_state, step_size):
        return MockDiscretizationState()

    def compute_step_derivative(
        self,
        ode,
        time_discretization_state,
        time_discretization_state_perturbation,
        step_size,
    ):
        return MockDiscretizationState()

    def compute_step_adjoint_derivative(
        self,
        ode,
        time_discretization_state,
        time_discretization_state_perturbation,
        step_size,
    ):
        return MockDiscretizationState()

    def time_discretization_starting_scheme(self, ode, starting_values, step_size):
        return MockDiscretizationState()

    def time_discretization_starting_scheme_derivative(
        self, ode, starting_values, starting_value_perturbations, step_size
    ):
        return MockDiscretizationState()

    def time_discretization_starting_scheme_adjoint_derivative(
        self,
        ode,
        starting_values,
        started_discretization_state_perturbations,
        step_size,
    ):
        return StartingValues(0.0, np.zeros(0), np.zeros(0))

    def time_discretization_finalization_scheme(
        self, ode, discretization_state, step_size
    ):
        return FinalizationValues(0.0, np.zeros(0), np.zeros(0))

    def time_discretization_finalization_scheme_derivative(
        self, ode, discretization_state, discretization_state_perturbations, step_size
    ):
        return FinalizationValues(0.0, np.zeros(0), np.zeros(0))

    def time_discretization_finalization_scheme_adjoint_derivative(
        self, ode, discretization_state, finalization_value_perturbations, step_size
    ):
        return MockDiscretizationState()

    def get_ode_state(self, ode, discretization_state, step_size):
        return DiscretizedODEResultState(
            np.zeros(ode.get_state_size()),
            np.zeros(ode.get_state_size()),
            np.zeros(ode.get_independent_input_size()),
        )

    def get_ode_error_estimate(self, ode, discretization_state, step_size):
        return DiscretizedODEResultState(
            np.zeros(ode.get_state_size()),
            np.zeros(ode.get_state_size()),
            np.zeros(ode.get_independent_input_size()),
        )
