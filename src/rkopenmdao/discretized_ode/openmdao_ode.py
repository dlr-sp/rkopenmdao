import numpy as np
import openmdao.api as om

from .discretized_ode import DiscretizedODE
from ..metadata_extractor import TimeIntegrationMetadata
from ..integration_control import IntegrationControl


class OpenMDAOODE(DiscretizedODE):
    time_stage_problem: om.Problem
    time_integration_metadata: TimeIntegrationMetadata
    integration_control: IntegrationControl

    # time + model in- and outputs
    CacheType = tuple[float, np.ndarray, np.ndarray]

    def compute_update(
        self,
        step_input: np.ndarray,
        stage_input: np.ndarray,
        independent_input: np.ndarray,
        time: float,
        step_size: float,
        stage_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """"""

        self.fill_problem_data(step_input, stage_input, independent_input)
        self.integration_control.stage_time = time
        self.integration_control.butcher_diagonal_element = stage_factor
        self.time_stage_problem.model.run_solve_nonlinear()

        stage_update = np.zeros_like(step_input)
        stage_state = np.zeros_like(
            0
        )  # Currently not used, needs update in metadata_extractor
        independent_output = np.zeros(
            0
        )  # Currently not used, needs update in metadata_extractor

        self.get_problem_data(stage_update, stage_state, independent_output)

        return stage_update, stage_state, independent_output

    def export_linearization(self) -> CacheType:
        inputs, outputs, _ = self.time_stage_problem.model.get_nonlinear_vectors()
        return (
            self.integration_control.stage_time,
            inputs.asarray(copy=True),
            outputs.asarray(copy=True),
        )

    def import_linearization(self, cache: CacheType) -> None:
        self.integration_control.stage_time = cache[0]
        inputs, outputs, _ = self.time_stage_problem.model.get_nonlinear_vectors()
        inputs.asarray()[:] = cache[1]
        outputs.asarray()[:] = cache[2]
