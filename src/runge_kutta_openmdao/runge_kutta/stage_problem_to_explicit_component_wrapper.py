import openmdao.api as om

from runge_kutta_openmdao.utils.problem_to_explicit_component_wrapper import (
    ProblemToExplicitComponentWrapper,
)
from runge_kutta_openmdao.runge_kutta.integration_control import IntegrationControl


class StageProblemToExplicitComponentWrapper(ProblemToExplicitComponentWrapper):
    def initialize(self):
        self.options.declare("mystage", types=int)
        self.options.declare("butcher_time_stage", types=float)
        self.options.declare("butcher_diagonal_element", types=float)
        self.options.declare("integration_control", types=IntegrationControl)

    def compute(self, inputs, outputs):
        integration_control = self.options["integration_control"]
        integration_control.stage = self.options["mystage"]
        integration_control.stage_time = (
            integration_control.step_time_old
            + integration_control.delta_t * self.options["butcher_time_stage"]
        )
        integration_control.butcher_diagonal_element = self.options[
            "butcher_diagonal_element"
        ]
        super().compute(inputs, outputs)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        integration_control = self.options["integration_control"]
        integration_control.stage = self.options["mystage"]
        integration_control.stage_time = (
            integration_control.step_time_old
            + integration_control.delta_t * self.options["butcher_time_stage"]
        )
        integration_control.butcher_diagonal_element = self.options[
            "butcher_diagonal_element"
        ]
        super().compute_jacvec_product(inputs, d_inputs, d_outputs, mode)
