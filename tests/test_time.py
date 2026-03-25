"""Tests that the RungeKuttaIntegrator/IntegrationControl has the correct time at the
stages/steps."""

import pytest
import openmdao.api as om

from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    runge_kutta_four,
    third_order_four_stage_sdirk,
)
from rkopenmdao.components import ExplicitUnsteadyComponent
from rkopenmdao.termination_criterion import PredefinedNumberOfSteps
from rkopenmdao.integration_config import IntegrationConfig


class DummyComponent(ExplicitUnsteadyComponent):
    """Component that doesn't compute anything, and only checks for the expected times
    in IntegrationControl."""

    _recorded_times: list

    def initialize(self):
        self.options.declare("initial_time", types=float)
        self.options.declare("butcher_tableau", types=ButcherTableau)
        self._recorded_times = []

    def setup(self):
        self.add_input("time", shape=1, tags=["time_variable"])
        self.add_input("dummy_old", tags=["dummy", "step_input_var"])
        self.add_input("dummy_acc", tags=["dummy", "accumulated_stage_var"])
        self.add_output("dummy_stage", tags=["dummy", "stage_output_var"])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        self._recorded_times.append(inputs["time"][0])
        if (
            len(self._recorded_times)
            == self.options["butcher_tableau"].number_of_stages()
        ):
            assert self._recorded_times == pytest.approx(
                self.options["butcher_tableau"].butcher_time_stages
                * self.om_data_exchange.step_size
                + self.options["initial_time"]
            )


@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, runge_kutta_four, third_order_four_stage_sdirk]
)
@pytest.mark.parametrize("initial_time", [0.0, 1.0])
@pytest.mark.parametrize("delta_t", [0.1, 0.01])
def test_integration_control_updating(butcher_tableau, initial_time, delta_t):
    """Tests integration control for step-times."""
    prob = om.Problem()
    prob.model.add_subsystem(
        "dummy",
        DummyComponent(initial_time=initial_time, butcher_tableau=butcher_tableau),
    )
    integration_config = IntegrationConfig(
        use_adaptive_time_stepping=False,
        termination_criterion=PredefinedNumberOfSteps(1),
        initial_step_size=delta_t,
    )

    rk_prob = om.Problem()
    rk_prob.model.add_subsystem(
        "rk_int",
        RungeKuttaIntegrator(
            time_stage_problem=prob,
            time_integration_quantities=["dummy"],
            butcher_tableau=butcher_tableau,
            integration_config=integration_config,
        ),
    )
    rk_prob.setup()
    rk_prob["rk_int.time_initial"] = initial_time
    rk_prob.run_model()
