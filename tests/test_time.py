"""Tests that the RungeKuttaIntegrator/IntegrationControl has the correct time at the
stages/steps."""

import pytest
import openmdao.api as om

from rkopenmdao.integration_control import IntegrationControl
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.butcher_tableau import ButcherTableau
from rkopenmdao.butcher_tableaux import (
    implicit_euler,
    runge_kutta_four,
    third_order_four_stage_sdirk,
)


class DummyComponent(om.ExplicitComponent):
    """Component that doesn't compute anything, and only checks for the expected times
    in IntegrationControl."""

    def initialize(self):
        self.options.declare("integration_control", types=IntegrationControl)
        self.options.declare("butcher_tableau", types=ButcherTableau)

    def setup(self):
        self.add_input("dummy_old", tags=["dummy", "step_input_var"])
        self.add_input("dummy_acc", tags=["dummy", "accumulated_stage_var"])
        self.add_output("dummy_stage", tags=["dummy", "stage_output_var"])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        integration_control: IntegrationControl = self.options["integration_control"]
        butcher_tableau: ButcherTableau = self.options["butcher_tableau"]
        assert integration_control.step_time_old == pytest.approx(
            integration_control.delta_t * (integration_control.step - 1)
            + integration_control.initial_time
        )
        assert integration_control.step_time_new == pytest.approx(
            integration_control.delta_t * integration_control.step
            + integration_control.initial_time
        )
        assert integration_control.stage_time == pytest.approx(
            integration_control.step_time_old
            + integration_control.delta_t
            * butcher_tableau.butcher_time_stages[integration_control.stage]
        )


@pytest.mark.parametrize(
    "butcher_tableau", [implicit_euler, runge_kutta_four, third_order_four_stage_sdirk]
)
@pytest.mark.parametrize("initial_time", [0.0, 1.0])
@pytest.mark.parametrize("delta_t", [0.1, 0.01])
def test_integration_control_updating(butcher_tableau, initial_time, delta_t):
    """Tests integration control for step/stage-times."""
    integration_control = IntegrationControl(initial_time, 10, delta_t)
    prob = om.Problem()
    prob.model.add_subsystem(
        "dummy",
        DummyComponent(
            integration_control=integration_control, butcher_tableau=butcher_tableau
        ),
    )

    rk_prob = om.Problem()
    rk_prob.model.add_subsystem(
        "rk_int",
        RungeKuttaIntegrator(
            time_stage_problem=prob,
            time_integration_quantities=["dummy"],
            integration_control=integration_control,
            butcher_tableau=butcher_tableau,
        ),
    )
    rk_prob.setup()
    rk_prob.run_model()
