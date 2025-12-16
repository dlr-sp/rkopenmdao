"""
Runs a homogeneous time integration for the Prothero-Robinson's problem for:
1. Number of steps,
2. Lambda (stiffness) parameter
"""

import numpy as np

import openmdao.api as om
from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as second_order_sdirk,
    embedded_second_order_three_stage_esdirk as second_order_esdirk,
    embedded_third_order_three_stage_sdirk as third_order_sdirk,
    embedded_third_order_four_stage_esdirk as third_order_esdirk,
    embedded_fourth_order_four_stage_sdirk as fourth_order_sdirk,
    embedded_fourth_order_five_stage_esdirk as fourth_order_esdirk,
)
from rkopenmdao.error_controllers import pseudo
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator

from .odes import ProtheroRobinson


OBJECTIVE_TIME=10.0

def component_integration(component_class, dt, butcher_tableau, quantity):
    """
    Integrates the component with the Runge-kutta Integrator for a given step size
    """
    write_file = f"homogeneous_{butcher_tableau.name}"
    write_file = write_file.replace(" ", "_")
    write_file = write_file.replace(",", "")
    write_file = write_file.lower()

    integration_control = TimeTerminationIntegrationControl(dt, OBJECTIVE_TIME, 0.0)
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", component_class(integration_control=integration_control, lambda_=-1e2)
    )
    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=[quantity],
            error_controller=[pseudo],
            adaptive_time_stepping=True,
            write_file=f"{write_file}.h5",
            write_out_distance=1, # set 0 to not produce .h5 files
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()

    runge_kutta_prob[quantity + "_initial"] = np.sin(np.pi / 4)
    runge_kutta_prob.run_model()


if __name__ == "__main__":
    butcher_tableaux = [
        second_order_sdirk,
        second_order_esdirk,
        third_order_sdirk,
        third_order_esdirk,
        fourth_order_sdirk,
        fourth_order_esdirk,
    ]
    steps = [3727, 3263, 29554, 5518, 17747, 4867]  # No. Steps in order of butcher_tableaux's scheme
    delta_t = [OBJECTIVE_TIME/step for step in steps]
    delta_t = np.array(delta_t)
    error_data = {}

    for i, scheme in enumerate(butcher_tableaux):
        error_data[f"{scheme.name}"] = []
        error_data[f"{scheme.name}"].append(
            component_integration(ProtheroRobinson, delta_t[i], scheme, "x")
        )