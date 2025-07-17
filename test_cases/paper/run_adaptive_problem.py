"""
Runs an adaptive time integration for the Prothero-Robinson's problem for and computes sensitivity:
1. Prescribed tolerance.
2. Lambda (stiffness) parameter.
5. Safety factor parameter.
6. Error-controller
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
from rkopenmdao.error_controllers import pid, integral, h0_211
from rkopenmdao.error_estimator import SimpleErrorEstimator, ImprovedErrorEstimator
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator

from .odes import ProtheroRobinson


OBJECTIVE_TIME=10.0

butcher_tableaux = [
    second_order_sdirk,
    second_order_esdirk,
    third_order_sdirk,
    third_order_esdirk,
    fourth_order_sdirk,
    fourth_order_esdirk,
]
errors = {}
integration_control = TimeTerminationIntegrationControl(1.0, OBJECTIVE_TIME, 0.0)
test_controller = [integral] # can change to [h0_211] or [h0_211,integral], similarly can be done with pid controller
test_estimator = SimpleErrorEstimator # SimpleErrorEstimator or normalized estimator ImprovedErrorEstimator

exact_solution = ProtheroRobinson.solution(OBJECTIVE_TIME)

for butcher_tableau in butcher_tableaux:
    errors[butcher_tableau.name] = {}

    file_name = f"adaptive_{butcher_tableau.name}"
    file_name = file_name.replace(" ", "_")
    file_name = file_name.replace(",", "")
    file_name = file_name.lower()

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", ProtheroRobinson(integration_control=integration_control,lambda_=-1e2)
    )
    if butcher_tableau.p < 3:
        print(butcher_tableau.name)
        tol = 1e-6
    else:
        tol = 1e-10

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=["x"],
            error_controller=test_controller,
            error_controller_options={"tol": tol, "safety_factor": 0.8},
            error_estimator_type=test_estimator,
            adaptive_time_stepping=True,
            write_file=f"{file_name}.h5",
            write_out_distance=1, # set 0 to not produce .h5 files
        ),
        promotes=["*"],
    )
    runge_kutta_prob.setup()
    runge_kutta_prob["x_initial"] = np.sin(np.pi / 4)

    runge_kutta_prob.run_model()

    errors[butcher_tableau.name]["x"] = (
        np.linalg.norm(runge_kutta_prob["x_final"] - exact_solution[j])
    )
    errors[butcher_tableau.name]["steps"] = integration_control.step

for butcher_tableau in butcher_tableaux:
    print(butcher_tableau.name)
    print(f"No. steps: {errors[butcher_tableau.name]['steps']}")
    print(f"Global Error of x: {errors[butcher_tableau.name]['x']}")
    print("-" * 10)