"""
Runs an adaptive time integration for the Kaps' problem for and computes sensitivity:
1. Prescribed tolerance.
2. Epsilon (stiffness) parameter.
3. Excluded quantity (y_1, y_2)
4. L-norm order (np.inf, 2)
5. Safety factor parameter.
6. Error-controller
For sensitivity set write_out_distance=0 and remove the '#' before runge_kutta_prob.check_partials()
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
from rkopenmdao.checkpoint_interface.all_checkpointer import AllCheckpointer
from rkopenmdao.error_controllers import pid, integral, h0_211
from rkopenmdao.error_estimator import SimpleErrorEstimator, ImprovedErrorEstimator
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.utils.convergence_test_components import KapsGroup, kaps_solution


OBJECTIVE_TIME=1.0

butcher_tableaux = [
    second_order_sdirk,
    second_order_esdirk,
    third_order_sdirk,
    third_order_esdirk,
    fourth_order_sdirk,
    fourth_order_esdirk,
]
integration_control = TimeTerminationIntegrationControl(0.1, OBJECTIVE_TIME, 0.0)
errors = {}
test_controller = [integral] # can change to [pid] or [pid,integral], similarly can be done with h0_211 controller
test_estimator = SimpleErrorEstimator # SimpleErrorEstimator or normalized estimator ImprovedErrorEstimator
quantities_base = ["y_1", "y_2"]
quantities = quantities_base + ["y_2_integral"]

exact_solution = kaps_solution(OBJECTIVE_TIME)

for butcher_tableau in butcher_tableaux:
    errors[butcher_tableau.name] = {}

    file_name = f"adaptive_{butcher_tableau.name}"
    file_name = file_name.replace(" ", "_")
    file_name = file_name.replace(",", "")
    file_name = file_name.lower()

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", KapsGroup(integration_control=integration_control, epsilon=1.0) # set epsilon = 1.e-3 or 1.0
    )
    tol = 1e-6

    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            checkpointing_type=AllCheckpointer,
            integration_control=integration_control,
            time_integration_quantities=quantities,
            error_controller=test_controller,
            error_controller_options={"tol": tol, "safety_factor": 0.95},
            error_estimator_type=test_estimator,
            error_estimator_options={"order": np.inf}, #, "exclude": [y_1], "exclude": [y_2] or leave empty
            adaptive_time_stepping=True,
            write_file=f"{file_name}.h5",
            write_out_distance=1,   # set 0 to not produce .h5 files
            time_independent_input_quantities=["a"],
        ),
        promotes=["*"],
    )
    runge_kutta_prob.setup()

    for quantity in quantities:
        runge_kutta_prob[quantity + "_initial"].fill(1.0)
    runge_kutta_prob["y_2_integral_initial"].fill(0.0)
    runge_kutta_prob["a"] = 1.0

    runge_kutta_prob.run_model()
    # runge_kutta_prob.check_partials()

    for j, var in enumerate(quantities_base):
        errors[butcher_tableau.name][var] = (
            np.linalg.norm(runge_kutta_prob[var + "_final"] - exact_solution[j])
        )
        errors[butcher_tableau.name]["steps"] = integration_control.step

for butcher_tableau in butcher_tableaux:
    print(butcher_tableau.name)
    print(f"No. steps: {errors[butcher_tableau.name]['steps']}")
    for quantity in quantities_base:
        print(f"Global Error of {quantity}: {errors[butcher_tableau.name][var]}")
    print("-" * 10)