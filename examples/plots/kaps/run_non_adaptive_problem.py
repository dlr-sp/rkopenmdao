"""
Runs a homogeneous time integration for the Kaps' problem for:
1. Time step size delta_t,
2. Epsilon (stiffness) parameter
4. L-norm order (np.inf, 2) (For different local error plot types L_2 or L_\infty)
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
from rkopenmdao.utils.convergence_test_components import KapsGroup, kaps_solution


OBJECTIVE_TIME = 1.0

def component_integration(component_class, dt, butcher_tableau, quantities):
    """
    Integrates the component with the Runge-kutta Integrator for a given step size
    """
    write_file = f"data_{dt:.0E}_{butcher_tableau.name}"
    write_file = write_file.replace(" ", "_")
    write_file = write_file.replace(",", "")
    write_file = write_file.lower()

    integration_control = TimeTerminationIntegrationControl(dt, OBJECTIVE_TIME, 0.0)
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp",
        component_class(integration_control=integration_control, epsilon=1.0), # set epsilon = 1.e-3 or 1.0
    )
    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integrator",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=quantities,
            error_controller=[pseudo],
            adaptive_time_stepping=True,
            write_file=f"{write_file}.h5",
            write_out_distance=1,
        ),
        promotes=["*"],
    )

    runge_kutta_prob.setup()

    for quantity in quantities:
        runge_kutta_prob[quantity + "_initial"].fill(1.0)
    try:
        runge_kutta_prob.run_model()
    except om.AnalysisError:
        print("Convergence failed")
        (
            inputs,
            outputs,
            residuals,
        ) = time_integration_prob.model.get_nonlinear_vectors()
        for var in inputs:
            print(var, time_integration_prob[var])
        for var in outputs:
            print(var, time_integration_prob[var])
            print(
                var + "residual",
                residuals[var],
            )

    for j, var in enumerate(quantities):
        exact_solution = kaps_solution(1.0)
        print(
            "Delta t:",
            dt,
            "\n Norm:",
            np.linalg.norm(runge_kutta_prob[var + "_final"] - exact_solution[j]),
        )


if __name__ == "__main__":
    butcher_tableaux = [
        second_order_sdirk,
        second_order_esdirk,
        third_order_sdirk,
        third_order_esdirk,
        fourth_order_sdirk,
        fourth_order_esdirk,
    ]
    delta_t = [
        1e-2,
        2e-2,
        4e-2,
        5e-2,
        1e-1,
    ] # Commits the step sizes for all schemes
    delta_t = np.array(delta_t)
    error_data = {}

    for scheme in butcher_tableaux:
        error_data[f"{scheme.name}"] = []
        for step_size in delta_t:
            error_data[f"{scheme.name}"].append(
                component_integration(
                    KapsGroup, step_size, scheme, ["y_1", "y_2"]
                )
            )