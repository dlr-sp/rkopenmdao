"""
Generates a logarithmic graph of error over time for Runge-Kutta methods of different
orders.
"""

import argparse

import numpy as np

import openmdao.api as om
from mpi4py import MPI

from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.error_controllers import pseudo
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.butcher_tableaux import (
    embedded_second_order_two_stage_sdirk as two_stage_sdirk,
    embedded_third_order_four_stage_esdirk as four_stage_esdirk,
    embedded_fourth_order_five_stage_esdirk as five_stage_esdirk,
)
from .odes import ODE_CFD

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def component_integration(component_class, dt, butcher_tableau, quantities, args):
    """
    Integrates the component with the Runge-kutta Integrator for a given step size
    """
    write_file = f"{args.base_name}_{dt:.0E}_{butcher_tableau.name}"
    write_file = write_file.replace(" ", "_")
    write_file = write_file.replace(",", "")
    write_file = write_file.lower()

    initial_values = np.array([1])
    integration_control = TimeTerminationIntegrationControl(dt, 10.0, 0.0)
    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", component_class(integration_control=integration_control)
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

    for i, quantity in enumerate(quantities):
        runge_kutta_prob[quantity + "_initial"] = initial_values[i]
    runge_kutta_prob.run_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_name", default="data", type=str)
    parsed_args = parser.parse_args()

    delta_t = [
        1.0e-4,
        0.5e-3,
        1e-3,
        0.5e-2,
        1e-2,
        0.5e-1,
        1e-1,
        0.5,
        1.0,
    ]
    delta_t = np.array(delta_t)
    error_data = {}
    butcher_tableaux = [
        two_stage_sdirk,
        four_stage_esdirk,
        five_stage_esdirk,
    ]
    for scheme in butcher_tableaux:
        error_data[f"{scheme.name}"] = []
        for step_size in delta_t:
            error_data[f"{scheme.name}"].append(
                component_integration(ODE_CFD, step_size, scheme, ["x"], parsed_args)
            )
