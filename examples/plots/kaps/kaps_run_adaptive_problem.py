"""
Generates a graph of the delta time over time of different adaptive Runge-Kutta methods.
"""

from mpi4py import MPI
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
from rkopenmdao.file_writer import TXTFileWriter
from rkopenmdao.integration_control import (
    TimeTerminationIntegrationControl,
)
from rkopenmdao.runge_kutta_integrator import RungeKuttaIntegrator
from rkopenmdao.utils.convergence_test_components import KapsGroup, kaps_solution


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

WRITE_FILE = "adaptive_test"

integration_control = TimeTerminationIntegrationControl(0.1, 1.0, 0.0)
butcher_tableaux = [
    # second_order_sdirk,
    # second_order_esdirk,
    third_order_sdirk,
    # third_order_esdirk,
    # fourth_order_sdirk,
    # fourth_order_esdirk,
]
test_controller = [integral]
test_estimator = SimpleErrorEstimator
errors = {}
quantities = ["y_1", "y_2"]

for butcher_tableau in butcher_tableaux:
    errors[butcher_tableau.name] = {}
    file_name = f"{WRITE_FILE}_{butcher_tableau.name}"
    file_name = file_name.replace(" ", "_")
    file_name = file_name.replace(",", "")
    file_name = file_name.lower()

    time_integration_prob = om.Problem()
    time_integration_prob.model.add_subsystem(
        "test_comp", KapsGroup(integration_control=integration_control, epsilon=1.0)
    )

    if butcher_tableau.p < 3:
        print(butcher_tableau.name)
        tol = 1e-6
    else:
        tol = 1e-6
    runge_kutta_prob = om.Problem()
    runge_kutta_prob.model.add_subsystem(
        "rk_integration",
        RungeKuttaIntegrator(
            time_stage_problem=time_integration_prob,
            butcher_tableau=butcher_tableau,
            integration_control=integration_control,
            time_integration_quantities=quantities,
            error_controller=test_controller,
            error_controller_options={"tol": tol, "safety_factor": 0.95},
            error_estimator_type=test_estimator,
            error_estimator_options={"order": np.inf, "exclude": ["y_2"]},
            adaptive_time_stepping=True,
            write_file=f"{file_name}.h5",
            write_out_distance=0,
        ),
        promotes=["*"],
    )
    runge_kutta_prob.setup()

    for quantity in quantities:
        runge_kutta_prob[quantity + "_initial"].fill(1.0)

    runge_kutta_prob.run_model()
    for j, var in enumerate(quantities):
        exact_solution = kaps_solution(1.0)
        errors[butcher_tableau.name][var] = {
            np.linalg.norm(runge_kutta_prob[var + "_final"] - exact_solution[j])
        }
        errors[butcher_tableau.name]["steps"] = integration_control.step
comm.Barrier()
if comm.rank == 0:
    for butcher_tableau in butcher_tableaux:
        print(butcher_tableau.name)
        print(f" {errors[butcher_tableau.name]['steps']}")
        for quantity in quantities:
            print(f" {quantity}: {errors[butcher_tableau.name][var]}")
        print("-" * 10)
